// src/directx/transparent.rs
//
// The engine's `PassId::Transparent` slot on the D3D12 backend: one pass, drawn
// after the SSR resolve and before TAA, with three producers -- glass panes
// (`glass.rs`), water surfaces (`water.rs`), and see-through glass meshes
// (`glass.rs` as well, since a mesh is the same material family as a pane). The
// pass snapshots the pre-transparent scene, orders every record of every
// producer back-to-front by camera distance, and draws them into the post-SSR
// scene target with straight-alpha blending.
//
// The pane and water producers contribute records built once at init. The mesh
// producer cannot: a see-through mesh draws from the SHARED scene vertex / index
// buffers at the offsets its `DrawObject` carries, and both those offsets (LOD
// picks per frame) and its params (model matrix, material tint) change at
// runtime. So it owns only its pipelines plus a per-frame params ring, and the
// encoder rebuilds its draw list each frame.
//
// One pass rather than one per producer, mirroring the Metal backend: the scene
// snapshot the refraction taps is a full render-resolution HDR image and a copy
// of it every frame, so a second one would be pure waste; and a single ordering
// over both producers is what puts a pane standing in a pool on the correct side
// of the water.
//
// The producers also share their root signatures, because `glass.slang`,
// `water.slang` and `glass_mesh.slang` declare the same registers on purpose.
// There are two: the base
// signature (probe / planar reflection) and the RT one, whose ray-tracing SRVs
// at t4..t10 push the probe cube array to t20. Which one runs is a per-frame
// choice, not a per-producer one -- see `DxContext::rt_transparent_active`.
//
// The shaders are the shared `shaders/{glass,glass_mesh,water}.slang`, compiled
// through `slang_builtins`; the ray-traced fragments need shader model 6.5 for
// their inline ray query, the base pairs 6.0. The mesh producer is ray-traced
// only -- the per-pixel trace is what makes it see-through rather than the
// opaque reflective glass the main pass draws -- so it runs only under the RT
// root signature and is inert while RT is off.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use crate::components::{GlassPanel, WaterSurface};
use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::pipeline::{main_input_layout, serialize_desc_and_create};
use crate::directx::texture::{
    HDR_FORMAT, create_buffer, create_hdr_resolve_target, transition_barrier, upload_buffer,
};
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::RtParams;
use crate::gfx::rt_reflections::RtParamsInputs;

// RtParams push size (144 B; see gfx::render_types::RtParams), shared with the
// RT-reflection resolve.
const RT_PARAMS_UBO_SIZE: u64 = 144;

// `TransparentView` (the per-frame view cbuffer) is a GPU-free layout struct
// that lives in `core::render`; re-export it so the encode path and the
// graph's view builder can keep naming it through this module.
use concinnity_core::render::uniforms::GlassMeshParams;
pub(in crate::directx) use concinnity_core::render::uniforms::TransparentView;

// Which producer a record belongs to, and so which pipeline draws it. The
// records themselves are identical in shape, so this is the only thing the
// combined draw loop needs to tell them apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::directx) enum Producer {
    Glass,
    Water,
    GlassMesh,
}

// One drawable record of either producer: a static world-space (glass) or
// origin-centred grid (water) VB + IB plus a per-record uniform CBV. Both are
// built once at init and never change at runtime, so there is no per-frame work
// beyond projection.
pub(in crate::directx) struct TransparentRecord {
    #[expect(
        dead_code,
        reason = "held to keep the GPU memory alive; the encoder binds through vertex_buffer_view"
    )]
    vertex_buffer: PooledBuffer,
    vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    #[expect(
        dead_code,
        reason = "held to keep the GPU memory alive; the encoder binds through index_buffer_view"
    )]
    index_buffer: PooledBuffer,
    index_buffer_view: D3D12_INDEX_BUFFER_VIEW,
    index_count: u32,
    #[expect(
        dead_code,
        reason = "held to keep the GPU memory alive; the encoder binds through params_cbuffer_gva"
    )]
    params_cbuffer: PooledBuffer,
    params_cbuffer_gva: u64,
    visible: bool,
    // World-space centre, used for the back-to-front camera-distance sort.
    centre: [f32; 3],
    // Planar reflection resolve slot this record samples (index into the
    // `PlanarReflectionSet`). `None` when the world has no planar set or this
    // record's plane overflowed the budget; the shader then keeps the probe/sky
    // path. Assigned at init by `assign_planar_slots`.
    planar_slot: Option<usize>,
}

// The geometry, uniform payload and per-record state one producer hands over for
// a `TransparentRecord`. Keeps the buffer uploads in one place instead of once
// per producer.
pub(in crate::directx) struct RecordUpload<'a> {
    pub vertices: &'a [Vertex],
    pub indices: &'a [u16],
    pub params: &'a [u8],
    pub visible: bool,
    pub centre: [f32; 3],
    pub planar_slot: Option<usize>,
}

impl TransparentRecord {
    // Upload one record's static geometry and its per-record params CBV. The
    // params buffer is persistently mapped and written once here: nothing in the
    // record changes after init.
    pub(in crate::directx) fn upload(
        alloc: &DeviceAllocator,
        upload: RecordUpload<'_>,
    ) -> Result<Self, String> {
        let vbytes = bytemuck::cast_slice(upload.vertices);
        let ibytes = bytemuck::cast_slice(upload.indices);
        let vertex_buffer = upload_buffer(
            alloc,
            vbytes,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
        )?;
        let index_buffer = upload_buffer(alloc, ibytes, D3D12_RESOURCE_STATE_INDEX_BUFFER)?;
        let vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&vertex_buffer),
            SizeInBytes: vbytes.len() as u32,
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        };
        let index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&index_buffer),
            SizeInBytes: ibytes.len() as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };

        let params_cbuffer = create_buffer(
            alloc,
            align256(upload.params.len() as u64),
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut p = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { params_cbuffer.Map(0, None, Some(&mut p)) }
            .map_err(|e| format!("map transparent params cb: {e}"))?;
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the
        // source is a separate allocation, so the ranges cannot overlap. Persistently mapped,
        // never unmapped.
        unsafe {
            std::ptr::copy_nonoverlapping(upload.params.as_ptr(), p as *mut u8, upload.params.len())
        };
        let params_cbuffer_gva = com::gpu_va(&params_cbuffer);

        Ok(Self {
            vertex_buffer,
            vertex_buffer_view,
            index_buffer,
            index_buffer_view,
            index_count: upload.indices.len() as u32,
            params_cbuffer,
            params_cbuffer_gva,
            visible: upload.visible,
            centre: upload.centre,
            planar_slot: upload.planar_slot,
        })
    }
}

// One producer's pipelines plus its records. The RT pair is `Some` only when the
// GPU supports DXR and the DXC compile succeeded; a live RT toggle then selects
// them with no rebuild.
pub(in crate::directx) struct TransparentProducer {
    pub pso: ID3D12PipelineState,
    pub flat_rt_pso: Option<ID3D12PipelineState>,
    pub textured_rt_pso: Option<ID3D12PipelineState>,
    pub records: Vec<TransparentRecord>,
}

impl TransparentProducer {
    // Pick this producer's pipeline for the frame: the sharp per-pixel trace when
    // RT is live, the textured variant when the bindless pool exists as well, and
    // the probe / planar pipeline otherwise.
    //
    // The `expect`s are the point rather than an inconvenience: the base PSO is
    // built against the base root signature and the RT pair against the RT one,
    // so falling back across the two would bind a PSO under a signature it was
    // not built for, which is a D3D12 error rather than a degraded frame. Both
    // choices are whole-pass decisions the encoder makes from
    // `rt_pipelines_ready` / `rt_textured_ready`, which require the pipeline of
    // every live producer -- so a producer is never asked for one it lacks.
    fn pipeline(&self, rt_live: bool, textured: bool) -> &ID3D12PipelineState {
        match (rt_live, textured) {
            (true, true) => self
                .textured_rt_pso
                .as_ref()
                .expect("rt_textured_ready gated the frame on every producer's textured PSO"),
            (true, false) => self
                .flat_rt_pso
                .as_ref()
                .expect("rt_pipelines_ready gated the frame on every producer's flat RT PSO"),
            _ => &self.pso,
        }
    }
}

// The see-through glass MESH producer. Ray-traced only: what makes the mesh
// see-through rather than the opaque reflective glass of the main pass is a real
// per-pixel reflection ray, so there is no probe-path pipeline and the whole
// producer is inert while RT is off (those meshes then render opaque).
//
// It holds no records. A mesh draws from the shared scene vertex / index buffers
// at its `DrawObject`'s offsets, and both those offsets and its params change per
// frame, so `collect_mesh_draws` rebuilds the list every frame and writes each
// mesh's params into this frame's slice of the ring.
pub(in crate::directx) struct GlassMeshProducer {
    flat_rt_pso: ID3D12PipelineState,
    // `Some` only when the bindless pool exists, matching the other producers.
    textured_rt_pso: Option<ID3D12PipelineState>,
    // Indices into `DxContext::draw.objects` of every see-through mesh,
    // precomputed at init so the per-frame collect does not rescan all objects.
    // The objects stay IN `draw.objects` -- a slot is a key into the cull /
    // prev-model / RT parallel arrays -- this only marks which to reroute.
    object_indices: Vec<usize>,
    // Per-frame params ring: one 256-aligned `GlassMeshParams` block per mesh per
    // frame slot, persistently mapped. Sized at init from `object_indices`, which
    // never grows (a material edit can only flip an existing slot's flag).
    params_ring: Vec<PooledBuffer>,
    params_ptrs: Vec<*mut u8>,
}

// One see-through mesh's draw for this frame: the shared-buffer slice its
// `DrawObject` resolved to, the GPU address of its params block in this frame's
// ring, and its world-space centre for the back-to-front sort.
struct GlassMeshDraw {
    index_offset: u32,
    index_count: u32,
    base_vertex: i32,
    params_gva: u64,
    centre: [f32; 3],
}

impl GlassMeshProducer {
    // Allocate the per-frame params ring (one 256-aligned block per mesh per
    // frame slot, persistently mapped) and take ownership of the pipelines.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        flat_rt_pso: ID3D12PipelineState,
        textured_rt_pso: Option<ID3D12PipelineState>,
        object_indices: Vec<usize>,
    ) -> Result<Self, String> {
        let block = align256(std::mem::size_of::<GlassMeshParams>() as u64);
        let ring_size = block * object_indices.len().max(1) as u64;
        let mut params_ring: Vec<PooledBuffer> = Vec::with_capacity(FRAMES);
        let mut params_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                alloc,
                ring_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map glass mesh params ring: {e}"))?;
            params_ptrs.push(ptr as *mut u8);
            params_ring.push(buf);
        }
        Ok(Self {
            flat_rt_pso,
            textured_rt_pso,
            object_indices,
            params_ring,
            params_ptrs,
        })
    }

    // Pick the frame's pipeline. Same all-or-nothing gate as the other
    // producers: `rt_textured_ready` requires every live producer's textured
    // pipeline, so this is never asked for one it lacks.
    fn pipeline(&self, textured: bool) -> &ID3D12PipelineState {
        match textured {
            true => self
                .textured_rt_pso
                .as_ref()
                .expect("rt_textured_ready gated the frame on every producer's textured PSO"),
            false => &self.flat_rt_pso,
        }
    }
}

// Owned by `DxContext` when the world declared any `GlassPanel`, `WaterSurface`
// or see-through material. Holds both root signatures, the per-frame view +
// RtParams CBV rings, the scene snapshot the fragments refract, and each live
// producer. The
// depth SRV is the main-pass depth slot shared with the decal pass; the
// scene-copy SRV is the transparent pass's own heap slot.
pub(in crate::directx) struct TransparentResources {
    root_sig: ID3D12RootSignature,
    glass: Option<TransparentProducer>,
    water: Option<TransparentProducer>,
    glass_mesh: Option<GlassMeshProducer>,

    // Per-frame view UBO (single 160-byte block), persistently mapped.
    view_ubo_resources: Vec<PooledBuffer>,
    view_ubo_ptrs: Vec<*mut u8>,

    // Pre-transparent scene snapshot. `encode_transparent` copies the scene
    // target into this each frame before the draws so refraction reads a stable
    // copy instead of the attachment being written.
    scene_copy: ID3D12Resource,
    scene_copy_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    scene_copy_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // Main-depth SRV (shared with the decal pass); bound at t1 for the manual
    // occlusion test.
    depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,

    // Ray-traced reflection state, present only when the GPU supports DXR AND
    // every RT pipeline compiled. One root signature shared by both producers'
    // flat-tint + textured-bindless PSOs, chosen per frame in
    // `encode_transparent` while RT is live; otherwise the base `pso`s run the
    // probe/planar path. The RtParams ring feeds the trace and is empty (never
    // read) when the RT pipelines are absent.
    rt_root_sig: Option<ID3D12RootSignature>,
    rt_params_ubo_resources: Vec<PooledBuffer>,
    rt_params_ubo_ptrs: Vec<*mut u8>,
}

// The mapped ring pointers are POD raw pointers; the upload buffers stay alive
// through the `Vec<PooledBuffer>` fields and the pointers are written on the
// render thread only. Mirrors `RaymarchResources`.
// SAFETY: the raw pointers `TransparentResources` holds are the mappings of upload buffers the
// struct also owns, so they stay valid for as long as it does. They are only values here: every
// dereference goes through a `&mut self` method on the context, which the main-thread guard keeps
// on the render thread.
unsafe impl Send for TransparentResources {}
// SAFETY: sharing `&TransparentResources` hands out the pointer values but no way to dereference
// them; every write goes through a `&mut self` method on the context.
unsafe impl Sync for TransparentResources {}

// World-space distance from the camera to a record centre. Larger = farther =
// drawn first. Pure; unit tested.
fn sort_distance(centre: [f32; 3], cam: [f32; 3]) -> f32 {
    let dx = centre[0] - cam[0];
    let dy = centre[1] - cam[1];
    let dz = centre[2] - cam[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// Every visible record of every producer, ordered farthest-camera-distance
// first. Pure; unit tested. Invisible records are excluded, and the producers
// interleave so a pane standing in a pool composites in the right order; the
// visible set is sorted via the shared `gfx::transparent::back_to_front_order`.
//
// The mesh slice is already filtered to this frame's visible meshes (the encoder
// builds it), so every entry it carries is live.
fn ordered_visible(
    glass: &[([f32; 3], bool)],
    water: &[([f32; 3], bool)],
    meshes: &[[f32; 3]],
    cam: [f32; 3],
) -> Vec<(Producer, usize)> {
    let live_of = |records: &[([f32; 3], bool)], kind: Producer| -> Vec<(Producer, usize)> {
        records
            .iter()
            .enumerate()
            .filter(|(_, (_, vis))| *vis)
            .map(|(i, _)| (kind, i))
            .collect()
    };
    let live: Vec<(Producer, usize)> = live_of(glass, Producer::Glass)
        .into_iter()
        .chain(live_of(water, Producer::Water))
        .chain((0..meshes.len()).map(|i| (Producer::GlassMesh, i)))
        .collect();
    let dists: Vec<f32> = live
        .iter()
        .map(|&(kind, i)| {
            let centre = match kind {
                Producer::Glass => glass[i].0,
                Producer::Water => water[i].0,
                Producer::GlassMesh => meshes[i],
            };
            sort_distance(centre, cam)
        })
        .collect();
    crate::gfx::transparent::back_to_front_order(&dists)
        .into_iter()
        .map(|oi| live[oi])
        .collect()
}

// Root-signature layout (binds 1:1 with the `DXIL_ABI` declarations in
// glass.slang and water.slang, which are deliberately identical):
//   [0] root CBV b0   TransparentView (per-frame)
//   [1] root CBV b1   GlassParams / WaterParams (per-record)
//   [2] table  t0     scene-copy SRV  (Texture2D<float4>)
//   [3] table  t1     scene depth SRV (Texture2D[MS]<float>)
//   [4] table  t2     sky prefilter cube SRV
//   [5] table  t7..   reflection-probe cube array
//   [6] root CBV b4   ProbeSet
//   [7] table  t3     planar reflection resolve SRV (per record)
//   static sampler s0 : linear clamp ; s2 : cube mip-linear clamp
//
// b1 is visible to every stage: the water vertex stage reads its wave table out
// of the params block, where the glass vertex stage reads only the view.
// Root parameter index of the per-record planar resolve table. It is the last
// parameter of both signatures, so each builder asserts its own length against
// the constant rather than the encoder repeating a literal that can drift.
const PLANAR_ROOT_BASE: u32 = 7;
const PLANAR_ROOT_RT: u32 = 15;

fn create_transparent_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let scene_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let depth_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 1, // t1
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // t2: the sky IBL prefilter cube (the reflection fallback where no probe covers).
    let prefilter_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 2, // t2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // t7..t7+MAX_PROBES: the reflection-probe cube array. Unbaked
    // slots hold the sky prefilter, so a sample at any index is valid; box-projected
    // when ProbeSet.count > 0.
    let probe_cube_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: concinnity_core::render::uniforms::MAX_PROBES as u32,
        BaseShaderRegister: 7, // t7..
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // t3: this record's planar reflection resolve (the sharp mirror render), bound
    // per record. A valid SRV is always bound (the scene snapshot stands in for
    // records with no planar slot); the shader only samples it when `planar > 0.5`.
    let planar_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 3, // t3
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let cbv = |reg: u32, vis: D3D12_SHADER_VISIBILITY| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: reg,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: vis,
    };
    let table = |range: &D3D12_DESCRIPTOR_RANGE| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };
    let params = [
        cbv(0, D3D12_SHADER_VISIBILITY_ALL),   // [0] b0 TransparentView
        cbv(1, D3D12_SHADER_VISIBILITY_ALL),   // [1] b1 per-record params
        table(&scene_range),                   // [2] t0 scene copy
        table(&depth_range),                   // [3] t1 depth
        table(&prefilter_range),               // [4] t2 prefilter cube
        table(&probe_cube_range),              // [5] t7.. probe cubes
        cbv(4, D3D12_SHADER_VISIBILITY_PIXEL), // [6] b4 ProbeSet
        table(&planar_range),                  // [7] t3 planar resolve
    ];
    debug_assert_eq!(params.len() as u32 - 1, PLANAR_ROOT_BASE);
    // s0: linear-clamp for the scene snapshot / depth. s2: cube mip-linear clamp for
    // the prefilter + probe cube array.
    let samp = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        ..Default::default()
    };
    let cube_samp = D3D12_STATIC_SAMPLER_DESC {
        ShaderRegister: 2, // s2
        ..samp
    };
    let samplers = [samp, cube_samp];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    serialize_desc_and_create(device, &desc, "transparent root sig")
}

// PSO for a transparent producer. Writes the single-sample post-SSR scene target
// with src-alpha / inv-src-alpha blending. No depth attachment (the fragment
// shader does the manual occlusion test) and no face culling (both shaders are
// two-sided). Standard 5-attribute vertex layout shared with the main pass.
pub(in crate::directx) fn create_transparent_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let layout = main_input_layout();
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: com::borrowed(root_sig),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as _,
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr() as _,
            BytecodeLength: ps.len(),
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = HDR_FORMAT;
            a
        },
        DSVFormat: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthClipEnable: false.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            StencilEnable: false.into(),
            ..Default::default()
        },
        BlendState: D3D12_BLEND_DESC {
            RenderTarget: {
                let mut arr = [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8];
                arr[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: true.into(),
                    SrcBlend: D3D12_BLEND_SRC_ALPHA,
                    DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_SRC_ALPHA,
                    DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOpAlpha: D3D12_BLEND_OP_ADD,
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                arr
            },
            ..Default::default()
        },
        ..Default::default()
    };
    // SAFETY: `desc` outlives this synchronous call, and so do the root signature, shader bytecode
    // and input-element array whose raw pointers it borrows.
    unsafe { crate::directx::pso_library::create_graphics(device, &pso_desc) }
        .map_err(|e| format!("create transparent PSO: {e}"))
}

// Root signature for the RT PSOs (binds 1:1 with the `DXIL_ABI` declarations
// under GLASS_RT / WATER_RT):
//   [0]  root CBV b0   TransparentView (per-frame, vertex + pixel)
//   [1]  root CBV b1   GlassParams / WaterParams (per-record, every stage)
//   [2]  table  t0     scene-copy SRV
//   [3]  table  t1     scene depth SRV
//   [4]  table  t2     sky prefilter cube SRV
//   [5]  table  t20..  reflection-probe cube array (remapped off t7)
//   [6]  root CBV b4   ProbeSet
//   [7]  root CBV b5   RtParams
//   [8]  root SRV t4   scene TLAS
//   [9]  root SRV t5   vertex buffer (raw)
//   [10] root SRV t6   index buffer (u32, raw)
//   [11] root SRV t10  geometry table (structured)
//   [12] root SRV t8   deformed skinned verts (raw)
//   [13] root SRV t9   skinned indices (raw)
//   [14] table  t0,sp1 bindless texture pool (textured PSOs only)
//   [15] table  t3     this record's planar reflection resolve
//   static samplers s0 linear-clamp, s1 linear-repeat, s2 cube linear-clamp
//
// [15] is what the base signature carries at [7]: the water fragment samples its
// mirror plane in place of tracing wherever it has one, so the resolve has to
// reach the RT PSOs too. Glass never reads t3 on this path; the encoder still
// binds the table for every draw so no PSO runs under an unset root parameter.
fn create_transparent_rt_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let table_range = |reg: u32, space: u32, count: u32| D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: count,
        BaseShaderRegister: reg,
        RegisterSpace: space,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let scene_range = table_range(0, 0, 1); // t0
    let depth_range = table_range(1, 0, 1); // t1
    let prefilter_range = table_range(2, 0, 1); // t2
    let probe_cube_range = table_range(20, 0, concinnity_core::render::uniforms::MAX_PROBES as u32);
    let planar_range = table_range(3, 0, 1); // t3
    let pool_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: u32::MAX, // unbounded bindless pool
        BaseShaderRegister: 0,    // t0
        RegisterSpace: 1,         // space1
        OffsetInDescriptorsFromTableStart: 0,
    };

    let cbv = |reg: u32, vis: D3D12_SHADER_VISIBILITY| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: reg,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: vis,
    };
    let root_srv = |reg: u32| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: reg,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };
    let table = |range: &D3D12_DESCRIPTOR_RANGE| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };

    let params = [
        cbv(0, D3D12_SHADER_VISIBILITY_ALL), // [0] b0 TransparentView (vertex reads vp)
        cbv(1, D3D12_SHADER_VISIBILITY_ALL), // [1] b1 per-record params (water's vertex reads it)
        table(&scene_range),                 // [2] t0 scene copy
        table(&depth_range),                 // [3] t1 depth
        table(&prefilter_range),             // [4] t2 prefilter cube
        table(&probe_cube_range),            // [5] t20.. probe cubes
        cbv(4, D3D12_SHADER_VISIBILITY_PIXEL), // [6] b4 ProbeSet
        cbv(5, D3D12_SHADER_VISIBILITY_PIXEL), // [7] b5 RtParams
        root_srv(4),                         // [8] t4 TLAS
        root_srv(5),                         // [9] t5 verts
        root_srv(6),                         // [10] t6 indices
        root_srv(10),                        // [11] t10 geom table
        root_srv(8),                         // [12] t8 skinned verts
        root_srv(9),                         // [13] t9 skinned indices
        table(&pool_range),                  // [14] t0,space1 bindless pool
        table(&planar_range),                // [15] t3 planar resolve
    ];
    debug_assert_eq!(params.len() as u32 - 1, PLANAR_ROOT_RT);

    let linear = |addr: D3D12_TEXTURE_ADDRESS_MODE, reg: u32| D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: addr,
        AddressV: addr,
        AddressW: addr,
        ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: reg,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        ..Default::default()
    };
    let samplers = [
        linear(D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0), // s0 scene / depth
        linear(D3D12_TEXTURE_ADDRESS_MODE_WRAP, 1),  // s1 hit albedo / normal map
        linear(D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 2), // s2 prefilter + probe cubes
    ];

    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    serialize_desc_and_create(device, &desc, "transparent rt root sig")
}

// The per-frame RtParams upload ring, built alongside the RT root signature.
type RtParamsRing = (Vec<PooledBuffer>, Vec<*mut u8>);

fn build_rt_params_ring(alloc: &DeviceAllocator) -> Result<RtParamsRing, String> {
    let params_size = align256(RT_PARAMS_UBO_SIZE);
    let mut resources: Vec<PooledBuffer> = Vec::with_capacity(FRAMES);
    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        let buf = create_buffer(
            alloc,
            params_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { buf.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("map transparent rt params ubo: {e}"))?;
        ptrs.push(ptr as *mut u8);
        resources.push(buf);
    }
    Ok((resources, ptrs))
}

// Write the scene-copy SRV (single-sample HDR Texture2D). Mirrors the raymarch
// scene-snapshot SRV; kept local so `resize_to` can re-point the descriptor.
fn write_scene_copy_srv(
    device: &ID3D12Device,
    scene_copy: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: HDR_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(scene_copy, Some(&desc), srv_cpu) };
}

// The device handles the transparent build submits against.
#[derive(Clone, Copy)]
pub(in crate::directx) struct TransparentDeviceCtx<'a> {
    pub alloc: &'a DeviceAllocator,
}

// Render-target build config: MSAA sample count, render dimensions, and the
// shader hot-reload toggle.
#[derive(Clone, Copy)]
pub(in crate::directx) struct TransparentBuildConfig {
    pub msaa_samples: u32,
    pub width: u32,
    pub height: u32,
    pub hot_reload: bool,
}

// GPU descriptor handles for the scene snapshot (CPU + GPU SRV) and the
// main-depth SRV the transparent fragments sample.
#[derive(Clone, Copy)]
pub(in crate::directx) struct TransparentSceneTargets {
    pub scene_copy_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub scene_copy_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

// The world's transparent content: the pane and water assets with their
// per-record planar slot assignment (aligned with each slice; `None` records
// keep the probe/sky reflection, from `assign_planar_slots`, which numbers water
// first), plus the draw slots carrying a see-through material. A mesh has no
// planar slot -- it is curved, and the RT trace it takes is sharper than a
// mirror render anyway.
#[derive(Clone, Copy)]
pub(in crate::directx) struct TransparentContent<'a> {
    pub glass_panels: &'a [GlassPanel],
    pub glass_planar_slots: &'a [Option<usize>],
    pub water_surfaces: &'a [WaterSurface],
    pub water_planar_slots: &'a [Option<usize>],
    // Indices into the context's draw objects of every see-through material.
    // Empty when no material opted in; those meshes then render opaque.
    pub seethrough_mesh_indices: &'a [usize],
}

impl TransparentResources {
    // Build the shared root signatures, each live producer's pipelines + records,
    // the per-frame view ring, and the scene snapshot. Called from
    // `DxContext::new` when the world declares any `GlassPanel` or `WaterSurface`.
    pub(in crate::directx) fn new(
        device_ctx: TransparentDeviceCtx,
        config: TransparentBuildConfig,
        scene: TransparentSceneTargets,
        content: TransparentContent,
        info_queue: Option<&ID3D12InfoQueue>,
    ) -> Result<Self, String> {
        let TransparentDeviceCtx { alloc } = device_ctx;
        let device = alloc.device();
        let TransparentBuildConfig {
            msaa_samples,
            width,
            height,
            hot_reload,
        } = config;
        let TransparentSceneTargets {
            scene_copy_srv_cpu,
            scene_copy_srv_gpu,
            depth_srv_gpu,
        } = scene;
        let root_sig = dump_on_err(info_queue, create_transparent_root_signature(device))?;

        // RT reflection state: built when the GPU supports DXR, regardless of
        // whether RT is on at launch (a live toggle selects it with no rebuild). A
        // DXC failure leaves it None and the base probe/planar path runs.
        let rt = if crate::directx::raytrace::raytracing_supported(device) {
            match dump_on_err(info_queue, create_transparent_rt_root_signature(device))
                .and_then(|sig| build_rt_params_ring(alloc).map(|ring| (sig, ring)))
            {
                Ok((sig, ring)) => Some((sig, ring)),
                Err(e) => {
                    tracing::warn!(
                        "transparent RT reflection setup failed ({e}); \
                         using the probe/planar path"
                    );
                    None
                }
            }
        } else {
            None
        };
        let (rt_root_sig, rt_params_ubo_resources, rt_params_ubo_ptrs) = match rt {
            Some((sig, (res, ptrs))) => (Some(sig), res, ptrs),
            None => (None, Vec::new(), Vec::new()),
        };

        let glass = if content.glass_panels.is_empty() {
            None
        } else {
            Some(super::glass::build_glass_producer(
                super::glass::GlassBuild {
                    alloc,
                    root_sig: &root_sig,
                    rt_root_sig: rt_root_sig.as_ref(),
                    msaa_samples,
                    hot_reload,
                    info_queue,
                },
                content.glass_panels,
                content.glass_planar_slots,
            )?)
        };
        let water = if content.water_surfaces.is_empty() {
            None
        } else {
            Some(super::water::build_water_producer(
                super::water::WaterBuild {
                    alloc,
                    root_sig: &root_sig,
                    rt_root_sig: rt_root_sig.as_ref(),
                    msaa_samples,
                    hot_reload,
                    info_queue,
                },
                content.water_surfaces,
                content.water_planar_slots,
            )?)
        };

        // The see-through mesh producer, built only when a material opted in AND
        // the pass has an RT root signature: the trace is the whole feature, so
        // without DXR there is nothing to build and those meshes stay opaque. A
        // shader-compile failure is non-fatal for the same reason -- it is logged
        // and the meshes keep the Layer 1 opaque-reflective look.
        let glass_mesh = match (content.seethrough_mesh_indices.is_empty(), &rt_root_sig) {
            (false, Some(sig)) => {
                match super::glass::build_glass_mesh_producer(
                    super::glass::GlassMeshBuild {
                        alloc,
                        rt_root_sig: sig,
                        msaa_samples,
                        hot_reload,
                        info_queue,
                    },
                    content.seethrough_mesh_indices,
                ) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        tracing::warn!(
                            "see-through glass mesh pipeline build failed ({e});                              those meshes render opaque"
                        );
                        None
                    }
                }
            }
            _ => None,
        };

        // Per-frame view UBO ring.
        let view_size = align256(std::mem::size_of::<TransparentView>() as u64);
        let mut view_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(FRAMES);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                alloc,
                view_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map transparent view ubo: {e}"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        // Pre-transparent scene snapshot. Created in PIXEL_SHADER_RESOURCE;
        // `encode_transparent` flips it to COPY_DEST for the snapshot copy and
        // back each frame.
        let scene_copy = create_hdr_resolve_target(device, width.max(1), height.max(1))?;
        write_scene_copy_srv(device, &scene_copy, scene_copy_srv_cpu);

        Ok(Self {
            root_sig,
            glass,
            water,
            glass_mesh,
            view_ubo_resources,
            view_ubo_ptrs,
            scene_copy,
            scene_copy_srv_cpu,
            scene_copy_srv_gpu,
            depth_srv_gpu,
            rt_root_sig,
            rt_params_ubo_resources,
            rt_params_ubo_ptrs,
        })
    }

    // The shared root signature every base-path PSO is built against, for the
    // shader hot-reload rebuild.
    pub(in crate::directx) fn root_sig(&self) -> &ID3D12RootSignature {
        &self.root_sig
    }

    // Swap in freshly compiled base-path PSOs after a shader hot-reload. Either
    // producer may be absent; a `None` argument leaves that producer's pipeline
    // alone.
    pub(in crate::directx) fn swap_pipelines(
        &mut self,
        glass_pso: Option<ID3D12PipelineState>,
        water_pso: Option<ID3D12PipelineState>,
    ) {
        if let (Some(pso), Some(p)) = (glass_pso, self.glass.as_mut()) {
            p.pso = pso;
        }
        if let (Some(pso), Some(p)) = (water_pso, self.water.as_mut()) {
            p.pso = pso;
        }
    }

    pub(in crate::directx) fn has_glass(&self) -> bool {
        self.glass.is_some()
    }

    pub(in crate::directx) fn has_water(&self) -> bool {
        self.water.is_some()
    }

    // True when the per-pixel RT pipelines are built (DXR-capable GPU + the DXC
    // compile + RT root sig succeeded) for every live producer. Single-sources
    // the "the transparent pass can trace" half of
    // `DxContext::rt_transparent_active`: gating on the whole set is what keeps
    // the RT choice a per-frame one rather than a per-producer one, so the planar
    // mirror render the graph skips is never one a producer still needs.
    pub(in crate::directx) fn rt_pipelines_ready(&self) -> bool {
        self.rt_root_sig.is_some()
            && self.glass.as_ref().is_none_or(|p| p.flat_rt_pso.is_some())
            && self.water.as_ref().is_none_or(|p| p.flat_rt_pso.is_some())
    }

    // True when every live producer also built its textured RT pipeline, so the
    // whole pass can take the bindless hit-shading variant. Like the flat gate
    // this is all-or-nothing across producers: the two variants are built against
    // the same root signature but the textured one additionally reads the pool
    // table, and a per-producer split would leave one drawing under a pipeline
    // the encoder's bind sequence does not match.
    pub(in crate::directx) fn rt_textured_ready(&self) -> bool {
        self.glass
            .as_ref()
            .is_none_or(|p| p.textured_rt_pso.is_some())
            && self
                .water
                .as_ref()
                .is_none_or(|p| p.textured_rt_pso.is_some())
            && self
                .glass_mesh
                .as_ref()
                .is_none_or(|p| p.textured_rt_pso.is_some())
    }

    // Recreate the scene snapshot at new render-target dimensions and rewrite
    // its SRV in place. The descriptor slot does not move, so the encoder's GPU
    // handle stays valid. Mirrors `RaymarchResources::resize_to`.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        self.scene_copy = create_hdr_resolve_target(device, width.max(1), height.max(1))?;
        write_scene_copy_srv(device, &self.scene_copy, self.scene_copy_srv_cpu);
        Ok(())
    }

    // True when any record of the pane or water producer is currently visible.
    // The mesh producer is not covered here: its visibility is per-frame state
    // that lives in `draw.objects`, so `DxContext::transparent_enabled` asks it
    // separately. Together they drive `FrameGraphInputs::transparent_enabled`.
    // True when a visible water surface holds a planar slot, so the mirror
    // re-render has a consumer this frame even while the trace is live. Water
    // takes the mirror over its own trace (see `water.slang`), so this is what
    // `planar_pass_needed` reads; glass is deliberately not counted.
    pub(in crate::directx) fn water_planar_slot_live(&self) -> bool {
        self.water.as_ref().is_some_and(|p| {
            p.records
                .iter()
                .any(|r| r.visible && r.planar_slot.is_some())
        })
    }

    pub(in crate::directx) fn any_visible(&self) -> bool {
        let live = |p: &Option<TransparentProducer>| {
            p.as_ref()
                .is_some_and(|p| p.records.iter().any(|r| r.visible))
        };
        live(&self.glass) || live(&self.water)
    }

    // The see-through meshes this producer was built over, or an empty slice
    // when the world declared none.
    pub(in crate::directx) fn seethrough_mesh_indices(&self) -> &[usize] {
        self.glass_mesh
            .as_ref()
            .map(|p| p.object_indices.as_slice())
            .unwrap_or_default()
    }

    // True when the see-through mesh pipelines are built, so the Layer 2 reroute
    // can engage as soon as RT is live. Independent of `rt_accel`, because the
    // init-time BLAS build has to exclude the meshes it will reroute before the
    // acceleration structure it gates on exists.
    pub(in crate::directx) fn mesh_pipelines_ready(&self) -> bool {
        self.glass_mesh.is_some()
    }

    // Every visible record of the static producers plus this frame's mesh draws,
    // farthest first.
    fn draw_order(&self, meshes: &[[f32; 3]], cam: [f32; 3]) -> Vec<(Producer, usize)> {
        let centres = |p: &Option<TransparentProducer>| -> Vec<([f32; 3], bool)> {
            p.as_ref()
                .map(|p| p.records.iter().map(|r| (r.centre, r.visible)).collect())
                .unwrap_or_default()
        };
        ordered_visible(&centres(&self.glass), &centres(&self.water), meshes, cam)
    }

    fn record(&self, kind: Producer, index: usize) -> &TransparentRecord {
        let producer = match kind {
            Producer::Glass => self.glass.as_ref(),
            Producer::Water => self.water.as_ref(),
            Producer::GlassMesh => {
                unreachable!("mesh draws are per-frame and never resolve to a static record")
            }
        };
        &producer
            .expect("the draw order only names live producers")
            .records[index]
    }
}

// Refraction offset + Fresnel falloff for a see-through glass MESH. A `Material`
// carries no glass-specific tunables (unlike a `GlassPanel`), so these match the
// `GlassPanel` defaults: a gentle screen-space refraction and a fresnel power of
// 1 (subtle reflection head-on, full mirror at grazing). Same constants the
// Metal backend uses, so a mesh reads the same on every backend.
const GLASS_MESH_REFRACTION: f32 = 0.02;
const GLASS_MESH_FRESNEL_POWER: f32 = 1.0;

impl DxContext {
    // Build this frame's see-through mesh draw list and write each mesh's params
    // into its slot of the producer's ring. Only called while RT is live.
    //
    // Each mesh resolves its own LOD slice by camera distance exactly as the
    // opaque passes do, so a mesh rerouted here rasterises the same triangles it
    // would have rasterised opaque.
    fn collect_mesh_draws(
        &self,
        transparent: &TransparentResources,
        frame_idx: usize,
        cam: [f32; 3],
    ) -> Vec<GlassMeshDraw> {
        let Some(producer) = transparent.glass_mesh.as_ref() else {
            return Vec::new();
        };
        let block = align256(std::mem::size_of::<GlassMeshParams>() as u64);
        let ring_base = com::gpu_va(&producer.params_ring[frame_idx]);
        let ring_ptr = producer.params_ptrs[frame_idx];
        let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;

        let mut draws = Vec::with_capacity(producer.object_indices.len());
        for (slot, &idx) in producer.object_indices.iter().enumerate() {
            let Some(obj) = self.draw.objects.get(idx) else {
                continue;
            };
            // The flag is re-read rather than trusted from the init list, so this
            // producer and the opaque-pass skip decide from the same live
            // predicate and cannot disagree about which meshes are rerouted.
            if !obj.visible || !obj.resident || obj.material.see_through == 0 {
                continue;
            }
            let centre = [
                0.5 * (obj.bb_min[0] + obj.bb_max[0]),
                0.5 * (obj.bb_min[1] + obj.bb_max[1]),
                0.5 * (obj.bb_min[2] + obj.bb_max[2]),
            ];
            let d = crate::gfx::lod::camera_distance(obj, cam);
            let (index_offset, index_count) = obj.active_lod(d);
            let t = obj.material.tint;
            let params = GlassMeshParams {
                model: obj.model,
                tint: [t[0], t[1], t[2], 0.0],
                opacity: obj.material.opacity,
                refraction_strength: GLASS_MESH_REFRACTION,
                fresnel_power: GLASS_MESH_FRESNEL_POWER,
                prefilter_mip_count,
            };
            let offset = slot as u64 * block;
            // SAFETY: the destination is `slot`'s block of the persistent mapping of an
            // UPLOAD-heap buffer init sized for one block per mesh per frame, and the source is a
            // separate live local, so the ranges cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &params as *const GlassMeshParams as *const u8,
                    ring_ptr.add(offset as usize),
                    std::mem::size_of::<GlassMeshParams>(),
                );
            }
            draws.push(GlassMeshDraw {
                index_offset: index_offset as u32,
                index_count: index_count as u32,
                base_vertex: obj.base_vertex,
                params_gva: ring_base + offset,
                centre,
            });
        }
        draws
    }

    // Encode the transparent pass: snapshot the scene for refraction, then draw
    // every visible glass pane and water surface back-to-front into the post-SSR
    // scene target with SRC_ALPHA blending. No-op when the world has no
    // transparent content or nothing is visible.
    pub(in crate::directx) fn encode_transparent(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view: &TransparentView,
        // Projection inputs for the per-pixel RT reflection trace's RtParams (the
        // same values the RT-reflection resolve uses); only consumed on the RT path.
        fov_y_radians: f32,
        aspect: f32,
    ) -> Result<(), String> {
        let transparent = match &self.transparent {
            Some(t) => t,
            None => return Ok(()),
        };
        let cam = [view.camera_pos[0], view.camera_pos[1], view.camera_pos[2]];

        // Per-pixel RT reflection is selected over the probe/planar path when RT
        // is live (the scene TLAS is built) AND every live producer's RT pipelines
        // compiled at init -- single-sourced via `rt_transparent_active`, the same
        // predicate graph_exec uses to skip the planar mirror re-render (so the
        // two always agree). The textured variant additionally needs the bindless
        // albedo/normal pool the GPU-cull path populates; without it, the flat-tint
        // trace runs. Mirrors Metal's transparent-draw pipeline selection.
        let rt_live = self.rt_transparent_active();
        let textured =
            rt_live && self.cull.main_bindless_pso.is_some() && transparent.rt_textured_ready();

        // This frame's see-through mesh draws. Empty unless RT is live: the
        // per-pixel trace is the feature, and with RT off those meshes rasterise
        // opaque in the main pass instead.
        let mesh_draws = if rt_live {
            self.collect_mesh_draws(transparent, frame_idx, cam)
        } else {
            Vec::new()
        };
        let mesh_centres: Vec<[f32; 3]> = mesh_draws.iter().map(|d| d.centre).collect();
        let order = transparent.draw_order(&mesh_centres, cam);
        if order.is_empty() {
            return Ok(());
        }

        // Upload this frame's view UBO.
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                view as *const TransparentView as *const u8,
                transparent.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<TransparentView>(),
            );
        }
        let view_gva = com::gpu_va(&transparent.view_ubo_resources[frame_idx]);

        // On the RT path, upload this frame's RtParams (sun + ray tunables) into
        // the shared RtParams ring. Mirrors `encode_rt_reflections`'s build.
        let rt_params_gva = if rt_live {
            let rt = self.rt_reflections.as_ref().expect("rt_reflections_active");
            let v = self.view.matrix;
            let inv_view_rot = [
                [v[0][0], v[1][0], v[2][0], 0.0],
                [v[0][1], v[1][1], v[2][1], 0.0],
                [v[0][2], v[1][2], v[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let params = rt.settings.params(RtParamsInputs {
                fov_y_radians,
                aspect,
                inv_view_rot,
                cam_pos: cam,
                sun_dir: self.fog.sun_dir,
                sun_color: self.fog.sun_color,
                prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
            });
            // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer
            // that init sized for this payload, and the source is a separate live value, so the
            // ranges cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &params as *const RtParams as *const u8,
                    transparent.rt_params_ubo_ptrs[frame_idx],
                    std::mem::size_of::<RtParams>(),
                );
            }
            Some(com::gpu_va(&transparent.rt_params_ubo_resources[frame_idx]))
        } else {
            None
        };

        // The scene this pass blends into is the graph's `scene_pre_taa` or, with
        // no reflection resolve, `hdr_resolve` -- the same branch the graph
        // builder takes for the Transparent node's read-modify-write, so the
        // executor has already put whichever one applies in RENDER_TARGET.
        let scene_res = self.post_scene_target();
        let scene_rtv = self.post_scene_rtv();

        // Snapshot the scene into `scene_copy` so refraction reads a stable copy
        // of what it is also blending into: a fragment cannot sample the
        // attachment it blends into. The copy and its restore both live inside
        // this node and leave no net state at the boundary.
        let scene_to_copy = transition_barrier(
            self.post_scene_target(),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let copy_to_dst = transition_barrier(
            &transparent.scene_copy,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[scene_to_copy, copy_to_dst]) };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.CopyResource(&transparent.scene_copy, scene_res) };
        let scene_to_rt = transition_barrier(
            self.post_scene_target(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let copy_to_psr = transition_barrier(
            &transparent.scene_copy,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[scene_to_rt, copy_to_psr]) };

        // Main depth is already in a shader-resource state for the fragment's
        // manual occlusion Load: the graph declares this pass's depth read and the
        // executor emits the transition ahead of this command list.

        let w = self.extent.render_width;
        let h = self.extent.render_height;
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&scene_rtv), false, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: w as i32,
                bottom: h as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
        }

        // Reflection sources frame-constant across every record (like Metal's
        // encode_transparent): the sky prefilter cube (t2), the probe cube array,
        // and the per-frame ProbeSet CBV (b4). count == 0 keeps the sky fallback.
        let prefilter_srv = self.prefilter_cube_srv_gpu();
        let probe_cube_srv = self.probe_cube_table_gpu();
        let probe_set_gva = com::gpu_va(&self.probe.set_cbvs[frame_idx]);

        // The one per-record binding whose root parameter moves between the two
        // signatures (the RT one appends it past the trace's inputs).
        let planar_root = if rt_live {
            PLANAR_ROOT_RT
        } else {
            PLANAR_ROOT_BASE
        };
        // The two producers share their root signature, so it binds once and only
        // the pipeline changes across the draw loop.
        let root_sig = if rt_live {
            transparent
                .rt_root_sig
                .as_ref()
                .expect("rt_live built the rt root sig")
        } else {
            &transparent.root_sig
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetGraphicsRootSignature(root_sig);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.SetGraphicsRootDescriptorTable(2, transparent.scene_copy_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(3, transparent.depth_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(4, prefilter_srv);
            cmd.SetGraphicsRootDescriptorTable(5, probe_cube_srv);
            cmd.SetGraphicsRootConstantBufferView(6, probe_set_gva);
        }
        if rt_live {
            // Sharp per-pixel trace. Bind the RT inputs once before the draw loop
            // (mirrors `encode_rt_reflections`); no per-record planar, since the RT
            // root sig has no planar slot -- planar is the RT-off sharp path.
            let rt_params_gva = rt_params_gva.expect("rt_live uploaded RtParams");
            let accel = self.rt_accel.as_ref().expect("rt_reflections_active");
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.SetGraphicsRootConstantBufferView(7, rt_params_gva);
                cmd.SetGraphicsRootShaderResourceView(8, accel.tlas_gva());
                cmd.SetGraphicsRootShaderResourceView(9, com::gpu_va(&self.geometry.vertex_buffer));
                cmd.SetGraphicsRootShaderResourceView(10, com::gpu_va(&self.geometry.index_buffer));
                cmd.SetGraphicsRootShaderResourceView(11, accel.geom_table_gva());
                cmd.SetGraphicsRootShaderResourceView(12, accel.deformed_verts_gva());
                cmd.SetGraphicsRootShaderResourceView(13, accel.skinned_index_gva());
                if textured {
                    cmd.SetGraphicsRootDescriptorTable(
                        14,
                        self.cull.bindless_pool_gpu[self.current_frame],
                    );
                }
            }
        }

        // A valid stand-in for every draw that has no mirror plane of its own --
        // the see-through meshes below and any slotless record. The shaders gate
        // on `planar > 0.5`, so none of them sample it; binding it keeps the
        // table set for every PSO that runs under this signature.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.SetGraphicsRootDescriptorTable(planar_root, transparent.scene_copy_srv_gpu) };

        let mut bound: Option<Producer> = None;
        for &(kind, i) in &order {
            if bound != Some(kind) {
                // SAFETY: the command list is in the recording state, and every resource,
                // descriptor and slice these commands name is live for the call.
                unsafe {
                    match kind {
                        Producer::GlassMesh => cmd.SetPipelineState(
                            transparent
                                .glass_mesh
                                .as_ref()
                                .expect("the draw order only names live producers")
                                .pipeline(textured),
                        ),
                        Producer::Glass | Producer::Water => {
                            let producer = match kind {
                                Producer::Glass => transparent.glass.as_ref(),
                                _ => transparent.water.as_ref(),
                            }
                            .expect("the draw order only names live producers");
                            cmd.SetPipelineState(producer.pipeline(rt_live, textured));
                        }
                    }
                }
                bound = Some(kind);
            }
            if kind == Producer::GlassMesh {
                // A mesh draws its `DrawObject`'s slice of the shared scene
                // buffers, so the vertex / index views are the scene's rather than
                // a record's own, and the slice rides the draw arguments.
                let d = &mesh_draws[i];
                // SAFETY: the command list is in the recording state, and every resource,
                // descriptor and slice these commands name is live for the call.
                unsafe {
                    cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
                    cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
                    cmd.SetGraphicsRootConstantBufferView(1, d.params_gva);
                    cmd.DrawIndexedInstanced(d.index_count, 1, d.index_offset, d.base_vertex, 0);
                }
                self.inc_draw_calls(1);
                continue;
            }
            let r = transparent.record(kind, i);
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.IASetVertexBuffers(0, Some(&[r.vertex_buffer_view]));
                cmd.IASetIndexBuffer(Some(&r.index_buffer_view));
                cmd.SetGraphicsRootConstantBufferView(1, r.params_cbuffer_gva);
                // Planar resolve table (t3), per record: this record's mirror
                // render when it has a planar slot, else the scene snapshot as a
                // valid stand-in (the shaders gate on `planar > 0.5`, so a
                // slotless record never samples it).
                let planar_srv = r
                    .planar_slot
                    .and_then(|s| {
                        self.planar_reflection
                            .as_ref()
                            .map(|set| set.resolve_srv_gpu(s))
                    })
                    .unwrap_or(transparent.scene_copy_srv_gpu);
                cmd.SetGraphicsRootDescriptorTable(planar_root, planar_srv);
                cmd.DrawIndexedInstanced(r.index_count, 1, 0, 0, 0);
            }
            self.inc_draw_calls(1);
        }

        // The scene target and main depth are both graph resources; the next
        // consumer's barrier takes the scene back out of RENDER_TARGET.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_distance_is_euclidean_and_monotone() {
        let cam = [0.0, 0.0, 0.0];
        let near = sort_distance([0.0, 0.0, 1.0], cam);
        let far = sort_distance([0.0, 0.0, 5.0], cam);
        assert!((near - 1.0).abs() < 1e-5);
        assert!((far - 5.0).abs() < 1e-5);
        assert!(far > near);
    }

    #[test]
    fn ordered_visible_excludes_hidden_and_sorts_back_to_front() {
        // Pane 1 is hidden; 0 (dist 5) and 2 (dist 3) are visible. Farthest
        // first => [0, 2]; the hidden pane never appears.
        let glass = [
            ([0.0, 0.0, 5.0], true),
            ([0.0, 0.0, 9.0], false),
            ([0.0, 0.0, 3.0], true),
        ];
        let order = ordered_visible(&glass, &[], &[], [0.0, 0.0, 0.0]);
        assert_eq!(order, vec![(Producer::Glass, 0), (Producer::Glass, 2)]);
    }

    #[test]
    fn ordered_visible_interleaves_the_two_producers() {
        // A pane standing in a pool has to composite in distance order, not in
        // producer order: the far pane draws first, then the water, then the near
        // pane.
        let glass = [([0.0, 0.0, 9.0], true), ([0.0, 0.0, 1.0], true)];
        let water = [([0.0, 0.0, 5.0], true), ([0.0, 0.0, 7.0], false)];
        let order = ordered_visible(&glass, &water, &[], [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![
                (Producer::Glass, 0),
                (Producer::Water, 0),
                (Producer::Glass, 1),
            ]
        );
    }

    #[test]
    fn ordered_visible_interleaves_mesh_draws_with_the_static_producers() {
        // A see-through mesh sorts against panes and water by the same camera
        // distance, so it is not simply appended after them. Every mesh entry the
        // encoder passes is already visible, which is why the slice carries
        // centres alone.
        let glass = [([0.0, 0.0, 9.0], true)];
        let water = [([0.0, 0.0, 3.0], true)];
        let meshes = [[0.0, 0.0, 6.0], [0.0, 0.0, 1.0]];
        let order = ordered_visible(&glass, &water, &meshes, [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![
                (Producer::Glass, 0),
                (Producer::GlassMesh, 0),
                (Producer::Water, 0),
                (Producer::GlassMesh, 1),
            ]
        );
    }

    #[test]
    fn ordered_visible_orders_meshes_alone_back_to_front() {
        // A world whose only transparent content is see-through meshes: the pass
        // still runs, and they still sort farthest first.
        let meshes = [[0.0, 0.0, 2.0], [0.0, 0.0, 8.0]];
        let order = ordered_visible(&[], &[], &meshes, [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![(Producer::GlassMesh, 1), (Producer::GlassMesh, 0)]
        );
    }

    #[test]
    fn ordered_visible_is_empty_with_no_visible_records() {
        let glass = [([0.0, 0.0, 5.0], false)];
        let water = [([0.0, 0.0, 3.0], false)];
        assert!(ordered_visible(&glass, &water, &[], [0.0, 0.0, 0.0]).is_empty());
    }
}
