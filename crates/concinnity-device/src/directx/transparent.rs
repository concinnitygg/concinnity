//! The engine's `PassId::Transparent` slot on the D3D12 backend: one pass, drawn
//! after the SSR resolve and before TAA, with three producers -- glass panes
//! (`glass.rs`), water surfaces (`water.rs`), and see-through glass meshes
//! (`glass.rs` as well, since a mesh is the same material family as a pane). The
//! pass snapshots the pre-transparent scene, orders every record of every
//! producer back-to-front by camera distance, and draws them into the post-SSR
//! scene target with straight-alpha blending.
//!
//! The pane and water producers contribute records built once at init; a water
//! record's params are rewritten each frame as its waves move. The mesh
//! producer cannot: a see-through mesh draws from the SHARED scene vertex / index
//! buffers at the offsets its `DrawObject` carries, and both those offsets (LOD
//! picks per frame) and its params (model matrix, material tint) change at
//! runtime. So it owns only its pipelines plus a per-frame params ring, and the
//! encoder rebuilds its draw list each frame.
//!
//! One pass rather than one per producer, mirroring the Metal backend: the scene
//! snapshot the refraction taps is a full render-resolution HDR image and a copy
//! of it every frame, so a second one would be pure waste; and a single ordering
//! over both producers is what puts a pane standing in a pool on the correct side
//! of the water.
//!
//! The producers also share their root signatures, because `glass.hlsl`,
//! `water.hlsl` and `glass_mesh.hlsl` declare the same registers on purpose.
//! There are two: the base signature (probe / planar reflection) and the RT
//! one, which adds the ray-tracing SRVs at t4..t12. Both bind the probe set at
//! t20..t22, clear of those SRVs. Which one runs is a per-frame choice, not a
//! per-producer one -- see `DxContext::rt_transparent_active`.
//!
//! The shaders are the shared `shaders/glass.hlsl`, `shaders/glass_mesh.hlsl`
//! and `shaders/water.hlsl`, compiled through `builtin_shaders`; the ray-traced
//! fragments need shader model 6.5 for their inline ray query, the base pairs
//! 6.0. The mesh producer is ray-traced only -- the per-pixel trace is what
//! makes it see-through rather than the opaque reflective glass the main pass
//! draws -- so it runs only under the RT root signature and is inert while RT
//! is off.

use concinnity_core::components::GlassPanel;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types::RtParams;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::planet::Rebase;
use concinnity_core::render::backend::error::{RenderError, RenderResult};
use concinnity_core::render::depth::DEPTH_CLEAR;
use concinnity_core::render::temporal::reactive_mask::ReactiveWrite;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use crate::directx::context::{DxContext, align256, dump_on_err};
use crate::directx::descriptor_slot::{DescriptorTables, SrvSlot};
use crate::directx::error::map_hresult;
use crate::directx::init::heap_layout::GLASS_REFLECTION_SRV_SLOTS;
use crate::directx::pipeline::main_input_layout;
use crate::directx::pso::{Blend, Depth, GraphicsPso, Raster};
use crate::directx::root_sig::{Range, RootSig, SamplerState, Visibility};
use crate::directx::texture::{
    HDR_FORMAT, create_hdr_resolve_target, create_main_depth_texture, create_rt_target,
    transition_barrier, upload_buffer, write_format_rtv,
};

// RtParams push size (144 B; see gfx::render_types::RtParams), shared with the
// RT-reflection resolve.
const RT_PARAMS_UBO_SIZE: u64 = 144;

// `TransparentView` (the per-frame view cbuffer) is a GPU-free layout struct
// that lives in `core::render`; re-export it so the encode path and the
// graph's view builder can keep naming it through this module.
use concinnity_core::render::transparent::producer::{self, TransparentProducers, WaterParamsSink};
use concinnity_core::render::transparent::{SeeThroughMesh, TransparentFramePlan};
pub(in crate::directx) use concinnity_core::render::uniforms::TransparentView;
use concinnity_core::render::uniforms::{GlassMeshParams, WaterParams};
use std::sync::Mutex;

pub(in crate::directx) use concinnity_core::render::transparent::Producer;

// One drawable record's GPU state: a static world-space (glass) or
// origin-centered grid (water) VB + IB plus a per-record uniform CBV. Built
// once at init; only a water record's params change afterwards, rewritten into
// the frame slot's block each frame.
pub(in crate::directx) struct RecordBuffers {
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
    // One 256-aligned params block per frame slot, persistently mapped, so a
    // record whose params move (a water surface's waves) writes this frame's
    // block while the GPU may still read the previous one.
    #[expect(
        dead_code,
        reason = "held to keep the GPU memory alive; the encoder binds through params_cbuffer_gva"
    )]
    params_cbuffer: PooledBuffer,
    params_cbuffer_gva: u64,
    params_ptr: *mut u8,
    params_block: u64,
    frames: usize,
}

pub(in crate::directx) type TransparentRecord = producer::TransparentRecord<RecordBuffers>;
pub(in crate::directx) type TransparentProducer =
    producer::TransparentProducer<ID3D12PipelineState, RecordBuffers>;
pub(in crate::directx) type ShapePsos = producer::ShapePipelines<ID3D12PipelineState>;

// Upload one record's static geometry and its params into every frame slot's
// block of its CBV buffer, which stays persistently mapped for the producers
// whose params move.
pub(in crate::directx) fn upload_record(
    alloc: &DeviceAllocator,
    upload: RecordUpload<'_>,
) -> RenderResult<TransparentRecord> {
    Ok(TransparentRecord {
        gpu: RecordBuffers::upload(alloc, &upload)?,
        visible: upload.visible,
        center: upload.center,
        planar_slot: upload.planar_slot,
    })
}

// The geometry, uniform payload and per-record state one producer hands over for
// a `TransparentRecord`. Keeps the buffer uploads in one place instead of once
// per producer.
#[derive(Clone, Copy)]
pub(in crate::directx) struct RecordUpload<'a> {
    pub vertices: &'a [Vertex],
    pub indices: &'a [u16],
    pub params: &'a [u8],
    pub visible: bool,
    pub center: [f32; 3],
    pub planar_slot: Option<usize>,
}

impl RecordBuffers {
    fn upload(alloc: &DeviceAllocator, upload: &RecordUpload<'_>) -> RenderResult<Self> {
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
            SizeInBytes: u32::saturating_from(vbytes.len()),
            StrideInBytes: u32::saturating_from(size_of::<Vertex>()),
        };
        let index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&index_buffer),
            SizeInBytes: u32::saturating_from(ibytes.len()),
            Format: DXGI_FORMAT_R16_UINT,
        };

        let frames = alloc.frames_in_flight().max(1);
        let params_block = align256(upload.params.len() as u64);
        let params_cbuffer = alloc.alloc_buffer(
            params_block * frames as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut p = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { params_cbuffer.Map(0, None, Some(&raw mut p)) }
            .map_err(|e| map_hresult(e.code(), "map transparent params cb"))?;
        let params_ptr = p.cast::<u8>();
        for frame in 0..frames {
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold one block per
            // frame slot, so `frame`'s block starts inside it.
            let block = unsafe { params_ptr.add(frame * usize::saturating_from(params_block)) };
            // SAFETY: `frame`'s block holds the payload, and the source is a separate allocation,
            // so the ranges cannot overlap. Persistently mapped, never unmapped.
            unsafe {
                std::ptr::copy_nonoverlapping(upload.params.as_ptr(), block, upload.params.len());
            }
        }
        let params_cbuffer_gva = com::gpu_va(&params_cbuffer);

        Ok(Self {
            vertex_buffer,
            vertex_buffer_view,
            index_buffer,
            index_buffer_view,
            index_count: u32::saturating_from(upload.indices.len()),
            params_cbuffer,
            params_cbuffer_gva,
            params_ptr,
            params_block,
            frames,
        })
    }

    // The GPU address of the params block for frame slot `frame_idx`.
    fn params_gva(&self, frame_idx: usize) -> u64 {
        self.params_cbuffer_gva + (frame_idx % self.frames) as u64 * self.params_block
    }
}

// A water record's params, which only the water producer writes: its records
// were uploaded from a `WaterParams`, so each block holds exactly one.
impl WaterParamsSink for RecordBuffers {
    fn write_water_params(&self, frame_idx: usize, params: &WaterParams) {
        let bytes = bytemuck::bytes_of(params);
        let offset = (frame_idx % self.frames) * usize::saturating_from(self.params_block);
        // SAFETY: `offset` is `frame_idx`'s block of the persistent mapping, which holds one
        // block per frame slot.
        let block = unsafe { self.params_ptr.add(offset) };
        // SAFETY: this water record's upload sized the block (256-aligned) for one `WaterParams`,
        // and the source is a separate live value, so the ranges cannot overlap.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), block, bytes.len()) };
    }
}

// A traced glass producer's PSOs: the shading pair and the reflection
// pre-pass pair.
pub(in crate::directx) struct TracedGlassPsos {
    pub shade_flat: ID3D12PipelineState,
    pub shade_textured: ID3D12PipelineState,
    pub reflection_flat: ID3D12PipelineState,
    pub reflection_textured: ID3D12PipelineState,
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
    // The glass reflection pre-pass pair.
    reflection_flat_pso: ID3D12PipelineState,
    reflection_textured_pso: ID3D12PipelineState,
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
// ring, and its world-space center for the back-to-front sort.
struct GlassMeshDraw {
    index_offset: u32,
    index_count: u32,
    base_vertex: i32,
    params_gva: u64,
    center: [f32; 3],
}

impl GlassMeshProducer {
    // Allocate the per-frame params ring (one 256-aligned block per mesh per
    // frame slot, persistently mapped) and take ownership of the pipelines.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        psos: TracedGlassPsos,
        object_indices: Vec<usize>,
    ) -> RenderResult<Self> {
        let block = align256(size_of::<GlassMeshParams>() as u64);
        let ring_size = block * object_indices.len().max(1) as u64;
        let frames = alloc.frames_in_flight();
        let mut params_ring: Vec<PooledBuffer> = Vec::with_capacity(frames);
        let mut params_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                ring_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map glass mesh params ring"))?;
            params_ptrs.push(ptr.cast::<u8>());
            params_ring.push(buf);
        }
        Ok(Self {
            flat_rt_pso: psos.shade_flat,
            textured_rt_pso: Some(psos.shade_textured),
            reflection_flat_pso: psos.reflection_flat,
            reflection_textured_pso: psos.reflection_textured,
            object_indices,
            params_ring,
            params_ptrs,
        })
    }

    // Pick the frame's pipeline. Same all-or-nothing gate as the other
    // producers: `rt_textured_ready` requires every live producer's textured
    // pipeline, so this is never asked for one it lacks.
    fn pipeline(&self, textured: bool) -> &ID3D12PipelineState {
        if textured {
            self.textured_rt_pso
                .as_ref()
                .expect("rt_textured_ready gated the frame on every producer's textured PSO")
        } else {
            &self.flat_rt_pso
        }
    }

    // The reflection pre-pass PSO, under the same gate as `pipeline`.
    fn reflection_pipeline(&self, textured: bool) -> &ID3D12PipelineState {
        if textured {
            &self.reflection_textured_pso
        } else {
            &self.reflection_flat_pso
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
    producers: TransparentProducers<ID3D12PipelineState, RecordBuffers>,
    glass_mesh: Option<GlassMeshProducer>,
    // This frame's plan, reused frame to frame. Locked once by the encoder.
    plan: Mutex<TransparentFramePlan<GlassMeshDraw>>,

    // Per-frame view UBO (single 256-byte block), persistently mapped.
    view_ubo_resources: Vec<PooledBuffer>,
    view_ubo_ptrs: Vec<*mut u8>,

    // Pre-transparent scene snapshot. `encode_transparent` copies the scene
    // target into this each frame before the draws so refraction reads a stable
    // copy instead of the attachment being written.
    scene_copy: ID3D12Resource,
    scene_copy_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    scene_copy_srv_gpu: SrvSlot,
    // Main-depth SRV (shared with the decal pass); bound at t1 for the manual
    // occlusion test.
    depth_srv_gpu: SrvSlot,

    // Ray-traced reflection state, present only when the GPU supports DXR AND
    // every RT pipeline compiled. One root signature shared by both producers'
    // flat-tint + textured-bindless PSOs, chosen per frame in
    // `encode_transparent` while RT is live; otherwise the base `pso`s run the
    // probe/planar path. The RtParams ring feeds the trace and is empty (never
    // read) when the RT pipelines are absent.
    rt_root_sig: Option<ID3D12RootSignature>,
    rt_params_ubo_resources: Vec<PooledBuffer>,
    rt_params_ubo_ptrs: Vec<*mut u8>,

    // The glass reflection pre-pass: its fixed slots, the reduced layers while
    // the trace divisor is above 1, and that divisor, kept so a resize rebuilds
    // the layers at the same one.
    reflection_slots: GlassReflectionSlots,
    reflection: Option<GlassReflectionLayers>,
    reflection_divisor: u32,
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

// Root-signature layout (binds 1:1 with the `CN_BACKEND_DIRECTX` declarations
// in glass.hlsl and water.hlsl, which are deliberately identical):
//   [0] root CBV b0   TransparentView (per-frame)
//   [1] root CBV b1   GlassParams / WaterParams (per-record)
//   [2] table  t0     scene-copy SRV  (Texture2D<float4>)
//   [3] table  t1     scene depth SRV (Texture2D[MS]<float>)
//   [4] table  t2     sky prefilter cube SRV
//   [5] table  t20    reflection-probe cube array
//   [6] root CBV b4   ProbeSet
//   [7] table  t3     planar reflection resolve SRV (per record)
//   [8] root SRV t21  reflection-probe records
//   [9] root CBV b6   ClusterParams (the main camera's cluster grid)
//   [10] root SRV t22 per-cluster light lists and probe masks
//   [11] table  t13..14 the cascaded shadow map and the sky irradiance cube
//   [12] root CBV b3  ShadowUniforms
//   [13] table  t15..16 the simulated oceans' displacement and slope arrays
//   static sampler s0 : linear clamp ; s2 : cube mip-linear clamp ;
//   s3 : shadow compare ; s4 : anisotropic wrap (every stage)
//
// b1 is visible to every stage: the water vertex stage reads its wave table out
// of the params block, where the glass vertex stage reads only the view.
// Root parameter index of the per-record planar resolve table: the one before
// the probe records in the base signature and before the reflection layers in
// the RT one, so each builder asserts its own length against the constants
// rather than the encoder repeating a literal that can drift.
const PLANAR_ROOT_BASE: u32 = 7;
const PLANAR_ROOT_RT: u32 = 15;
// Root parameter index of the RT signature's glass reflection layer table.
const GLASS_REFLECTION_ROOT_RT: u32 = 16;
// Root parameter index of the probe records root SRV (t21), followed in both
// signatures by the cluster grid's params CBV (b6) and lists SRV (t22).
const PROBE_RECORDS_ROOT_BASE: u32 = 8;
const PROBE_RECORDS_ROOT_RT: u32 = 17;
// Root parameter index of the water body's light table (t13..t14: the shadow
// map and the irradiance cube, the heap's two leading global SRVs), followed in
// both signatures by the shadow uniforms CBV (b3).
const BODY_LIGHT_ROOT_BASE: u32 = 11;
const BODY_LIGHT_ROOT_RT: u32 = 20;
// Root parameter index of the oceans' array table (t15..t16), last in both
// signatures and visible to every stage: the water's vertex stage displaces
// by it.
const OCEAN_ROOT_BASE: u32 = 13;
const OCEAN_ROOT_RT: u32 = 22;

fn create_transparent_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    use Visibility::{All, Pixel};
    let sig = RootSig::new()
        .cbv(0, All) // [0] b0 TransparentView
        .cbv(1, All) // [1] b1 per-record params
        .srv_table(0, 1, Pixel) // [2] t0 scene copy
        .srv_table(1, 1, Pixel) // [3] t1 depth
        .srv_table(2, 1, Pixel) // [4] t2 sky prefilter cube, the fallback where no probe covers
        .srv_table(20, 1, Pixel) // [5] t20 probe cubes, box-projected when ProbeSet.count > 0
        .cbv(4, Pixel) // [6] b4 ProbeSet
        // [7] t3 planar resolve: a valid SRV is always bound (the scene snapshot
        // stands in for records with no planar slot); sampled only when `planar > 0.5`.
        .srv_table(3, 1, Pixel)
        .srv(21, Pixel) // [8] t21 probe records
        .cbv(6, Pixel) // [9] b6 ClusterParams
        .srv(22, Pixel) // [10] t22 cluster lists
        .srv_table(13, 2, Pixel) // [11] t13..t14 shadow map + irradiance cube
        .cbv(3, Pixel) // [12] b3 ShadowUniforms
        .srv_table(15, 2, All) // [13] t15..t16 ocean arrays
        // s0: the scene snapshot / depth. s2: the prefilter + probe cube array.
        // s3: the shadow compare. s4: the ocean arrays.
        .static_sampler(SamplerState::LinearClamp, 0, Pixel)
        .static_sampler(SamplerState::LinearClamp, 2, Pixel)
        .static_sampler(SamplerState::ShadowCompare, 3, Pixel)
        .static_sampler(SamplerState::AnisotropicWrap, 4, All)
        .input_layout();
    debug_assert_eq!(sig.len() - 7, PLANAR_ROOT_BASE);
    debug_assert_eq!(sig.len() - 6, PROBE_RECORDS_ROOT_BASE);
    debug_assert_eq!(sig.len() - 3, BODY_LIGHT_ROOT_BASE);
    debug_assert_eq!(sig.len() - 1, OCEAN_ROOT_BASE);
    sig.build(device, "transparent root sig")
}

// PSO for a transparent producer. Writes the single-sample post-SSR scene target
// with src-alpha / inv-src-alpha blending, and the reactive mask beside it. No
// depth attachment (the fragment shader does the manual occlusion test) and no
// face culling (both shaders are two-sided). Standard 5-attribute vertex layout
// shared with the main pass.
pub(in crate::directx) fn create_transparent_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    transparent_pso(device, root_sig, (vs, ps), TransparentOutput::Scene)
}

// The glass reflection pre-pass PSO: the transparent PSO's stages, overwriting
// a reflection layer and depth-tested (write test, writing) against its `D32_FLOAT`
// depth, so each layer keeps the nearest surface it accepts.
pub(in crate::directx) fn create_glass_reflection_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    transparent_pso(
        device,
        root_sig,
        (vs, ps),
        TransparentOutput::ReflectionLayer,
    )
}

// What a transparent PSO draws into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TransparentOutput {
    // Straight-alpha blended over the scene, with no depth target.
    Scene,
    // Overwriting a glass reflection layer, depth-tested against its target.
    ReflectionLayer,
}

fn transparent_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    (vs, ps): (&[u8], &[u8]),
    output: TransparentOutput,
) -> RenderResult<ID3D12PipelineState> {
    let (blend, depth_format, depth) = match output {
        TransparentOutput::Scene => (Blend::AlphaOver, DXGI_FORMAT_UNKNOWN, Depth::Off),
        TransparentOutput::ReflectionLayer => {
            (Blend::Opaque, DXGI_FORMAT_D32_FLOAT, Depth::write())
        }
    };
    let layout = main_input_layout();
    let pso = GraphicsPso::new(root_sig, vs, ps)
        .input_layout(&layout)
        .target(HDR_FORMAT, blend);
    let pso = match output {
        TransparentOutput::Scene => crate::directx::reactive_mask::mask_target(pso),
        TransparentOutput::ReflectionLayer => pso,
    };
    pso.depth(depth_format, depth)
        .raster(Raster {
            depth_clip: false,
            ..Raster::default()
        })
        .build(device, "transparent")
}

// Root signature for the RT PSOs (binds 1:1 with the `CN_BACKEND_DIRECTX`
// declarations the RT producers take from `transparent_rt.hlsl`, plus their own):
//   [0]  root CBV b0   TransparentView (per-frame, vertex + pixel)
//   [1]  root CBV b1   GlassParams / WaterParams (per-record, every stage)
//   [2]  table  t0     scene-copy SRV
//   [3]  table  t1     scene depth SRV
//   [4]  table  t2     sky prefilter cube SRV
//   [5]  table  t20    reflection-probe cube array
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
//   [16] table  t11..12 the glass reflection layers (one window of
//                      `GlassReflectionSlots` per pass)
//   [17] root SRV t21  reflection-probe records
//   [18] root CBV b6   ClusterParams (the main camera's cluster grid)
//   [19] root SRV t22  per-cluster light lists and probe masks
//   [20] table  t13..14 the cascaded shadow map and the sky irradiance cube
//   [21] root CBV b3   ShadowUniforms
//   [22] table  t15..16 the simulated oceans' displacement and slope arrays
//   static samplers s0 linear-clamp, s1 linear-repeat, s2 cube linear-clamp,
//                   s3 shadow compare, s4 anisotropic wrap (every stage)
//
// [15] is what the base signature carries at [7]: the water fragment samples its
// mirror plane in place of tracing wherever it has one, so the resolve has to
// reach the RT PSOs too. Glass never reads t3 on this path; the encoder still
// binds the table for every draw so no PSO runs under an unset root parameter.
fn create_transparent_rt_root_signature(
    device: &ID3D12Device,
) -> RenderResult<ID3D12RootSignature> {
    use Visibility::{All, Pixel};
    let sig = RootSig::new()
        .cbv(0, All) // [0] b0 TransparentView (vertex reads vp)
        .cbv(1, All) // [1] b1 per-record params (water's vertex reads it)
        .srv_table(0, 1, Pixel) // [2] t0 scene copy
        .srv_table(1, 1, Pixel) // [3] t1 depth
        .srv_table(2, 1, Pixel) // [4] t2 prefilter cube
        .srv_table(20, 1, Pixel) // [5] t20 probe cubes
        .cbv(4, Pixel) // [6] b4 ProbeSet
        .cbv(5, Pixel) // [7] b5 RtParams
        .srv(4, Pixel) // [8] t4 TLAS
        .srv(5, Pixel) // [9] t5 verts
        .srv(6, Pixel) // [10] t6 indices
        .srv(10, Pixel) // [11] t10 geom table
        .srv(8, Pixel) // [12] t8 skinned verts
        .srv(9, Pixel) // [13] t9 skinned indices
        .table(&[Range::bindless_srv(1)], Pixel) // [14] t0,space1 bindless pool
        .srv_table(3, 1, Pixel) // [15] t3 planar resolve
        .srv_table(11, 2, Pixel) // [16] t11..t12 glass reflection layers
        .srv(21, Pixel) // [17] t21 probe records
        .cbv(6, Pixel) // [18] b6 ClusterParams
        .srv(22, Pixel) // [19] t22 cluster lists
        .srv_table(13, 2, Pixel) // [20] t13..t14 shadow map + irradiance cube
        .cbv(3, Pixel) // [21] b3 ShadowUniforms
        .srv_table(15, 2, All) // [22] t15..t16 ocean arrays
        .static_sampler(SamplerState::LinearClamp, 0, Pixel) // s0 scene / depth
        .static_sampler(SamplerState::LinearWrap, 1, Pixel) // s1 hit albedo / normal map
        .static_sampler(SamplerState::LinearClamp, 2, Pixel) // s2 prefilter + probe cubes
        .static_sampler(SamplerState::ShadowCompare, 3, Pixel) // s3 shadow compare
        .static_sampler(SamplerState::AnisotropicWrap, 4, All) // s4 ocean arrays
        .input_layout();
    debug_assert_eq!(sig.len() - 8, PLANAR_ROOT_RT);
    debug_assert_eq!(sig.len() - 7, GLASS_REFLECTION_ROOT_RT);
    debug_assert_eq!(sig.len() - 6, PROBE_RECORDS_ROOT_RT);
    debug_assert_eq!(sig.len() - 3, BODY_LIGHT_ROOT_RT);
    debug_assert_eq!(sig.len() - 1, OCEAN_ROOT_RT);
    sig.build(device, "transparent rt root sig")
}

// The per-frame RtParams upload ring, built alongside the RT root signature.
type RtParamsRing = (Vec<PooledBuffer>, Vec<*mut u8>);

fn build_rt_params_ring(alloc: &DeviceAllocator) -> RenderResult<RtParamsRing> {
    let params_size = align256(RT_PARAMS_UBO_SIZE);
    let frames = alloc.frames_in_flight();
    let mut resources: Vec<PooledBuffer> = Vec::with_capacity(frames);
    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
    for _ in 0..frames {
        let buf = alloc.alloc_buffer(
            params_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "map transparent rt params ubo"))?;
        ptrs.push(ptr.cast::<u8>());
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
    unsafe { device.CreateShaderResourceView(scene_copy, Some(&raw const desc), srv_cpu) };
}

// The glass reflection pre-pass's fixed descriptor slots: an RTV per layer, the
// shared DSV, and the six SRVs that form the three two-descriptor windows the
// RT signature's t11..t12 table points at (see `GLASS_REFLECTION_SRV_SLOTS`):
// `windows[0]` for the first layer's pass, `[1]` for the second's, `[2]` for the
// scene pass.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GlassReflectionSlots {
    pub rtv: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_cpu: [D3D12_CPU_DESCRIPTOR_HANDLE; GLASS_REFLECTION_SRV_SLOTS],
    pub windows: [SrvSlot; 3],
}

// The reduced glass reflection layers (rgb the traced reflection, a the
// surface's distance from the camera) and the depth they share.
struct GlassReflectionLayers {
    layers: [ID3D12Resource; 2],
    _depth: ID3D12Resource,
    extent: (u32, u32),
}

impl GlassReflectionLayers {
    fn new(
        device: &ID3D12Device,
        slots: &GlassReflectionSlots,
        extent: (u32, u32),
    ) -> RenderResult<Self> {
        let (w, h) = extent;
        let layers = [
            create_rt_target(device, w, h, HDR_FORMAT)?,
            create_rt_target(device, w, h, HDR_FORMAT)?,
        ];
        for (layer, rtv) in layers.iter().zip(slots.rtv) {
            write_format_rtv(device, layer, rtv, HDR_FORMAT);
        }
        let depth = create_main_depth_texture(device, w, h, slots.dsv, 1, false)?;
        Ok(Self {
            layers,
            _depth: depth,
            extent,
        })
    }
}

// Write the three layer windows: the first layer's pass peels behind nothing
// (both null), the second's behind the first layer, and the scene pass reads
// both. With no layers every descriptor is null; the shaders then see
// `trace_divisor` 1 and never read them. A null SRV reads as zero.
fn write_reflection_windows(
    device: &ID3D12Device,
    slots: &GlassReflectionSlots,
    layers: Option<&GlassReflectionLayers>,
) {
    let [front, back] = layers.map_or([None, None], |l| [Some(&l.layers[0]), Some(&l.layers[1])]);
    let contents = [None, None, front, None, front, back];
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: HDR_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    for (resource, cpu) in contents.into_iter().zip(slots.srv_cpu) {
        // SAFETY: the view descriptor and the resource it names (or none, for a null view) are
        // live for the call, and the destination handle addresses a slot this context reserved
        // for the view in a heap it owns.
        unsafe { device.CreateShaderResourceView(resource, Some(&raw const desc), cpu) };
    }
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
    // Per-axis divisor of the glass reflection pre-pass; 1 traces in place.
    pub reflection_divisor: u32,
}

// GPU descriptor handles for the scene snapshot (CPU + GPU SRV) and the
// main-depth SRV the transparent fragments sample.
#[derive(Clone, Copy)]
pub(in crate::directx) struct TransparentSceneTargets {
    pub scene_copy_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub scene_copy_srv_gpu: SrvSlot,
    pub depth_srv_gpu: SrvSlot,
    pub reflection_slots: GlassReflectionSlots,
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
    pub water_surfaces: &'a [concinnity_core::render::environment::water::WaterBody],
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
        device_ctx: TransparentDeviceCtx<'_>,
        config: TransparentBuildConfig,
        scene: TransparentSceneTargets,
        content: TransparentContent<'_>,
        info_queue: Option<&ID3D12InfoQueue>,
    ) -> RenderResult<Self> {
        let TransparentDeviceCtx { alloc } = device_ctx;
        let device = alloc.device();
        let TransparentBuildConfig {
            msaa_samples,
            width,
            height,
            hot_reload,
            reflection_divisor,
        } = config;
        let TransparentSceneTargets {
            scene_copy_srv_cpu,
            scene_copy_srv_gpu,
            depth_srv_gpu,
            reflection_slots,
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
        let view_size = align256(size_of::<TransparentView>() as u64);
        let frames = alloc.frames_in_flight();
        let mut view_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(frames);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                view_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map transparent view ubo"))?;
            view_ubo_ptrs.push(ptr.cast::<u8>());
            view_ubo_resources.push(buf);
        }

        // Pre-transparent scene snapshot. Created in PIXEL_SHADER_RESOURCE;
        // `encode_transparent` flips it to COPY_DEST for the snapshot copy and
        // back each frame.
        let scene_copy = create_hdr_resolve_target(device, width.max(1), height.max(1))?;
        write_scene_copy_srv(device, &scene_copy, scene_copy_srv_cpu);

        let mut me = Self {
            root_sig,
            producers: TransparentProducers { glass, water },
            plan: Mutex::default(),
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
            reflection_slots,
            reflection: None,
            reflection_divisor,
        };
        me.rebuild_reflection_layers(device, width, height)?;
        Ok(me)
    }

    // Take a new glass reflection divisor, rebuilding the layers at it. The
    // caller has idled the device.
    pub(in crate::directx) fn set_reflection_divisor(
        &mut self,
        device: &ID3D12Device,
        (width, height): (u32, u32),
        divisor: u32,
    ) -> RenderResult<()> {
        if divisor == self.reflection_divisor {
            return Ok(());
        }
        self.reflection_divisor = divisor;
        self.rebuild_reflection_layers(device, width, height)
    }

    // (Re)build the reduced reflection layers for a `width` x `height` render at
    // the current divisor and rewrite their windows; `None` when glass traces in
    // place (a divisor of 1, or no glass producer that can trace).
    fn rebuild_reflection_layers(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        let traced = self
            .producers
            .glass
            .as_ref()
            .is_some_and(|p| p.reflection_flat.is_some())
            || self.glass_mesh.is_some();
        let d = self.reflection_divisor;
        self.reflection = None;
        if d > 1 && traced {
            let extent = ((width / d).max(1), (height / d).max(1));
            self.reflection = Some(GlassReflectionLayers::new(
                device,
                &self.reflection_slots,
                extent,
            )?);
        }
        write_reflection_windows(device, &self.reflection_slots, self.reflection.as_ref());
        Ok(())
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
        if let (Some(pso), Some(p)) = (glass_pso, self.producers.glass.as_mut()) {
            p.shape.base = pso;
        }
        if let (Some(pso), Some(p)) = (water_pso, self.producers.water.as_mut()) {
            p.shape.base = pso;
        }
    }

    pub(in crate::directx) fn has_glass(&self) -> bool {
        self.producers.glass.is_some()
    }

    pub(in crate::directx) fn has_water(&self) -> bool {
        self.producers.water.is_some()
    }

    // True when the per-pixel RT pipelines are built (DXR-capable GPU + the DXC
    // compile + RT root sig succeeded) for every live producer. Single-sources
    // the "the transparent pass can trace" half of
    // `DxContext::rt_transparent_active`: gating on the whole set is what keeps
    // the RT choice a per-frame one rather than a per-producer one, so the planar
    // mirror render the graph skips is never one a producer still needs.
    pub(in crate::directx) fn rt_pipelines_ready(&self) -> bool {
        self.rt_root_sig.is_some() && self.producers.traces(false)
    }

    // True when every live producer also built its textured RT pipeline, so the
    // whole pass can take the bindless hit-shading variant. Like the flat gate
    // this is all-or-nothing across producers: the two variants are built against
    // the same root signature but the textured one additionally reads the pool
    // table, and a per-producer split would leave one drawing under a pipeline
    // the encoder's bind sequence does not match.
    pub(in crate::directx) fn rt_textured_ready(&self) -> bool {
        self.producers.traces(true)
            && self
                .glass_mesh
                .as_ref()
                .is_none_or(|p| p.textured_rt_pso.is_some())
    }

    // Recreate the scene snapshot and the glass reflection layers at new
    // render-target dimensions and rewrite their descriptors in place. The slots
    // do not move, so the encoder's GPU handles stay valid. Mirrors
    // `RaymarchResources::resize_to`.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        self.scene_copy = create_hdr_resolve_target(device, width.max(1), height.max(1))?;
        write_scene_copy_srv(device, &self.scene_copy, self.scene_copy_srv_cpu);
        self.rebuild_reflection_layers(device, width, height)
    }

    // True when any record of the pane or water producer is currently visible.
    // The mesh producer is not covered here: its visibility is per-frame state
    // that lives in `draw.objects`, so `DxContext::transparent_enabled` asks it
    // separately. Together they drive `FrameGraphInputs::transparent_enabled`.
    // True when a visible water surface holds a planar slot, so the mirror
    // re-render has a consumer this frame even while the trace is live. Water
    // takes the mirror over its own trace (see `water.hlsl`), so this is what
    // `planar_pass_needed` reads; glass is deliberately not counted.
    pub(in crate::directx) fn water_planar_slot_live(&self) -> bool {
        self.producers.water_planar_slot_live()
    }

    pub(in crate::directx) fn any_visible(&self) -> bool {
        self.producers.any_visible()
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
    // can engage as soon as RT is live. Independent of `rt.accel`, because the
    // init-time BLAS build has to exclude the meshes it will reroute before the
    // acceleration structure it gates on exists.
    pub(in crate::directx) fn mesh_pipelines_ready(&self) -> bool {
        self.glass_mesh.is_some()
    }

    // The record-holding producer `kind` names; the mesh producer holds none.
    fn producer(&self, kind: Producer) -> &TransparentProducer {
        self.producers
            .producer(kind)
            .expect("the draw order only names live static producers")
    }

    fn record(&self, kind: Producer, index: usize) -> &TransparentRecord {
        &self.producer(kind).records[index]
    }
}

impl DxContext {
    // Build this frame's see-through mesh draw list and write each mesh's params
    // into its slot of the producer's ring. Only called while RT is live.
    //
    // Each mesh resolves its own LOD slice by camera distance exactly as the
    // opaque passes do, so a mesh rerouted here rasterizes the same triangles it
    // would have rasterized opaque.
    fn collect_mesh_draws(
        &self,
        transparent: &TransparentResources,
        frame_idx: usize,
        cam: [f32; 3],
        draws: &mut Vec<GlassMeshDraw>,
    ) {
        let Some(producer) = transparent.glass_mesh.as_ref() else {
            return;
        };
        let block = align256(size_of::<GlassMeshParams>() as u64);
        let ring_base = com::gpu_va(&producer.params_ring[frame_idx]);
        let ring_ptr = producer.params_ptrs[frame_idx];
        let prefilter_mip_count = self.scene.env_map.prefilter_mip_count as f32;

        for (slot, &idx) in producer.object_indices.iter().enumerate() {
            let Some(obj) = self.state.draw.objects.get(idx) else {
                continue;
            };
            // The flag is re-read rather than trusted from the init list, so this
            // producer and the opaque-pass skip decide from the same live
            // predicate and cannot disagree about which meshes are rerouted.
            if !obj.visible || !obj.resident || obj.material.see_through == 0 {
                continue;
            }
            let mesh = SeeThroughMesh::new(obj, cam, prefilter_mip_count);
            let offset = slot as u64 * block;
            // SAFETY: init sized the persistent mapping for one block per mesh per frame, so
            // `slot`'s block starts inside it.
            let dst = unsafe { ring_ptr.add(usize::saturating_from(offset)) };
            // SAFETY: `dst` is `slot`'s block of an UPLOAD-heap buffer, and the source is a
            // separate live local, so the ranges cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const mesh.params).cast::<u8>(),
                    dst,
                    size_of::<GlassMeshParams>(),
                );
            }
            draws.push(GlassMeshDraw {
                index_offset: u32::saturating_from(mesh.index_offset),
                index_count: u32::saturating_from(mesh.index_count),
                base_vertex: mesh.base_vertex,
                params_gva: ring_base + offset,
                center: mesh.center,
            });
        }
    }

    // Encode the transparent pass: snapshot the scene for refraction, then draw
    // every visible glass pane and water surface back-to-front into the post-SSR
    // scene target with SRC_ALPHA blending. No-op when the world has no
    // transparent content or nothing is visible.
    // The water body's light table: the heap's leading global SRVs, the cascaded
    // shadow map (slot 0) and the irradiance cube (slot 1), which the main pass
    // and the raymarched surfaces read from the same slots.
    fn body_light_srvs_gpu(&self) -> SrvSlot {
        SrvSlot::at(
            &self.descriptors.srv_heap,
            self.descriptors.srv_descriptor_size,
            0,
        )
    }

    // Carry the water surfaces into the frame the world moved to.
    pub(in crate::directx) fn rebase_transparent_water(&mut self, rebase: &Rebase) {
        if let Some(water) = self
            .transparent
            .as_mut()
            .and_then(|t| t.producers.water.as_mut())
        {
            water.rebase_water(rebase);
        }
    }

    // Write each visible water surface's params, advanced to `time` seconds
    // into the world, into frame slot `frame_idx`'s copy, ahead of this frame's
    // `encode_transparent`.
    pub(in crate::directx) fn advance_transparent_water(
        &self,
        frame_idx: usize,
        time: f64,
        camera: [f32; 3],
    ) {
        if let Some(water) = self
            .transparent
            .as_ref()
            .and_then(|t| t.producers.water.as_ref())
        {
            water.advance_water(frame_idx, time, camera);
        }
    }

    // True when a visible surface samples a simulated ocean, so `WaterSim` has
    // a reader this frame.
    pub(in crate::directx) fn water_sim_live(&self) -> bool {
        self.water_sim
            .as_ref()
            .is_some_and(super::water_sim::DxOceanSim::simulates)
            && self
                .transparent
                .as_ref()
                .and_then(|t| t.producers.water.as_ref())
                .is_some_and(TransparentProducer::samples_ocean)
    }

    pub(in crate::directx) fn encode_transparent(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view: &TransparentView,
        // The frame's RT reflection block, which the per-pixel trace shares with
        // the RT-reflection resolve; only consumed on the RT path.
        rt_params: Option<&RtParams>,
        reactive: ReactiveWrite,
    ) -> RenderResult<()> {
        let Some(transparent) = &self.transparent else {
            return Ok(());
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
        // per-pixel trace is the feature, and with RT off those meshes rasterize
        // opaque in the main pass instead.
        let mut plan = transparent
            .plan
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        plan.begin(rt_live, textured);
        if rt_live {
            self.collect_mesh_draws(transparent, frame_idx, cam, plan.meshes_mut());
        }
        plan.order_draws(&transparent.producers, |d| d.center, cam);
        if plan.order().is_empty() {
            return Ok(());
        }
        let order = plan.order();
        let mesh_draws = plan.meshes();
        // The glass reflection pre-pass runs when RT is live, the reduced layers
        // exist, and a traced glass surface draws this frame.
        let reflection_layers = transparent
            .reflection
            .as_ref()
            .filter(|_| plan.reflection_prepass(true));

        // Upload this frame's view UBO.
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref::<TransparentView>(view).cast::<u8>(),
                transparent.view_ubo_ptrs[frame_idx],
                size_of::<TransparentView>(),
            );
        }
        let view_gva = com::gpu_va(&transparent.view_ubo_resources[frame_idx]);

        // On the RT path, upload this frame's RtParams (sun + ray tunables) into
        // the shared RtParams ring: the plan's block, which
        // `encode_rt_reflections` binds too.
        let rt_params_gva = if rt_live {
            let params = *rt_params.ok_or_else(|| {
                RenderError::Other("transparent rt_live without an RT reflection block".to_string())
            })?;
            // Traced glass reads the reduced layers back only when the pre-pass
            // below fills them this frame; otherwise it traces in place.
            let params = RtParams {
                trace_divisor: if reflection_layers.is_some() {
                    params.trace_divisor
                } else {
                    1.0
                },
                ..params
            };
            // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer
            // that init sized for this payload, and the source is a separate live value, so the
            // ranges cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const params).cast::<u8>(),
                    transparent.rt_params_ubo_ptrs[frame_idx],
                    size_of::<RtParams>(),
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
        // SAFETY: the list is recording, and the scene target and `scene_copy` are both live for
        // the frame.
        unsafe { cmd.ResourceBarrier(&[scene_to_copy, copy_to_dst]) };
        // SAFETY: the barrier above put the scene in COPY_SOURCE and `scene_copy` in COPY_DEST.
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

        let w = self.targets.extent.render_width;
        let h = self.targets.extent.render_height;
        // SAFETY: the list is recording; setting the topology names no resource.
        unsafe { cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) };
        // SAFETY: the list is recording, and the SRV heap is owned by `self.descriptors`.
        unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };

        // Reflection sources frame-constant across every record (like Metal's
        // encode_transparent): the sky prefilter cube (t2), the probe cube array
        // (t20), the per-frame ProbeSet CBV (b4) and records (t21), and the main
        // camera's cluster grid binning them (b6, t22). count == 0 keeps the sky
        // fallback.
        let prefilter_srv = self.prefilter_cube_srv_gpu();
        let probes = self.probe_bindings(frame_idx);

        // The one per-record binding whose root parameter moves between the two
        // signatures (the RT one appends it past the trace's inputs).
        let (planar_root, probe_records_root, body_light_root, ocean_root) = if rt_live {
            (
                PLANAR_ROOT_RT,
                PROBE_RECORDS_ROOT_RT,
                BODY_LIGHT_ROOT_RT,
                OCEAN_ROOT_RT,
            )
        } else {
            (
                PLANAR_ROOT_BASE,
                PROBE_RECORDS_ROOT_BASE,
                BODY_LIGHT_ROOT_BASE,
                OCEAN_ROOT_BASE,
            )
        };
        let shadow_gva = com::gpu_va(&self.uniforms.shadow_ubo_resources[frame_idx]);
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
        // SAFETY: the list is recording, and the chosen root signature is owned by `transparent`.
        unsafe { cmd.SetGraphicsRootSignature(root_sig) };
        // SAFETY: both signatures declare the view CBV at 0, and this frame's view block is live.
        unsafe { cmd.SetGraphicsRootConstantBufferView(0, view_gva) };
        // SAFETY: the SRV heap is set above, and the scene snapshot's descriptor is live.
        unsafe { cmd.set_graphics_srv_table(2, transparent.scene_copy_srv_gpu) };
        // SAFETY: the SRV heap is set above, and the depth descriptor `transparent` holds is live.
        unsafe { cmd.set_graphics_srv_table(3, transparent.depth_srv_gpu) };
        // SAFETY: the SRV heap is set above, and the sky prefilter cube's descriptor is live.
        unsafe { cmd.set_graphics_srv_table(4, prefilter_srv) };
        // SAFETY: the SRV heap is set above, and the probe cube array's descriptor is live.
        unsafe { cmd.set_graphics_srv_table(5, probes.cube_table) };
        // SAFETY: the list is recording, and this frame's ProbeSet constant buffer is live.
        unsafe { cmd.SetGraphicsRootConstantBufferView(6, probes.set_cbv) };
        // SAFETY: the list is recording, and this frame's probe records buffer is live.
        unsafe { cmd.SetGraphicsRootShaderResourceView(probe_records_root, probes.records) };
        // SAFETY: the list is recording, and this frame's `ClusterParams` buffer is live.
        unsafe {
            cmd.SetGraphicsRootConstantBufferView(probe_records_root + 1, probes.cluster_cbv);
        }
        // SAFETY: the list is recording, and the per-cluster probe lists are live for the frame.
        unsafe {
            cmd.SetGraphicsRootShaderResourceView(probe_records_root + 2, probes.cluster_list);
        }
        // SAFETY: the SRV heap is set above, and the body light tables this context owns are live.
        unsafe { cmd.set_graphics_srv_table(body_light_root, self.body_light_srvs_gpu()) };
        // SAFETY: the list is recording, and this frame's shadow constant buffer is live.
        unsafe { cmd.SetGraphicsRootConstantBufferView(body_light_root + 1, shadow_gva) };
        // The oceans' arrays, which exist wherever water does.
        if let Some(ocean) = &self.water_sim {
            // SAFETY: the SRV heap is set above, and the ocean's array table is live.
            unsafe { cmd.set_graphics_srv_table(ocean_root, ocean.srv_table()) };
        }
        if rt_live {
            // Sharp per-pixel trace. Bind the RT inputs once before the draw loop
            // (mirrors `encode_rt_reflections`); no per-record planar, since the RT
            // root sig has no planar slot -- planar is the RT-off sharp path.
            let rt_params_gva = rt_params_gva.expect("rt_live uploaded RtParams");
            let accel = self.rt.accel.as_ref().expect("rt_reflections_active");
            // SAFETY: the RT signature declares the params CBV at 7, and this frame's `RtParams`
            // buffer is live.
            unsafe { cmd.SetGraphicsRootConstantBufferView(7, rt_params_gva) };
            // SAFETY: the list is recording, and the TLAS `accel` holds is live for the frame.
            unsafe { cmd.SetGraphicsRootShaderResourceView(8, accel.tlas_gva()) };
            let vertex_gva = com::gpu_va(&self.scene.geometry.vertex_buffer);
            // SAFETY: the list is recording, and `self.scene.geometry` owns the vertex buffer.
            unsafe { cmd.SetGraphicsRootShaderResourceView(9, vertex_gva) };
            let index_gva = com::gpu_va(&self.scene.geometry.index_buffer);
            // SAFETY: the list is recording, and `self.scene.geometry` owns the index buffer.
            unsafe { cmd.SetGraphicsRootShaderResourceView(10, index_gva) };
            // SAFETY: the list is recording, and the geometry table `accel` holds is live.
            unsafe { cmd.SetGraphicsRootShaderResourceView(11, accel.geom_table_gva()) };
            // SAFETY: the deformed vertex GVA is always a live buffer, a dummy without skinning.
            unsafe { cmd.SetGraphicsRootShaderResourceView(12, accel.deformed_verts_gva()) };
            // SAFETY: the skinned index GVA is always a live buffer, a dummy without skinning.
            unsafe { cmd.SetGraphicsRootShaderResourceView(13, accel.skinned_index_gva()) };
            if textured {
                let pool = self.cull.bindless_pool_gpu[self.current_frame];
                // SAFETY: the SRV heap is set above, and this frame's bindless pool table is live.
                unsafe { cmd.set_graphics_srv_table(14, pool) };
            }
        }

        // A valid stand-in for every draw that has no mirror plane of its own --
        // the see-through meshes below and any slotless record. The shaders gate
        // on `planar > 0.5`, so none of them sample it; binding it keeps the
        // table set for every PSO that runs under this signature.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.set_graphics_srv_table(planar_root, transparent.scene_copy_srv_gpu) };

        let frame = TransparentFrame {
            cmd,
            frame_idx,
            planar_root,
            textured,
            mesh_draws,
        };
        if let Some(layers) = reflection_layers {
            self.encode_glass_reflection_layers(transparent, layers, &frame, order);
        }
        if rt_live {
            // SAFETY: the command list is in the recording state, and the table names this
            // context's own reserved descriptors.
            unsafe {
                cmd.set_graphics_srv_table(
                    GLASS_REFLECTION_ROOT_RT,
                    transparent.reflection_slots.windows[2],
                );
            }
        }
        let rtvs = [scene_rtv, self.targets.reactive_mask.rtv(reactive)];
        // SAFETY: the list is recording, and `rtvs` is a live local naming the scene target and
        // the reactive mask, both live for the frame.
        unsafe { cmd.OMSetRenderTargets(2, Some(rtvs.as_ptr()), false, None) };
        // SAFETY: the list is recording, which is all `set_viewport` requires.
        unsafe { set_viewport(cmd, (w, h)) };

        let mut bound: Option<&ID3D12PipelineState> = None;
        for &(kind, i) in order {
            let pso = match kind {
                Producer::GlassMesh => transparent
                    .glass_mesh
                    .as_ref()
                    .expect("the draw order only names live producers")
                    .pipeline(textured),
                _ => transparent
                    .producer(kind)
                    .record_pipeline(i, rt_live, textured),
            };
            if bound != Some(pso) {
                // SAFETY: the command list is in the recording state, and the PSO is live for
                // the call.
                unsafe { cmd.SetPipelineState(pso) };
                bound = Some(pso);
            }
            self.draw_transparent_entry(transparent, &frame, kind, i);
        }

        // The scene target and main depth are both graph resources; the next
        // consumer's barrier takes the scene back out of RENDER_TARGET.
        Ok(())
    }

    // Draw the traced glass into the two reduced reflection layers: each layer
    // clears its color and the shared depth, then draws every glass surface
    // behind the layer its table window names, keeping the nearest. Water traces
    // in place and is skipped. Leaves both layers PIXEL_SHADER_RESOURCE.
    fn encode_glass_reflection_layers(
        &self,
        transparent: &TransparentResources,
        layers: &GlassReflectionLayers,
        frame: &TransparentFrame<'_>,
        order: &[(Producer, usize)],
    ) {
        let cmd = frame.cmd;
        let slots = &transparent.reflection_slots;
        for layer in 0..2 {
            let target = &layers.layers[layer];
            let to_rt = transition_barrier(
                target,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            // SAFETY: the list is recording, and the reflection layer is owned by `layers`.
            unsafe { cmd.ResourceBarrier(&[to_rt]) };
            // SAFETY: the list is recording, and the layer's RTV and the shared DSV are slots this
            // context reserved.
            unsafe {
                cmd.OMSetRenderTargets(
                    1,
                    Some(&raw const slots.rtv[layer]),
                    false,
                    Some(&raw const slots.dsv),
                );
            }
            // SAFETY: the barrier above put the layer in RENDER_TARGET, and its RTV is bound.
            unsafe { cmd.ClearRenderTargetView(slots.rtv[layer], &[0.0; 4], None) };
            // SAFETY: the list is recording, and the reflection DSV is bound above.
            unsafe {
                cmd.ClearDepthStencilView(slots.dsv, D3D12_CLEAR_FLAG_DEPTH, DEPTH_CLEAR, 0, None);
            }
            // SAFETY: the list is recording, which is all `set_viewport` requires.
            unsafe { set_viewport(cmd, layers.extent) };
            // SAFETY: the RT signature declares the reflection table here, and the layer's window
            // descriptors are reserved by this context.
            unsafe { cmd.set_graphics_srv_table(GLASS_REFLECTION_ROOT_RT, slots.windows[layer]) };
            for &(kind, i) in order {
                let pso = match kind {
                    Producer::GlassMesh => transparent
                        .glass_mesh
                        .as_ref()
                        .map(|m| m.reflection_pipeline(frame.textured)),
                    _ => transparent
                        .producer(kind)
                        .reflection_pipeline(frame.textured),
                };
                let Some(pso) = pso else {
                    continue;
                };
                // SAFETY: the command list is in the recording state, and the PSO is live for
                // the call.
                unsafe { cmd.SetPipelineState(pso) };
                self.draw_transparent_entry(transparent, frame, kind, i);
            }
            let to_read = transition_barrier(
                target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            // SAFETY: the command list is in the recording state, and the resource is live.
            unsafe { cmd.ResourceBarrier(&[to_read]) };
        }
    }

    // Bind one draw-order entry's geometry and params and draw it under the set
    // PSO. A mesh draws its `DrawObject`'s slice of the shared scene buffers; a
    // pane or water surface draws its record's own pair plus its planar resolve
    // table (t3): its mirror render when it has a planar slot, else the scene
    // snapshot as a valid stand-in (the shaders gate on `planar > 0.5`).
    fn draw_transparent_entry(
        &self,
        transparent: &TransparentResources,
        frame: &TransparentFrame<'_>,
        kind: Producer,
        i: usize,
    ) {
        let cmd = frame.cmd;
        if kind == Producer::GlassMesh {
            let d = &frame.mesh_draws[i];
            // SAFETY: the view names the scene vertex buffer `self.scene.geometry` owns.
            unsafe { cmd.IASetVertexBuffers(0, Some(&[self.scene.geometry.vertex_buffer_view])) };
            // SAFETY: the view names the scene index buffer `self.scene.geometry` owns.
            unsafe { cmd.IASetIndexBuffer(Some(&raw const self.scene.geometry.index_buffer_view)) };
            // SAFETY: the signature declares the per-draw CBV at 1, and this mesh's block of the
            // params ring is live for the frame.
            unsafe { cmd.SetGraphicsRootConstantBufferView(1, d.params_gva) };
            // SAFETY: the draw's slice lies inside the scene buffers bound above.
            unsafe {
                cmd.DrawIndexedInstanced(d.index_count, 1, d.index_offset, d.base_vertex, 0);
            }
        } else {
            let r = transparent.record(kind, i);
            let planar_srv = r
                .planar_slot
                .and_then(|s| {
                    self.planar_reflection
                        .as_ref()
                        .map(|set| set.resolve_srv_gpu(s))
                })
                .unwrap_or(transparent.scene_copy_srv_gpu);
            // SAFETY: the view names the record's own vertex buffer, which `r.gpu` owns.
            unsafe { cmd.IASetVertexBuffers(0, Some(&[r.gpu.vertex_buffer_view])) };
            // SAFETY: the view names the record's own index buffer, which `r.gpu` owns.
            unsafe { cmd.IASetIndexBuffer(Some(&raw const r.gpu.index_buffer_view)) };
            let params_gva = r.gpu.params_gva(frame.frame_idx);
            // SAFETY: the signature declares the per-draw CBV at 1, and this frame's params block
            // of the record is live.
            unsafe { cmd.SetGraphicsRootConstantBufferView(1, params_gva) };
            // SAFETY: the SRV heap is set, and the planar resolve or scene snapshot is live.
            unsafe { cmd.set_graphics_srv_table(frame.planar_root, planar_srv) };
            // SAFETY: the record's buffers and every table its PSO reads are bound.
            unsafe { cmd.DrawIndexedInstanced(r.gpu.index_count, 1, 0, 0, 0) };
        }
        self.inc_draw_calls(1);
    }
}

// One frame's transparent recording state, shared by the reflection pre-pass
// and the scene pass.
struct TransparentFrame<'a> {
    cmd: &'a ID3D12GraphicsCommandList,
    frame_idx: usize,
    planar_root: u32,
    textured: bool,
    mesh_draws: &'a [GlassMeshDraw],
}

// Set a full-target viewport and scissor over `(width, height)`.
//
// # Safety
// `cmd` must be in the recording state.
unsafe fn set_viewport(cmd: &ID3D12GraphicsCommandList, (w, h): (u32, u32)) {
    let vp = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: w as f32,
        Height: h as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let scissor = RECT {
        left: 0,
        top: 0,
        right: i32::saturating_from(w),
        bottom: i32::saturating_from(h),
    };
    // SAFETY: the caller guarantees `cmd` is recording; the viewport slice is live for the call.
    unsafe { cmd.RSSetViewports(&[vp]) };
    // SAFETY: as above, with the scissor slice a live local.
    unsafe { cmd.RSSetScissorRects(&[scissor]) };
}
