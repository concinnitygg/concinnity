//! The grass field on DirectX 12 (see `concinnity_core::render::grass`): the
//! bend pass that relaxes and stamps the trampling field, the kernel that
//! places this frame's visible blades and, a second time, the nearest shadow
//! cascade's, the indirect draws, one per detail level, that render them at the
//! tail of the G-buffer pre-pass and the main pass, and the cascade's depth-only
//! draw.
//!
//! The view's draws run under their pass's own root signature, which carries
//! the grass block and the blade buffer as two extra root parameters, so the
//! lit blades shade on the lights, shadows and environment the surfaces bound.
//! The bend pass and the cascade draw have small root signatures of their own.
//! The blade, draw-argument and bend buffers rest in UNORDERED_ACCESS; the
//! graph moves them to the states their readers need.

use std::cell::Cell;

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::grass::bend::GRASS_BEND_WORDS;
use concinnity_core::render::grass::lod::{GRASS_LOD_COUNT, GRASS_LOD_VERTEX_STRIDE};
use concinnity_core::render::grass::{
    GrassBender, GrassCamera, GrassField, GrassFrameInputs, GrassHistory, GrassHiz, GrassPass,
    GrassShadowView,
};
use concinnity_core::render::pass_timing;
use concinnity_core::render::render_graph::PassId;
use concinnity_core::render::shadow_bias;
use concinnity_core::render::uniforms::grass::{
    GRASS_ARGS_BYTES, GRASS_ARGS_STRIDE, GpuGrassBlade, GrassBendParams, GrassParams,
    grass_args_offset,
};
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::builtin_shaders::{self, CompileProgram};
use super::com;
use super::context::{DxContext, align256};
use super::descriptor_slot::DescriptorTables;
use super::error::map_hresult;
use super::pso::{Blend, Depth, DepthBias, GraphicsPso, Raster, compute_pso};
use super::root_constants::RootConstants;
use super::root_sig::{RootSig, Visibility};
use super::texture::{HDR_FORMAT, create_uav_buffer, upload_buffer};

/// Where the main pass's root signature carries the grass block (b7) and the
/// visible blades (t23).
pub(in crate::directx) const MAIN_GRASS_PARAMS_PARAM: u32 = 21;
pub(in crate::directx) const MAIN_GRASS_BLADES_PARAM: u32 = 22;
/// The b0 root constant both pass root signatures lead with, which carries each
/// grass draw's first vertex: D3D's vertex ids ignore the draw's own.
const FIRST_VERTEX_PARAM: u32 = 0;
/// The same pair in the G-buffer pre-pass's root signature.
pub(in crate::directx) const PREPASS_GRASS_PARAMS_PARAM: u32 = 9;
pub(in crate::directx) const PREPASS_GRASS_BLADES_PARAM: u32 = 10;

// The kernel's root signature: the block at b0, the blades it appends at u1,
// the draw arguments at u2, the terrain heights and mask texels it reads at t3
// and t4, last frame's depth pyramid at t5, and the bend field at t6, as
// `grass.hlsl` declares them.
fn create_generate_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .uav(1, Visibility::All)
        .uav(2, Visibility::All)
        .srv(3, Visibility::All)
        .srv(4, Visibility::All)
        .srv_table(5, 1, Visibility::All)
        .srv(6, Visibility::All)
        .build(device, "grass generate root sig")
}

// The kernel's root parameters holding the pyramid's descriptor table and the
// bend field.
const KERNEL_HIZ_PARAM: u32 = 5;
const KERNEL_BEND_PARAM: u32 = 6;

// The bend pass's root signature: its block at b0 and the field at u1.
fn create_bend_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .uav(1, Visibility::All)
        .build(device, "grass bend root sig")
}

// The cascade draw's root signature: the shadow block at b0 and the cascade's
// blades at t1.
fn create_shadow_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::Vertex)
        .srv(1, Visibility::Vertex)
        .build(device, "grass shadow root sig")
}

// One non-indexed draw per command, matching the records the kernel fills, one
// per detail level. It sets no root argument, so it needs no root signature.
fn create_draw_signature(device: &ID3D12Device) -> RenderResult<ID3D12CommandSignature> {
    let arg_descs = [D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
        ..Default::default()
    }];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: GRASS_ARGS_STRIDE as u32,
        NumArgumentDescs: arg_descs.len() as u32,
        pArgumentDescs: arg_descs.as_ptr(),
        NodeMask: 0,
    };
    let mut sig: Option<ID3D12CommandSignature> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig) }
        .map_err(|e| map_hresult(e.code(), "create grass command signature"))?;
    sig.ok_or_else(|| {
        RenderError::Other("create grass command signature: returned None".to_string())
    })
}

// The two root signatures the grass draws run under.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GrassRootSigs<'a> {
    pub main: &'a ID3D12RootSignature,
    pub prepass: &'a ID3D12RootSignature,
}

// The kernels and the draw pipelines; rebuilt as a set on a shader reload.
pub(in crate::directx) struct GrassPipelines {
    generate_root_sig: ID3D12RootSignature,
    generate: ID3D12PipelineState,
    bend_root_sig: ID3D12RootSignature,
    bend: ID3D12PipelineState,
    prepass: ID3D12PipelineState,
    main: ID3D12PipelineState,
    shadow_root_sig: ID3D12RootSignature,
    shadow: ID3D12PipelineState,
}

impl GrassPipelines {
    pub(in crate::directx) fn build(
        device: &ID3D12Device,
        roots: GrassRootSigs<'_>,
        msaa_samples: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let generate_root_sig = create_generate_root_signature(device)?;
        let cs = builtin_shaders::GRASS_GENERATE.compile(hot_reload)?;
        let generate = compute_pso(device, &generate_root_sig, &cs, "grass generate")?;
        let bend_root_sig = create_bend_root_signature(device)?;
        let cs = builtin_shaders::GRASS_BEND.compile(hot_reload)?;
        let bend = compute_pso(device, &bend_root_sig, &cs, "grass bend")?;
        let shadow_root_sig = create_shadow_root_signature(device)?;
        let vs = builtin_shaders::GRASS_SHADOW_VERT.compile(hot_reload)?;
        let shadow = GraphicsPso::new(&shadow_root_sig, &vs, &[])
            .depth(DXGI_FORMAT_D32_FLOAT, Depth::write())
            .raster(Raster {
                bias: DepthBias {
                    constant: shadow_bias::RASTER_CONSTANT as i32,
                    clamp: shadow_bias::RASTER_CLAMP,
                    slope: shadow_bias::RASTER_SLOPE,
                },
                ..Raster::default()
            })
            .build(device, "grass shadow")?;

        let vs = builtin_shaders::GRASS_PREPASS_VERT.compile(hot_reload)?;
        let ps = builtin_shaders::GRASS_PREPASS_FRAG.compile(hot_reload)?;
        let prepass =
            super::post::gbuffer::gbuffer_targets(GraphicsPso::new(roots.prepass, &vs, &ps))
                .depth(DXGI_FORMAT_D32_FLOAT, Depth::write())
                .build(device, "grass prepass")?;

        let vs = builtin_shaders::GRASS_VERT.compile(hot_reload)?;
        let ps = builtin_shaders::GRASS_FRAG.compile(hot_reload)?;
        let main = GraphicsPso::new(roots.main, &vs, &ps)
            .target(HDR_FORMAT, Blend::Opaque)
            .depth(DXGI_FORMAT_D32_FLOAT, Depth::write())
            .samples(msaa_samples)
            .build(device, "grass")?;
        Ok(Self {
            generate_root_sig,
            generate,
            bend_root_sig,
            bend,
            prepass,
            main,
            shadow_root_sig,
            shadow,
        })
    }
}

// The grass field and what draws it: built once at init when the world grows
// one and the GPU-driven main pass exists to draw it in.
pub(in crate::directx) struct GrassResources {
    pub(in crate::directx) field: GrassField,
    pub(in crate::directx) pipelines: GrassPipelines,
    draw_signature: ID3D12CommandSignature,
    // The visible blades the kernel appends, `field.capacity` of them, each
    // detail level's region after the last.
    pub(in crate::directx) blades: ID3D12Resource,
    // Two slots of one draw per detail level; see `GRASS_ARGS_SLOTS`.
    pub(in crate::directx) args: ID3D12Resource,
    // The nearest shadow cascade's blades and draw arguments, laid out like
    // the view's.
    pub(in crate::directx) shadow_blades: ID3D12Resource,
    pub(in crate::directx) shadow_args: ID3D12Resource,
    // The bend field's two halves.
    pub(in crate::directx) bend_field: ID3D12Resource,
    // Every terrain's heights and every layer's mask texels, which the kernel
    // reads to root and thin the blades.
    heights: PooledBuffer,
    masks: PooledBuffer,
    // Per frame in flight, persistently mapped: the view's `GrassParams`, the
    // cascade's, and the bend pass's block.
    params: MappedBlocks,
    shadow_params: MappedBlocks,
    bend_params: MappedBlocks,
    // What the field's frames carry from one to the next. Advanced from
    // `&self` on the render thread before the fan-out.
    history: Cell<GrassHistory>,
}

// One persistently mapped upload-heap block per frame in flight.
struct MappedBlocks {
    buffers: Vec<PooledBuffer>,
    ptrs: Vec<*mut u8>,
}

impl MappedBlocks {
    fn new(alloc: &DeviceAllocator, size: usize, label: &str) -> RenderResult<Self> {
        let size = align256(size as u64);
        let frames = alloc.frames_in_flight();
        let mut buffers = Vec::with_capacity(frames);
        let mut ptrs = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| map_hresult(e.code(), label))?;
            ptrs.push(ptr as *mut u8);
            buffers.push(buf);
        }
        Ok(Self { buffers, ptrs })
    }

    // Write `value` into frame `frame_idx`'s block and return its address.
    fn write<T: bytemuck::NoUninit>(&self, frame_idx: usize, value: &T) -> u64 {
        let bytes = bytemuck::bytes_of(value);
        // SAFETY: the destination is the persistent mapping of this frame's UPLOAD-heap constant
        // buffer, sized for `T`, which the GPU no longer reads once this frame's slot has been
        // waited on; the source is a separate live value, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptrs[frame_idx], bytes.len());
        }
        com::gpu_va(&self.buffers[frame_idx])
    }
}

impl GrassResources {
    pub(in crate::directx) fn build(
        alloc: &DeviceAllocator,
        field: GrassField,
        roots: GrassRootSigs<'_>,
        msaa_samples: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let pipelines = GrassPipelines::build(device, roots, msaa_samples, hot_reload)?;
        let draw_signature = create_draw_signature(device)?;
        let blade_bytes =
            u64::from(field.capacity.total()) * std::mem::size_of::<GpuGrassBlade>() as u64;
        let blades = create_uav_buffer(device, blade_bytes, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &blades, blade_bytes)?;
        let args = create_uav_buffer(device, GRASS_ARGS_BYTES as u64, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &args, GRASS_ARGS_BYTES as u64)?;
        let shadow_bytes =
            u64::from(field.shadow_capacity) * std::mem::size_of::<GpuGrassBlade>() as u64;
        let shadow_blades = create_uav_buffer(device, shadow_bytes, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &shadow_blades, shadow_bytes)?;
        let shadow_args =
            create_uav_buffer(device, GRASS_ARGS_BYTES as u64, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &shadow_args, GRASS_ARGS_BYTES as u64)?;
        let bend_bytes = (GRASS_BEND_WORDS * 4) as u64;
        let bend_field = create_uav_buffer(device, bend_bytes, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &bend_field, bend_bytes)?;
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let heights = upload_buffer(alloc, bytemuck::cast_slice(&field.buffers.heights), read)?;
        let masks = upload_buffer(
            alloc,
            bytemuck::cast_slice(field.buffers.bound_mask_words()),
            read,
        )?;

        let params_size = std::mem::size_of::<GrassParams>();
        Ok(Self {
            field,
            pipelines,
            draw_signature,
            blades,
            args,
            shadow_blades,
            shadow_args,
            bend_field,
            heights,
            masks,
            params: MappedBlocks::new(alloc, params_size, "map grass params")?,
            shadow_params: MappedBlocks::new(alloc, params_size, "map grass shadow params")?,
            bend_params: MappedBlocks::new(
                alloc,
                std::mem::size_of::<GrassBendParams>(),
                "map grass bend params",
            )?,
            history: Cell::new(GrassHistory::default()),
        })
    }
}

// One run of the blade kernel, as the frame's passes read it.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GrassPlacement {
    params_gva: u64,
    args_offset: u64,
    dispatch: [u32; 3],
}

impl GrassPlacement {
    fn of(params_gva: u64, pass: &GrassPass) -> Self {
        Self {
            params_gva,
            args_offset: grass_args_offset(pass.params.args_slot) as u64,
            dispatch: pass.dispatch,
        }
    }
}

// One frame's grass inputs, prepared before the fan-out so every pass that
// draws or dispatches grass reads the same blocks.
pub(in crate::directx) struct GrassFrame {
    view: GrassPlacement,
    // The cascade's run, when the grass casts this frame.
    shadow: Option<GrassPlacement>,
    bend_gva: u64,
    bend_dispatch: [u32; 3],
}

// What a frame's grass is prepared from.
pub(in crate::directx) struct GrassRequest<'a> {
    pub frame_idx: usize,
    pub cam_pos: [f32; 3],
    // The unjittered view-projection.
    pub vp: [[f32; 4]; 4],
    pub elapsed: f32,
    // The grass casts into the nearest cascade this frame.
    pub cast: bool,
    pub benders: &'a [GrassBender],
}

impl DxContext {
    // Whether this frame's grass casts into the nearest cascade: shadows are on
    // and that cascade re-renders.
    pub(in crate::directx) fn grass_casts(&self) -> bool {
        self.grass.is_some() && !self.shadow.dsvs.is_empty() && self.shadow.render_mask & 1 != 0
    }

    // Write this frame's grass blocks for `request`, advancing the field's
    // history. `None` when the world grows no grass.
    pub(in crate::directx) fn prepare_grass_frame(
        &self,
        request: GrassRequest<'_>,
    ) -> Option<GrassFrame> {
        let grass = self.grass.as_ref()?;
        let GrassRequest {
            frame_idx,
            cam_pos,
            vp,
            elapsed,
            cast,
            benders,
        } = request;
        // The pyramid holds last frame's depth once one has been built, tested
        // through the view-projection the draw cull tests through.
        let hiz = self
            .cull
            .hiz
            .as_ref()
            .filter(|_| self.cull.hiz_valid.get())
            .map(|h| GrassHiz {
                prev_vp: self.cull.prev_view_proj.get(),
                size: [h.width as f32, h.height as f32],
                mip_count: h.mip_count,
            });
        let inputs = GrassFrameInputs {
            camera: GrassCamera {
                position: cam_pos,
                vp,
                hiz,
            },
            elapsed,
            shadow: cast.then(|| GrassShadowView {
                vp: self.shadow.uniforms.light_vps[0],
                to_light: self.shadow.light_dir,
            }),
            benders,
        };
        let mut history = grass.history.get();
        let frame = grass.field.frame(&inputs, &mut history);
        grass.history.set(history);
        let view_gva = grass.params.write(frame_idx, &frame.view.params);
        let shadow = frame.shadow.map(|pass| {
            GrassPlacement::of(grass.shadow_params.write(frame_idx, &pass.params), &pass)
        });
        Some(GrassFrame {
            view: GrassPlacement::of(view_gva, &frame.view),
            shadow,
            bend_gva: grass.bend_params.write(frame_idx, &frame.bend.params),
            bend_dispatch: frame.bend.dispatch,
        })
    }

    // Encode the `GrassBend` node: relax the bend field and stamp this frame's
    // footprints into it. The field is already in UNORDERED_ACCESS.
    pub(in crate::directx) fn encode_grass_bend(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame: &GrassFrame,
    ) {
        let Some(grass) = &self.grass else {
            return;
        };
        let [x, y, z] = frame.bend_dispatch;
        // SAFETY: the command list is in the recording state, and every resource these commands
        // name is live for the call.
        unsafe {
            cmd.SetComputeRootSignature(&grass.pipelines.bend_root_sig);
            cmd.SetPipelineState(&grass.pipelines.bend);
            cmd.SetComputeRootConstantBufferView(0, frame.bend_gva);
            cmd.SetComputeRootUnorderedAccessView(1, com::gpu_va(&grass.bend_field));
            cmd.Dispatch(x, y, z);
        }
    }

    // Encode the `GrassShadow` node: place the blades the nearest cascade
    // draws.
    pub(in crate::directx) fn encode_grass_shadow(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame: &GrassFrame,
    ) {
        let (Some(grass), Some(shadow)) = (&self.grass, frame.shadow.as_ref()) else {
            return;
        };
        self.place_blades(
            cmd,
            grass,
            shadow,
            (&grass.shadow_blades, &grass.shadow_args),
        );
    }

    // Draw the cascade's blades into the nearest cascade's slice, `dsv`, with
    // the coarsest strip, under their own root signature.
    pub(in crate::directx) fn encode_grass_shadow_draw(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
        frame: &GrassFrame,
    ) {
        let (Some(grass), Some(shadow)) = (&self.grass, frame.shadow.as_ref()) else {
            return;
        };
        let coarsest = shadow.args_offset + ((GRASS_LOD_COUNT - 1) * GRASS_ARGS_STRIDE) as u64;
        self.timed(cmd, frame_idx, PassId::GrassShadowDraw, || {
            // SAFETY: the command list is in the recording state, and every resource these
            // commands name is live for the call; the args buffer is in INDIRECT_ARGUMENT for
            // this pass and the offset names one whole record of it.
            unsafe {
                cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
                cmd.SetGraphicsRootSignature(&grass.pipelines.shadow_root_sig);
                cmd.SetPipelineState(&grass.pipelines.shadow);
                cmd.SetGraphicsRootConstantBufferView(0, shadow.params_gva);
                cmd.SetGraphicsRootShaderResourceView(1, com::gpu_va(&grass.shadow_blades));
                cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
                cmd.ExecuteIndirect(
                    &grass.draw_signature,
                    1,
                    &grass.shadow_args,
                    coarsest,
                    None::<&ID3D12Resource>,
                    0,
                );
                cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            }
        });
        self.inc_draw_calls(1);
    }

    // Encode the `Grass` node: place, cull and append this frame's blades. Both
    // buffers are already in UNORDERED_ACCESS for the write.
    pub(in crate::directx) fn encode_grass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame: &GrassFrame,
    ) {
        let Some(grass) = &self.grass else {
            return;
        };
        self.place_blades(cmd, grass, &frame.view, (&grass.blades, &grass.args));
    }

    // One run of the blade kernel for `placement`, appending to `blades` and
    // filling `args`.
    fn place_blades(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        grass: &GrassResources,
        placement: &GrassPlacement,
        (blades, args): (&ID3D12Resource, &ID3D12Resource),
    ) {
        let [x, y, z] = placement.dispatch;
        // SAFETY: the command list is in the recording state, and every resource these commands
        // name is live for the call.
        unsafe {
            cmd.SetComputeRootSignature(&grass.pipelines.generate_root_sig);
            cmd.SetPipelineState(&grass.pipelines.generate);
            cmd.SetComputeRootConstantBufferView(0, placement.params_gva);
            cmd.SetComputeRootUnorderedAccessView(1, com::gpu_va(blades));
            cmd.SetComputeRootUnorderedAccessView(2, com::gpu_va(args));
            cmd.SetComputeRootShaderResourceView(3, com::gpu_va(&grass.heights));
            cmd.SetComputeRootShaderResourceView(4, com::gpu_va(&grass.masks));
            cmd.SetComputeRootShaderResourceView(KERNEL_BEND_PARAM, com::gpu_va(&grass.bend_field));
            // The pyramid's descriptor lives in the shader-visible SRV heap.
            // Bound whenever it exists so the table always points at a live
            // descriptor; the block's `hiz_enabled` gates the reads.
            if let Some(hiz) = &self.cull.hiz {
                cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
                cmd.set_compute_srv_table(KERNEL_HIZ_PARAM, hiz.srv_gpu);
            }
            cmd.Dispatch(x, y, z);
        }
    }

    // Draw the blades into the G-buffer pre-pass bound on `cmd`, under the
    // pre-pass root signature: the main view at b1 rasterizes them, the
    // pre-pass view at `gb_view_gva` gives their motion.
    pub(in crate::directx) fn encode_grass_prepass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        gb_view_gva: u64,
        frame: &GrassFrame,
    ) {
        let (Some(grass), Some(root_sig)) = (&self.grass, self.cull.prepass_root_sig.as_ref())
        else {
            return;
        };
        let main_view_gva = com::gpu_va(&self.uniforms.view_ubo_resources[frame_idx]);
        self.timed(cmd, frame_idx, PassId::GrassPrepass, || {
            // SAFETY: the command list is in the recording state, and every resource these
            // commands name is live for the call; the root signature declares each parameter set.
            unsafe {
                cmd.SetGraphicsRootSignature(root_sig);
                cmd.SetPipelineState(&grass.pipelines.prepass);
                cmd.SetGraphicsRootConstantBufferView(1, main_view_gva);
                cmd.SetGraphicsRootConstantBufferView(2, gb_view_gva);
                cmd.SetGraphicsRootConstantBufferView(
                    PREPASS_GRASS_PARAMS_PARAM,
                    frame.view.params_gva,
                );
                cmd.SetGraphicsRootShaderResourceView(
                    PREPASS_GRASS_BLADES_PARAM,
                    com::gpu_va(&grass.blades),
                );
            }
            self.draw_grass(cmd, grass, frame);
        });
    }

    // Draw the lit blades into the main pass bound on `cmd`, whose root
    // signature and every parameter the surfaces set are still in place.
    pub(in crate::directx) fn encode_grass_main(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        frame: &GrassFrame,
    ) {
        let Some(grass) = &self.grass else {
            return;
        };
        self.timed(cmd, frame_idx, PassId::GrassDraw, || {
            // SAFETY: the command list is in the recording state, and every resource these
            // commands name is live for the call; the bound root signature declares both
            // parameters.
            unsafe {
                cmd.SetPipelineState(&grass.pipelines.main);
                cmd.SetGraphicsRootConstantBufferView(
                    MAIN_GRASS_PARAMS_PARAM,
                    frame.view.params_gva,
                );
                cmd.SetGraphicsRootShaderResourceView(
                    MAIN_GRASS_BLADES_PARAM,
                    com::gpu_va(&grass.blades),
                );
            }
            self.draw_grass(cmd, grass, frame);
        });
    }

    // The draws both passes issue, one per detail level: a strip per visible
    // blade. Each level's first vertex rides the b0 root constant, which the
    // pass's surfaces set per command, so it is free here. The pass's list
    // topology is put back for whatever draws after it.
    fn draw_grass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        grass: &GrassResources,
        frame: &GrassFrame,
    ) {
        // SAFETY: the command list is in the recording state under a root signature whose
        // parameter 0 is one root-constant DWORD; the args buffer is in INDIRECT_ARGUMENT for this
        // pass and each offset names one whole record of it.
        unsafe {
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            for lod in 0..GRASS_LOD_COUNT {
                let first_vertex = lod as u32 * GRASS_LOD_VERTEX_STRIDE;
                cmd.set_graphics_root_constants(FIRST_VERTEX_PARAM, &first_vertex);
                cmd.ExecuteIndirect(
                    &grass.draw_signature,
                    1,
                    &grass.args,
                    frame.view.args_offset + (lod * GRASS_ARGS_STRIDE) as u64,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
        self.inc_draw_calls(GRASS_LOD_COUNT as u32);
    }

    // `record` bracketed by `pass`'s timestamps inside the enclosing pass's
    // command list.
    fn timed(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        pass: PassId,
        record: impl FnOnce(),
    ) {
        let (start, end) = pass_timing::pass_pair(frame_idx, pass);
        let heap = self.timestamps.query_heap.as_ref();
        if let Some(heap) = heap {
            // SAFETY: the command list is in the recording state and the query heap is live.
            unsafe { cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start) };
        }
        record();
        if let Some(heap) = heap {
            // SAFETY: as above.
            unsafe { cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end) };
        }
    }
}
