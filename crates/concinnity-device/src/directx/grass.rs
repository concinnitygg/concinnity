//! The grass field on DirectX 12 (see `concinnity_core::render::grass`): the
//! kernel that places this frame's visible blades, and the two indirect draws
//! that render them at the tail of the G-buffer pre-pass and the main pass.
//!
//! Both draws run under their pass's own root signature, which carries the
//! grass block and the blade buffer as two extra root parameters, so the lit
//! blades shade on the lights, shadows and environment the surfaces bound. The
//! visible-blade and draw-argument buffers rest in UNORDERED_ACCESS; the graph
//! moves them to the vertex-read and indirect-argument states the draws need.

use std::cell::Cell;

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::grass::{GrassCamera, GrassField};
use concinnity_core::render::pass_timing;
use concinnity_core::render::render_graph::PassId;
use concinnity_core::render::uniforms::grass::{
    GRASS_ARGS_BYTES, GRASS_ARGS_STRIDE, GpuGrassBlade, GrassParams, grass_args_offset,
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
use super::error::map_hresult;
use super::pso::{Blend, Depth, GraphicsPso, compute_pso};
use super::root_sig::{RootSig, Visibility};
use super::texture::{HDR_FORMAT, create_uav_buffer};

/// Where the main pass's root signature carries the grass block (b7) and the
/// visible blades (t23).
pub(in crate::directx) const MAIN_GRASS_PARAMS_PARAM: u32 = 21;
pub(in crate::directx) const MAIN_GRASS_BLADES_PARAM: u32 = 22;
/// The same pair in the G-buffer pre-pass's root signature.
pub(in crate::directx) const PREPASS_GRASS_PARAMS_PARAM: u32 = 9;
pub(in crate::directx) const PREPASS_GRASS_BLADES_PARAM: u32 = 10;

// The kernel's root signature: the block at b0, the blades it appends at u1 and
// the draw arguments at u2, as `grass.hlsl` declares them.
fn create_generate_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .uav(1, Visibility::All)
        .uav(2, Visibility::All)
        .build(device, "grass generate root sig")
}

// One non-indexed draw per command, matching the slots the kernel fills. It
// sets no root argument, so it needs no root signature.
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

// The kernel and the two draw pipelines; rebuilt as a set on a shader reload.
pub(in crate::directx) struct GrassPipelines {
    generate_root_sig: ID3D12RootSignature,
    generate: ID3D12PipelineState,
    prepass: ID3D12PipelineState,
    main: ID3D12PipelineState,
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
            prepass,
            main,
        })
    }
}

// The grass field and what draws it: built once at init when the world grows
// one and the GPU-driven main pass exists to draw it in.
pub(in crate::directx) struct GrassResources {
    pub(in crate::directx) field: GrassField,
    pub(in crate::directx) pipelines: GrassPipelines,
    draw_signature: ID3D12CommandSignature,
    // The visible blades the kernel appends, `field.capacity` of them.
    pub(in crate::directx) blades: ID3D12Resource,
    // Two slots of draw arguments; see `GRASS_ARGS_SLOTS`.
    pub(in crate::directx) args: ID3D12Resource,
    // One `GrassParams` block per frame in flight, persistently mapped.
    params: Vec<PooledBuffer>,
    params_ptrs: Vec<*mut u8>,
    // Frames the kernel has run, which picks the slot it fills. Advanced from
    // `&self` on the render thread before the fan-out.
    runs: Cell<u32>,
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
        let blade_bytes = field.capacity as u64 * std::mem::size_of::<GpuGrassBlade>() as u64;
        let blades = create_uav_buffer(device, blade_bytes, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &blades, blade_bytes)?;
        let args = create_uav_buffer(device, GRASS_ARGS_BYTES as u64, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &args, GRASS_ARGS_BYTES as u64)?;

        let size = align256(std::mem::size_of::<GrassParams>() as u64);
        let frames = alloc.frames_in_flight();
        let mut params = Vec::with_capacity(frames);
        let mut params_ptrs = Vec::with_capacity(frames);
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
                .map_err(|e| map_hresult(e.code(), "map grass params"))?;
            params_ptrs.push(ptr as *mut u8);
            params.push(buf);
        }
        Ok(Self {
            field,
            pipelines,
            draw_signature,
            blades,
            args,
            params,
            params_ptrs,
            runs: Cell::new(0),
        })
    }
}

// One frame's grass inputs, prepared before the fan-out so every pass that
// draws or dispatches grass reads the same block.
pub(in crate::directx) struct GrassFrame {
    params_gva: u64,
    args_offset: u64,
    dispatch: [u32; 3],
}

impl DxContext {
    // Write this frame's grass block for a camera at `cam_pos` seeing through
    // the unjittered `vp`, advancing the draw-argument slot. `None` when the
    // world grows no grass.
    pub(in crate::directx) fn prepare_grass_frame(
        &self,
        frame_idx: usize,
        cam_pos: [f32; 3],
        vp: [[f32; 4]; 4],
    ) -> Option<GrassFrame> {
        let grass = self.grass.as_ref()?;
        let runs = grass.runs.get();
        grass.runs.set(runs.wrapping_add(1));
        let camera = GrassCamera {
            position: cam_pos,
            vp,
        };
        let params = grass.field.frame_params(&camera, runs);
        // SAFETY: the destination is the persistent mapping of this frame's UPLOAD-heap constant
        // buffer, sized for the payload, which the GPU no longer reads once this frame's slot has
        // been waited on; the source is a separate live value, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &params as *const GrassParams as *const u8,
                grass.params_ptrs[frame_idx],
                std::mem::size_of::<GrassParams>(),
            );
        }
        Some(GrassFrame {
            params_gva: com::gpu_va(&grass.params[frame_idx]),
            args_offset: grass_args_offset(params.args_slot) as u64,
            dispatch: grass.field.dispatch(cam_pos),
        })
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
        let [x, y, z] = frame.dispatch;
        // SAFETY: the command list is in the recording state, and every resource these commands
        // name is live for the call.
        unsafe {
            cmd.SetComputeRootSignature(&grass.pipelines.generate_root_sig);
            cmd.SetPipelineState(&grass.pipelines.generate);
            cmd.SetComputeRootConstantBufferView(0, frame.params_gva);
            cmd.SetComputeRootUnorderedAccessView(1, com::gpu_va(&grass.blades));
            cmd.SetComputeRootUnorderedAccessView(2, com::gpu_va(&grass.args));
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
                cmd.SetGraphicsRootConstantBufferView(PREPASS_GRASS_PARAMS_PARAM, frame.params_gva);
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
                cmd.SetGraphicsRootConstantBufferView(MAIN_GRASS_PARAMS_PARAM, frame.params_gva);
                cmd.SetGraphicsRootShaderResourceView(
                    MAIN_GRASS_BLADES_PARAM,
                    com::gpu_va(&grass.blades),
                );
            }
            self.draw_grass(cmd, grass, frame);
        });
    }

    // The one indirect draw both passes issue: a strip per visible blade. The
    // pass's list topology is put back for whatever draws after it.
    fn draw_grass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        grass: &GrassResources,
        frame: &GrassFrame,
    ) {
        // SAFETY: the command list is in the recording state; the args buffer is in
        // INDIRECT_ARGUMENT for this pass and the offset names one of its whole slots.
        unsafe {
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            cmd.ExecuteIndirect(
                &grass.draw_signature,
                1,
                &grass.args,
                frame.args_offset,
                None::<&ID3D12Resource>,
                0,
            );
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
        self.inc_draw_calls(1);
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
