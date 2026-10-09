//! Unified geometry G-buffer pre-pass for the D3D12 backend. One jittered
//! traversal of the GPU cull's records rasterizes into a single MRT:
//!
//!   target 0  RGBA16F  view-space normal (rgb) + positive linear view depth (a)
//!   target 1  R8       perceptual roughness
//!   target 2  RG16F    screen-space motion (prev_uv - cur_uv)
//!
//! plus a private single-sample depth buffer. Every screen-space consumer (SSR
//! resolve, SSAO kernel/blur, SSGI trace/composite, TAA resolve, FSR upscaler)
//! reads this one output instead of re-rasterizing, replacing the separate
//! SSR, SSAO and velocity pre-passes. Rasterization uses the jittered
//! VP (matching the main pass coverage); the motion vector derives from the
//! un-jittered current / previous VPs in-shader so projection jitter never
//! contaminates motion. Mirrors src/metal/post/gbuffer.rs.

use concinnity_core::render::depth::DEPTH_CLEAR;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::uniforms::ModelHistoryParams;
use concinnity_core::render::view_history::{ViewFrame, ViewHistory};
use std::cell::RefCell;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::gbuffer_sky::GbufferSky;
use crate::directx::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::com;
use crate::directx::context::{DxContext, align256, dump_on_err};
use crate::directx::descriptor_slot::{DescriptorTables, SrvSlot};
use crate::directx::error::map_hresult;
use crate::directx::pso::{Blend, Depth, GraphicsPso, compute_pso};
use crate::directx::root_constants::RootConstants;
use crate::directx::root_sig::{Range, RootSig, Visibility};
use crate::directx::texture::{create_main_depth_texture, write_format_rtv, write_format_srv};

// Normal+depth target: rgb = unit view-space normal, a = positive linear view
// depth (-view_z). Alpha 0 (cleared background) marks "no geometry". Matches
// the SSR / SSAO G-buffer so the resolve / kernel maths is byte-identical.
pub(crate) const GBUFFER_NORMAL_DEPTH_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Single-channel perceptual roughness. 1.0 = fully rough (cleared background),
// 0.0 = mirror.
pub(crate) const GBUFFER_ROUGHNESS_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8_UNORM;

// Screen-space motion (prev_uv - cur_uv). Cleared to 0 (no motion).
pub(crate) const GBUFFER_VELOCITY_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16_FLOAT;

// Background roughness the prepass clears the roughness target to: fully rough,
// so untouched pixels emit no reflection. The per-frame clear uses it here; the
// matching optimized clear now comes from the graph's desc, and a test pins the
// two together.
pub(in crate::directx) const GBUFFER_ROUGHNESS_CLEAR: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

// Size of the per-frame view UBO: jittered_vp + cur_vp + prev_vp + view_mat,
// the previous clock and camera position, and the motion flag. Matches the
// `GbView` cbuffer in `gbuffer_common.hlsl`.
const GBUFFER_VIEW_UBO_SIZE: u64 = std::mem::size_of::<GBufferView>() as u64;

// `GBufferView` (the `GbView` cbuffer) is a GPU-free layout struct that lives in
// `core::render`; re-export it so
// `crate::directx::post::gbuffer::GBufferView` is unchanged.
pub(in crate::directx) use concinnity_core::render::uniforms::GBufferView;

// Root signatures

// The pre-pass's three color targets, in attachment order.
pub(in crate::directx) fn gbuffer_targets(pso: GraphicsPso<'_>) -> GraphicsPso<'_> {
    pso.target(GBUFFER_NORMAL_DEPTH_FORMAT, Blend::Opaque)
        .target(GBUFFER_ROUGHNESS_FORMAT, Blend::Opaque)
        .target(GBUFFER_VELOCITY_FORMAT, Blend::Opaque)
}

// Vertex input layout for the G-buffer pre-pass: the main pass's attributes on
// slot 0, which the vertex hook reads in full, plus the previous-frame position
// on slot 1. Both slots carry the 56-byte `Vertex`; the static prefix binds the
// static VB to both slots (prev_pos == cur_pos), the skinned tail binds the
// current deformed buffer to slot 0 and the previous-frame deformed buffer to
// slot 1.
fn prepass_input_layout() -> Vec<D3D12_INPUT_ELEMENT_DESC> {
    let mut layout = crate::directx::pipeline::main_input_layout();
    layout.push(D3D12_INPUT_ELEMENT_DESC {
        SemanticName: windows::core::s!("PREVPOSITION"),
        SemanticIndex: 0,
        Format: DXGI_FORMAT_R32G32B32_FLOAT,
        InputSlot: 1,
        AlignedByteOffset: 0,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        InstanceDataStepRate: 0,
    });
    layout
}

// Root signature every shader bucket's G-buffer pre-pass PSO binds: the
// `SURFACE_PREPASS` declarations of `main_bindless.hlsl`, which keep the main
// pass's registers for what both read. [0] is the per-command b0 object-id root
// constant (set by the `ExecuteIndirect` command signature, so it MUST stay at
// root parameter 0).
pub(in crate::directx) fn create_prepass_root_signature(
    device: &ID3D12Device,
) -> RenderResult<ID3D12RootSignature> {
    use Visibility::{All, Pixel, Vertex};
    RootSig::new()
        // [0] b0: object id (set per command by the command sig).
        .constant_dwords(0, 1, All)
        // [1] b1: the main pass's view block, which the vertex hook reads.
        .cbv(1, Vertex)
        // [2] b6: GbView (motion matrices, view matrix, previous clock).
        .cbv(6, All)
        // [3] t3: per-frame StructuredBuffer<GpuObjectData>.
        .srv(3, All)
        // [4] t21: the previous frame's model-history slot.
        .srv(21, Vertex)
        // [5] t22: this frame's draw args, read for `NO_HISTORY`.
        .srv(22, Vertex)
        // [6] t20: the material parameter table the vertex hook may read.
        .srv(20, Vertex)
        // [7] the unbounded bindless texture pool at t0, space1.
        .table(&[Range::bindless_srv(1)], Pixel)
        // [8] linear repeat (s1) + cube sampler (s2).
        .sampler_table(1, 2, Pixel)
        .input_layout()
        .build(device, "gbuffer prepass root sig")
}

// The pre-pass root parameters `encode_gbuffer_prepass_gpu_driven` binds.
const PREPASS_VIEW_PARAM: u32 = 1;
const PREPASS_GB_VIEW_PARAM: u32 = 2;
const PREPASS_OBJECTS_PARAM: u32 = 3;
const PREPASS_PREV_MODELS_PARAM: u32 = 4;
const PREPASS_DRAW_ARGS_PARAM: u32 = 5;
const PREPASS_MATERIAL_PARAMS_PARAM: u32 = 6;
const PREPASS_POOL_PARAM: u32 = 7;
const PREPASS_SAMPLERS_PARAM: u32 = 8;

// One shader bucket's G-buffer pre-pass PSO over its `vertex_prepass_bindless`
// / `fragment_prepass_bindless` pair: the three MRT targets over a private
// single-sample depth buffer, with the main pass's no-cull rasterizer and
// depth-write test so the G-buffer matches the main pass's visible surfaces.
pub(in crate::directx) fn create_prepass_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    let layout = prepass_input_layout();
    gbuffer_targets(GraphicsPso::new(root_sig, vs, ps).input_layout(&layout))
        .depth(DXGI_FORMAT_D32_FLOAT, Depth::write())
        .build(device, "gbuffer prepass")
}

// Threads per group, matching `[numthreads(64, 1, 1)]` in model_history.hlsl.
const MODEL_HISTORY_THREADGROUP: u32 = 64;

// A UAV barrier over one buffer: orders its shader writes against later
// accesses without claiming a state transition a buffer does not have.
fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: com::borrowed(resource),
            }),
        },
    }
}

// Root signature for the model-history snapshot kernel: the `CN_BACKEND_DIRECTX` arm of
// `model_history.hlsl` declares b0/t0/u0, which is what these three parameters
// bind.
fn create_model_history_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        // [0] b0: ModelHistoryParams (record count + padding).
        .constants::<ModelHistoryParams>(0, Visibility::All)
        // [1] t0: this frame's StructuredBuffer<GpuObjectData>.
        .srv(0, Visibility::All)
        // [2] u0: this frame's model-history slot.
        .uav(0, Visibility::All)
        .input_layout()
        .build(device, "model history root sig")
}

// Build the model-history snapshot kernel: the compute PSO and its root
// signature. Called under the same gate as the pre-pass it feeds.
pub(in crate::directx) fn build_model_history(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    hot_reload: bool,
) -> RenderResult<(ID3D12RootSignature, ID3D12PipelineState)> {
    let cs = builtin_shaders::MODEL_HISTORY.compile(hot_reload)?;
    let root_sig = dump_on_err(info_queue, create_model_history_root_signature(device))?;
    let pso = dump_on_err(
        info_queue,
        compute_pso(device, &root_sig, &cs, "model history"),
    )?;
    Ok((root_sig, pso))
}

// Descriptor-slot handles for the three G-buffer SRVs, minted by the caller
// (which owns the heap layout). `Copy` so the caller can both pass it to `new`
// and stash a copy for the live `apply_quality_settings` rebuild.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GbufferSlots {
    pub normal_depth_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub normal_depth_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, SrvSlot),
    pub roughness_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub roughness_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, SrvSlot),
    pub velocity_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub velocity_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, SrvSlot),
    pub depth_dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
}

// Unified G-buffer resources held by `DxContext` when any screen-space consumer
// (SSR, SSGI, SSAO, TAA, or temporal upscaling) is enabled. Drops cleanly with
// the context: every D3D12 object is COM-refcounted.
pub(in crate::directx) struct GbufferResources {
    // MRT targets + their private single-sample depth.
    pub(in crate::directx) normal_depth: ID3D12Resource,
    pub(in crate::directx) normal_depth_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) normal_depth_srv_gpu: SrvSlot,
    pub(in crate::directx) roughness: ID3D12Resource,
    pub(in crate::directx) roughness_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) roughness_srv_gpu: SrvSlot,
    pub(in crate::directx) velocity: ID3D12Resource,
    pub(in crate::directx) velocity_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) velocity_srv_gpu: SrvSlot,
    pub(in crate::directx) depth: ID3D12Resource,
    pub(in crate::directx) depth_dsv: D3D12_CPU_DESCRIPTOR_HANDLE,

    // Per-frame view UBO (jittered_vp + cur_vp + prev_vp + view), mapped.
    pub(in crate::directx) view_ubo_resources: Vec<PooledBuffer>,
    pub(in crate::directx) view_ubo_ptrs: Vec<*mut u8>,

    // Last frame's un-jittered VP, owned here so the velocity channel works for
    // any consumer (TAA or FSR) independent of whether engine-TAA is on. The
    // per-object half of the same history is the GPU-filled model-history ring.
    pub(in crate::directx) view_history: RefCell<ViewHistory>,

    // The sky's motion behind the geometry.
    pub(in crate::directx) sky: GbufferSky,
}

// The device the G-buffer builder allocates against.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GbufferDeviceCtx<'a> {
    pub alloc: &'a DeviceAllocator,
    pub info_queue: Option<&'a ID3D12InfoQueue>,
    pub hot_reload: bool,
}

// The three color targets the transient pool owns, handed to the G-buffer at
// build / resize. The feature no longer creates them: they are graph resources
// the pool places, so it only writes their views. `depth` is absent because it
// stays feature-owned (see `transient_pool::pooled`).
#[derive(Clone)]
pub(in crate::directx) struct GbufferPooled {
    pub normal_depth: ID3D12Resource,
    pub roughness: ID3D12Resource,
    pub velocity: ID3D12Resource,
}

// Render-target extent plus which pipeline variants the G-buffer pre-pass builds.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GbufferExtent {
    pub width: u32,
    pub height: u32,
}

// The RTV + SRV descriptor slots the three pooled color targets are viewed
// through. Built from `GbufferSlots` at construction and from the stored
// handles at resize, so one routine writes the views in both paths.
struct GbufferViewSlots {
    normal_depth: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE),
    roughness: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE),
    velocity: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE),
}

// Point the pre-reserved RTV / SRV slots at the pool's placed resources and
// hand back a reference to each. Every pool rebuild (init, resize, and a
// quality toggle that changes the pooled set) relocates these resources, so
// every one of those paths must call this or the descriptors name freed memory.
fn write_pooled_views(
    device: &ID3D12Device,
    slots: GbufferViewSlots,
    pooled: &GbufferPooled,
) -> (ID3D12Resource, ID3D12Resource, ID3D12Resource) {
    for (res, (rtv, srv), format) in [
        (
            &pooled.normal_depth,
            slots.normal_depth,
            GBUFFER_NORMAL_DEPTH_FORMAT,
        ),
        (&pooled.roughness, slots.roughness, GBUFFER_ROUGHNESS_FORMAT),
        (&pooled.velocity, slots.velocity, GBUFFER_VELOCITY_FORMAT),
    ] {
        write_format_rtv(device, res, rtv, format);
        write_format_srv(device, res, srv, format);
    }
    (
        pooled.normal_depth.clone(),
        pooled.roughness.clone(),
        pooled.velocity.clone(),
    )
}

impl GbufferResources {
    pub(in crate::directx) fn new(
        ctx: GbufferDeviceCtx,
        extent: GbufferExtent,
        slots: GbufferSlots,
        pooled: &GbufferPooled,
    ) -> RenderResult<Self> {
        let GbufferDeviceCtx {
            alloc,
            info_queue,
            hot_reload,
        } = ctx;
        let device = alloc.device();
        let GbufferExtent { width, height } = extent;
        // The three color targets come from the transient pool; this only
        // writes their views into the pre-reserved descriptor slots.
        let (normal_depth, roughness, velocity) = write_pooled_views(
            device,
            GbufferViewSlots {
                normal_depth: (slots.normal_depth_rtv, slots.normal_depth_srv.0),
                roughness: (slots.roughness_rtv, slots.roughness_srv.0),
                velocity: (slots.velocity_rtv, slots.velocity_srv.0),
            },
            pooled,
        );

        let depth = create_main_depth_texture(device, width, height, slots.depth_dsv, 1, true)?;

        // Per-frame view UBO.
        let view_size = align256(GBUFFER_VIEW_UBO_SIZE);
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
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map gbuffer view ubo"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        let sky = GbufferSky::build(device, info_queue, hot_reload)?;

        Ok(Self {
            normal_depth,
            normal_depth_rtv: slots.normal_depth_rtv,
            normal_depth_srv_gpu: slots.normal_depth_srv.1,
            roughness,
            roughness_rtv: slots.roughness_rtv,
            roughness_srv_gpu: slots.roughness_srv.1,
            velocity,
            velocity_rtv: slots.velocity_rtv,
            velocity_srv_gpu: slots.velocity_srv.1,
            depth,
            depth_dsv: slots.depth_dsv,
            view_ubo_resources,
            view_ubo_ptrs,
            view_history: RefCell::new(ViewHistory::default()),
            sky,
        })
    }

    // Re-point the MRT views at the rebuilt pool and recreate the private depth
    // at a new resolution. The descriptor *slots* stay put; only the resources
    // behind them change, so no consumer needs a re-bind.
    //
    // The caller must have rebuilt the transient pool first: `pooled` names the
    // new placed resources, and the old ones are freed with the pool.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
        srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
        srv_gpu_base: SrvSlot,
        pooled: &GbufferPooled,
    ) -> RenderResult<()> {
        self.repoint_pooled(device, srv_cpu_base, srv_gpu_base, pooled);
        self.depth = create_main_depth_texture(device, width, height, self.depth_dsv, 1, true)?;
        Ok(())
    }

    // Re-point the three pooled color views after a pool rebuild that did not
    // change the resolution -- a quality toggle that adds or removes another
    // pooled resource relocates these too, because the pool repacks every slot.
    pub(in crate::directx) fn repoint_pooled(
        &mut self,
        device: &ID3D12Device,
        srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
        srv_gpu_base: SrvSlot,
        pooled: &GbufferPooled,
    ) {
        let srv_cpu = |gpu: SrvSlot| gpu.cpu_in(srv_cpu_base, srv_gpu_base);
        let (normal_depth, roughness, velocity) = write_pooled_views(
            device,
            GbufferViewSlots {
                normal_depth: (self.normal_depth_rtv, srv_cpu(self.normal_depth_srv_gpu)),
                roughness: (self.roughness_rtv, srv_cpu(self.roughness_srv_gpu)),
                velocity: (self.velocity_rtv, srv_cpu(self.velocity_srv_gpu)),
            },
            pooled,
        );
        self.normal_depth = normal_depth;
        self.roughness = roughness;
        self.velocity = velocity;
    }
}

// Camera + view-projection inputs for the G-buffer pre-pass. The two VPs drive
// rasterization (jittered) and motion vectors (un-jittered current vs previous).
pub(in crate::directx) struct GbufferPrepassView {
    // Jittered view-projection (the sky's rasterization target).
    pub jittered_vp: [[f32; 4]; 4],
    // Un-jittered current view-projection (motion vectors).
    pub cur_vp: [[f32; 4]; 4],
    // The main pass's clock and camera position, which a vertex hook reads.
    pub elapsed: f32,
    pub cam_pos: [f32; 3],
}

// Per-frame decisions the G-buffer pre-pass takes as data, made on the main
// thread before the encode fan-out.
pub(in crate::directx) struct GbufferPrepassFrame {
    // A consumer (TAA, FSR or SSGI) reads motion; when false, cur == prev so the
    // motion channel is a harmless zero.
    pub velocity_active: bool,
    // The model-history snapshot fills every ring slot rather than only this
    // frame's, because a rebuild left the ring unwritten.
    pub prime_history: bool,
}

impl DxContext {
    // Whether anything reprojects through the pre-pass's motion channel this
    // frame: TAA, the FSR upscaler, or the SSGI accumulation.
    pub(in crate::directx) fn reads_motion(&self) -> bool {
        self.taa.is_some()
            || self.upscale.backend.is_some()
            || self.ssgi.as_ref().is_some_and(|s| s.settings.contributes())
    }

    // Give the G-buffer whatever its pre-pass still lacks to draw with: the
    // model-history ring and its snapshot kernel, which init builds only for a
    // world that starts with a G-buffer consumer, and each shader bucket's
    // pre-pass PSO. Builds only what is missing, so it is safe to repeat. A
    // world with no GPU-driven pass has nothing to draw here.
    pub(in crate::directx) fn enable_gbuffer_prepass(&mut self) -> RenderResult<()> {
        if self.cull.main_bindless_pso.is_some() && self.cull.model_history_pso.is_none() {
            let device = self.hw.alloc.device();
            let (root_sig, pso) =
                build_model_history(device, self.hw.info_queue.as_ref(), self.hot_reload.enabled)?;
            let size =
                align256((self.cull.bucket_stride * std::mem::size_of::<[[f32; 4]; 4]>()) as u64);
            let mut ring = Vec::with_capacity(self.hw.frames());
            for _ in 0..self.hw.frames() {
                ring.push(crate::directx::texture::create_uav_buffer(
                    device,
                    size,
                    D3D12_RESOURCE_STATE_COMMON,
                )?);
            }
            self.cull.prev_model_buffers = ring;
            self.cull.model_history_root_sig = Some(root_sig);
            self.cull.model_history_pso = Some(pso);
            // Nothing has written the fresh ring, so the first pre-pass primes it.
            self.state.model_history.borrow_mut().request_prime();
        }
        self.sync_prepass_psos();
        Ok(())
    }

    // Encode the unified G-buffer pre-pass: one jittered traversal of the cull
    // records into the normal+depth / roughness / velocity MRT, then this
    // frame's model-history snapshot.
    pub(in crate::directx) fn encode_gbuffer_prepass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view: GbufferPrepassView,
        frame: GbufferPrepassFrame,
    ) {
        let GbufferPrepassView {
            jittered_vp,
            cur_vp,
            elapsed,
            cam_pos,
        } = view;
        let GbufferPrepassFrame {
            velocity_active,
            prime_history,
        } = frame;
        let gb = match &self.gbuffer {
            Some(g) => g,
            None => return,
        };

        // Upload this frame's view UBO. When velocity is inactive the previous
        // camera and clock equal the current ones, so instanced + sky motion
        // is zero and the surfaces skip reprojecting.
        let cur = ViewFrame {
            vp: cur_vp,
            elapsed,
            cam_pos,
        };
        let view_uni = GBufferView::new(
            jittered_vp,
            self.state.view.matrix,
            cur,
            gb.view_history.borrow().prev_or(cur),
            velocity_active,
        );
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const GBufferView as *const u8,
                gb.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<GBufferView>(),
            );
        }
        let view_gva = com::gpu_va(&gb.view_ubo_resources[frame_idx]);

        let w = self.targets.extent.render_width;
        let h = self.targets.extent.render_height;

        // The three color targets are one graph resource (`gbuffer`), so the
        // executor has already put them in RENDER_TARGET for this pass's write
        // and the consumers' barrier takes them back out. `gb.depth` is not part
        // of it and stays in DEPTH_WRITE throughout.
        let rtvs = [gb.normal_depth_rtv, gb.roughness_rtv, gb.velocity_rtv];
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(3, Some(rtvs.as_ptr()), false, Some(&gb.depth_dsv));
            // Cleared alpha 0 marks "no geometry"; roughness 1.0 = non-reflective
            // background; velocity 0 = no motion.
            cmd.ClearRenderTargetView(gb.normal_depth_rtv, &[0.0_f32; 4], None);
            cmd.ClearRenderTargetView(gb.roughness_rtv, &GBUFFER_ROUGHNESS_CLEAR, None);
            cmd.ClearRenderTargetView(gb.velocity_rtv, &[0.0_f32; 4], None);
            cmd.ClearDepthStencilView(gb.depth_dsv, D3D12_CLEAR_FLAG_DEPTH, DEPTH_CLEAR, 0, None);
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
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
        }

        // The pre-pass is GPU-driven: it reuses the main pass's per-frame indirect
        // command buffer (same camera frustum + active LOD) with two
        // `ExecuteIndirect` draws (static + instance prefix, then the skinned
        // tail over the deformed VB). With nothing to draw the pass is the
        // clears above, which is what "no geometry" means to every reader.
        self.encode_gbuffer_prepass_gpu_driven(cmd, frame_idx, view_gva, velocity_active);
        self.encode_raymarch_prepass(cmd, frame_idx, &view, &view_uni);
        // The sky keeps the "no geometry" depth and roughness and adds the
        // camera's motion where nothing was drawn.
        if self.draws_sky(self.state.view.mode) {
            gb.sky.encode(cmd, view_gva);
            self.inc_draw_calls(1);
        }

        // Snapshot this frame's models into this frame's history slot, AFTER the
        // pass above read the previous one -- which is what keeps a single frame
        // in flight (one slot, read then rewritten) correct.
        self.encode_model_history(cmd, frame_idx, prime_history);
    }

    // Dispatch the model-history snapshot: one thread per cull record copying
    // `objects[i].model` into this frame's history slot, or into every slot when
    // `prime` is set. The slot rests as a shader resource (the pre-pass reads it
    // through a root SRV) and is transitioned to a UAV for the write and back.
    fn encode_model_history(&self, cmd: &ID3D12GraphicsCommandList, frame_idx: usize, prime: bool) {
        let (Some(root_sig), Some(pso), true) = (
            self.cull.model_history_root_sig.as_ref(),
            self.cull.model_history_pso.as_ref(),
            frame_idx < self.cull.prev_model_buffers.len(),
        ) else {
            return;
        };
        let records = self.cull_count();
        if records == 0 {
            return;
        }
        // A rebuilt ring holds nothing these records were written for, so the
        // priming frame fills every slot rather than only its own: the instance
        // region is the one the draw args cannot flag, being init-written.
        let slots = match prime {
            true => 0..self.cull.prev_model_buffers.len(),
            false => frame_idx..frame_idx + 1,
        };
        let params = ModelHistoryParams {
            record_count: records as u32,
            _pad: [0; 3],
        };
        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);
        // SAFETY: the command list is in the recording state, and every resource and slice these
        // commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(pso);
            cmd.SetComputeRootSignature(root_sig);
            cmd.set_compute_root_constants(0, &params);
            cmd.SetComputeRootShaderResourceView(1, object_gva);
            for slot in slots {
                let history = &self.cull.prev_model_buffers[slot];
                cmd.SetComputeRootUnorderedAccessView(2, com::gpu_va(history));
                cmd.Dispatch((records as u32).div_ceil(MODEL_HISTORY_THREADGROUP), 1, 1);
                // A UAV barrier, not a transition: a buffer lives in COMMON and
                // is promoted implicitly at each use, so this only has to order
                // the write against the next frame's read of the same slot.
                cmd.ResourceBarrier(&[uav_barrier(history)]);
            }
        }
    }

    // GPU-driven G-buffer pre-pass raster. Reuses the main pass's per-frame
    // indirect command buffer (the camera-frustum cull already produced it, so no
    // extra cull dispatch): the static + instance prefix `[0,
    // skinned_record_base())` once per shader bucket, each under that bucket's
    // pre-pass PSO, over the static VB (bound to BOTH vertex streams, so prev_pos
    // == cur_pos and the motion is the per-object model delta plus camera), then
    // the skinned tail `[skinned_record_base(), cull_count())` under bucket 0's
    // over the current deformed VB (slot 0) + the previous-frame deformed VB
    // (slot 1), so per-vertex skin deformation produces a correct motion vector.
    // The vertex hook reads the main pass's view block and parameter table; the
    // fragment samples the bindless pool. The CPU never walks the static /
    // skinned draw lists.
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view_gva: u64,
        velocity_active: bool,
    ) {
        // Bucket 0's PSO is `None` when its pre-pass failed to build; the other
        // buckets still draw.
        let pso = self.cull.main_prepass_pso.as_ref();
        let (Some(root_sig), Some(cmd_sig), true) = (
            self.cull.prepass_root_sig.as_ref(),
            self.cull.prepass_cmd_sig.as_ref(),
            frame_idx < self.cull.prev_model_buffers.len(),
        ) else {
            return;
        };
        let indirect = &self.cull.indirect_cmd_buffers[frame_idx];
        let stride = crate::directx::cull::INDIRECT_COMMAND_STRIDE as usize;
        let prefix = self.skinned_record_base();
        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);
        let main_view_gva = com::gpu_va(&self.uniforms.view_ubo_resources[frame_idx]);

        // The history slot the PREVIOUS frame's snapshot filled; this frame's
        // own snapshot runs after the pass below has read it.
        let frames = self.cull.prev_model_buffers.len();
        let prev_model_gva =
            com::gpu_va(&self.cull.prev_model_buffers[(frame_idx + frames - 1) % frames]);
        let draw_args_gva = com::gpu_va(&self.cull.draw_args_buffer_resources[frame_idx]);

        // Static + instance prefix: bind the static VB to BOTH vertex streams
        // (prev_pos == cur_pos) + the static u32 IB, then one `ExecuteIndirect`
        // per bucket over `[0, skinned_record_base())`.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
            cmd.SetGraphicsRootSignature(root_sig);
            cmd.IASetVertexBuffers(
                0,
                Some(&[
                    self.scene.geometry.vertex_buffer_view,
                    self.scene.geometry.vertex_buffer_view,
                ]),
            );
            cmd.IASetIndexBuffer(Some(&self.scene.geometry.index_buffer_view));
            cmd.SetGraphicsRootConstantBufferView(PREPASS_VIEW_PARAM, main_view_gva);
            cmd.SetGraphicsRootConstantBufferView(PREPASS_GB_VIEW_PARAM, view_gva);
            cmd.SetGraphicsRootShaderResourceView(PREPASS_OBJECTS_PARAM, object_gva);
            cmd.SetGraphicsRootShaderResourceView(PREPASS_PREV_MODELS_PARAM, prev_model_gva);
            cmd.SetGraphicsRootShaderResourceView(PREPASS_DRAW_ARGS_PARAM, draw_args_gva);
            cmd.SetGraphicsRootShaderResourceView(
                PREPASS_MATERIAL_PARAMS_PARAM,
                self.material_params_gva(frame_idx),
            );
            cmd.set_graphics_srv_table(
                PREPASS_POOL_PARAM,
                self.cull.bindless_pool_gpu[self.current_frame],
            );
            cmd.set_graphics_sampler_table(
                PREPASS_SAMPLERS_PARAM,
                self.descriptors.linear_sampler_gpu,
            );
        }
        if let Some(pso) = pso {
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.SetPipelineState(pso);
                cmd.ExecuteIndirect(
                    cmd_sig,
                    prefix as u32,
                    indirect,
                    0,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            self.inc_draw_calls(1);
        }
        // The material-referenced shader buckets write their own regions of the
        // command buffer, each drawn under its own pre-pass PSO; a bucket whose
        // Shader is not resident is skipped, matching what the color pass will
        // draw.
        self.inc_draw_calls(self.execute_prepass_bucket_regions(
            cmd,
            cmd_sig,
            indirect,
            prefix as u32,
        ));

        // Skinned tail: bind the current deformed VB (slot 0) + the previous-frame
        // deformed VB (slot 1) + the skinned IB, then one `ExecuteIndirect`
        // over `[skinned_record_base(), cull_count())`. The records carry
        // base_vertex = 0 (global skinned indexing). When velocity is inactive the
        // previous deformed VB is the current one, so prev_pos == cur_pos and the
        // motion channel stays zero (GbView prev_vp also equals cur_vp).
        if self.state.draw.n_skinned > 0
            && let (Some(pso), Some(cur_vbv)) = (pso, self.skinned.deformed_vbvs.get(frame_idx))
        {
            // Read the previous frame's deformed pose only once the ring has been
            // primed (a prior frame's `encode_skin` filled that slot). On the
            // first frame (or after a runtime ring rebuild) the prev slot is
            // unposed, so bind the current deformed buffer as the previous one --
            // prev_pos == cur_pos gives a harmless zero skinned motion vector
            // instead of garbage. Same collapse `velocity_active == false` uses,
            // and with one frame in flight the previous slot is this one anyway.
            let frames = self.hw.frames();
            let use_prev_pose = velocity_active
                && frames >= 2
                && self
                    .skinned
                    .deformed_primed
                    .load(std::sync::atomic::Ordering::Relaxed);
            let prev_frame_idx = if use_prev_pose {
                (frame_idx + frames - 1) % frames
            } else {
                frame_idx
            };
            let prev_vbv = self
                .skinned
                .deformed_vbvs
                .get(prev_frame_idx)
                .copied()
                .unwrap_or(*cur_vbv);
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                // Skinned records are always bucket 0.
                cmd.SetPipelineState(pso);
                cmd.IASetVertexBuffers(0, Some(&[*cur_vbv, prev_vbv]));
                cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
                cmd.ExecuteIndirect(
                    cmd_sig,
                    self.state.draw.n_skinned as u32,
                    indirect,
                    (prefix * stride) as u64,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            self.inc_draw_calls(1);
            // The current deformed buffer is posed this frame, so next frame's
            // history slot (this slot) is valid -- prime the ring.
            self.skinned
                .deformed_primed
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}
