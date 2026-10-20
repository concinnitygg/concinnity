//! GPU-compute particle system for the D3D12 backend. Each `ParticleEmitter`
//! declared in the world produces one persistent `ParticleEmitterGpuState`
//! carrying a default-heap pool buffer (UAV in the compute pass, SRV in the
//! vertex pass). Each frame the renderer:
//!
//!   1. Computes the per-emitter spawn run CPU-side (a fractional accumulator
//!      drives integer particle spawns per dispatch, into the pool slots a ring
//!      cursor names).
//!   2. Dispatches the `particle_simulate` compute kernel to age + integrate +
//!      respawn each pool.
//!   3. Transitions visible pools to `NON_PIXEL_SHADER_RESOURCE` and rasterizes
//!      one alpha-blended billboard quad per live particle into `hdr_resolve`.
//!
//! The render pass alpha-blends into the resolved HDR target after the
//! volumetric-fog pass and before SSR / TAA so particles appear in screen-
//! space reflections and are temporally stabilized by TAA history. It binds no
//! depth target; the fragment tests the main depth itself, so opaque geometry
//! hides a sprite behind it. Mirrors
//! src/metal/particle.rs.

use concinnity_core::gfx::frustum::Frustum;
use concinnity_core::gfx::render_types::ParticleParams;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::particles::{ParticleEmitterRecord, ParticleFrame, ParticleSystem};
use concinnity_core::render::reactive_mask::ReactiveWrite;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::{DxContext, align256, dump_on_err};
use crate::directx::descriptor_slot::{DescriptorTables, SrvSlot};
use crate::directx::error::map_hresult;
use crate::directx::pso::{Blend, GraphicsPso, Raster, compute_pso};
use crate::directx::reactive_mask::mask_target;
use crate::directx::root_sig::{RootSig, SamplerState, Visibility};
use crate::directx::texture::{
    HDR_FORMAT, create_uav_buffer, transition_barrier, write_texture_srv,
};
use crate::directx::{builtin_shaders, com};

// GPU-compute particle system. `resources` (compute + render PSOs + per-frame
// uniform rings) is built lazily either at init (when the world declared >= 1
// emitter) or on the first runtime `add_emitter`; it stays `None` when no
// emitter has ever existed. `system` holds the emitters, each with its pool.
// `srv_base_slot` is the SRV-heap slot where emitter `i`'s albedo SRV lives
// (written by `add_emitter`).
pub(in crate::directx) struct ParticleState {
    pub resources: Option<ParticleResources>,
    pub system: ParticleSystem<ParticleEmitterGpuState>,
    pub srv_base_slot: usize,
}

// The SRV heap reserves a fixed block of `MAX_EMITTERS` per-emitter albedo
// SRV slots at init.
pub(in crate::directx) const MAX_EMITTERS: usize = concinnity_core::render::particles::MAX_EMITTERS;

// `GpuParticle` (one simulation-pool slot) and `ParticleView` (the render-pass
// view cbuffer) are GPU-free layout structs that live in `core::render`;
// re-export them so `crate::directx::particle::{GpuParticle,ParticleView}` are
// unchanged.
pub(in crate::directx) use concinnity_core::render::uniforms::{GpuParticle, ParticleView};

// Compiled particle kernels: simulate cs, vertex vs, fragment ps bytecode.
type ParticleShaders = (Vec<u8>, Vec<u8>, Vec<u8>);

// Compile the particle compute + vertex + fragment shaders; the MSAA variant
// keeps the fragment's depth SRV declaration in sync with the resource's sample
// count. Used by [`ParticleResources::new`].
pub(in crate::directx) fn compile_particle_shaders(
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<ParticleShaders> {
    let cs = builtin_shaders::PARTICLE_SIMULATE.compile(hot_reload)?;
    let vs = builtin_shaders::PARTICLE_VERT.compile(hot_reload)?;
    let ps = builtin_shaders::PARTICLE_FRAG
        .at(msaa_samples > 1)
        .compile(hot_reload)?;
    Ok((cs, vs, ps))
}

// Per-emitter persistent GPU state: the particle pool. Dropping releases the underlying COM resources; the D3D12 driver keeps
// them alive until any in-flight command list referencing them completes.
pub(in crate::directx) struct ParticleEmitterGpuState {
    // Particle pool: `record.max_particles` slots of `GpuParticle`, used as a
    // UAV by the compute kernel and as a structured-buffer SRV by the vertex
    // stage. Resting state: UNORDERED_ACCESS.
    pub pool: ID3D12Resource,
}

// Compute root signature for `particle_simulate`:
//   [0] root CBV b0 : ParticleParams (per-emitter, per-frame)
//   [1] root UAV u0 : pool (RWStructuredBuffer<Particle>)
fn create_simulate_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .uav(0, Visibility::All)
        .build(device, "particle simulate root sig")
}

// Graphics root signature for `particle_vertex` + `particle_fragment`. The two
// constant buffers are why `particle.hlsl` branches on `CN_BACKEND_DIRECTX`:
// b0 / b1 here are Metal buffer indices 1 / 2 there.
//   [0] root CBV b0   : ParticleView   (per-frame)
//   [1] root CBV b1   : ParticleParams (per-emitter)
//   [2] root SRV t0   : pool           (structured-buffer SRV)
//   [3] descriptor table SRV t1 : emitter albedo texture
//   [4] descriptor table SRV t2 : main depth (Texture2D[MS]<float>)
//   static sampler s0 : linear clamp
// The vertex shader emits the quad from SV_VertexID; no input layout.
fn create_render_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::Vertex)
        .cbv(1, Visibility::All)
        .srv(0, Visibility::Vertex)
        .srv_table(1, 1, Visibility::Pixel)
        .srv_table(2, 1, Visibility::Pixel)
        .static_sampler(SamplerState::LinearClamp, 0, Visibility::Pixel)
        .build(device, "particle render root sig")
}

fn create_simulate_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    compute_pso(device, root_sig, cs, "particle simulate")
}

// No input layout; the vertex shader reads the pool by SV_InstanceID and
// synthesizes the quad corner from SV_VertexID.
fn create_render_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    mask_target(GraphicsPso::new(root_sig, vs, ps).target(HDR_FORMAT, Blend::AlphaOver))
        .raster(Raster {
            depth_clip: false,
            ..Raster::default()
        })
        .build(device, "particle render")
}

// Pipelines + per-frame uniform rings shared across emitters. Owned by
// `DxContext` exactly once; built lazily either at init (when the world
// declares ≥1 emitter) or on the first runtime `add_emitter`.
pub(in crate::directx) struct ParticleResources {
    pub(in crate::directx) simulate_root_sig: ID3D12RootSignature,
    pub(in crate::directx) simulate_pso: ID3D12PipelineState,
    pub(in crate::directx) render_root_sig: ID3D12RootSignature,
    pub(in crate::directx) render_pso: ID3D12PipelineState,

    // Per-frame view UBO (single 96-byte block), persistently mapped.
    pub(in crate::directx) view_ubo_resources: Vec<PooledBuffer>,
    pub(in crate::directx) view_ubo_ptrs: Vec<*mut u8>,

    // Per-frame, per-emitter `ParticleParams` ring. Each slot is
    // `align256(sizeof(ParticleParams))` so the per-emitter CBV GPU address is
    // naturally 256-aligned.
    pub(in crate::directx) params_ubo_resources: Vec<PooledBuffer>,
    pub(in crate::directx) params_ubo_ptrs: Vec<*mut u8>,
    pub(in crate::directx) params_stride: u64,

    // Heap slot of the first per-emitter albedo SRV; slot `i` is the SRV for
    // emitter id `i`. Written by `add_emitter`.
    pub(in crate::directx) emitter_srv_base_slot: usize,

    // Heap slot of the main-depth SRV, bound at t2 for the fragment's depth test.
    depth_srv_gpu: SrvSlot,
}

impl ParticleResources {
    // Build the particle compute + render pipelines and the per-frame
    // uniform rings. Called from `DxContext::new` (when the world declared
    // any emitter) or from the first runtime `add_emitter`.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        emitter_srv_base_slot: usize,
        msaa_samples: u32,
        depth_srv_gpu: SrvSlot,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let (cs, vs, ps) = compile_particle_shaders(msaa_samples, hot_reload)?;

        let simulate_root_sig = dump_on_err(info_queue, create_simulate_root_signature(device))?;
        let simulate_pso = dump_on_err(
            info_queue,
            create_simulate_pso(device, &simulate_root_sig, &cs),
        )?;
        let render_root_sig = dump_on_err(info_queue, create_render_root_signature(device))?;
        let render_pso = dump_on_err(
            info_queue,
            create_render_pso(device, &render_root_sig, &vs, &ps),
        )?;

        // Per-frame view UBO.
        let view_size = align256(size_of::<ParticleView>() as u64);
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
                .map_err(|e| map_hresult(e.code(), "map particle view ubo"))?;
            view_ubo_ptrs.push(ptr.cast::<u8>());
            view_ubo_resources.push(buf);
        }

        // Per-frame, per-emitter params ring. One CBV is 256-aligned, so size
        // each slot to align256(sizeof(ParticleParams)).
        let params_stride = align256(size_of::<ParticleParams>() as u64);
        let params_total = params_stride * MAX_EMITTERS as u64;
        let mut params_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(frames);
        let mut params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                params_total,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map particle params ubo"))?;
            params_ubo_ptrs.push(ptr.cast::<u8>());
            params_ubo_resources.push(buf);
        }

        Ok(Self {
            simulate_root_sig,
            simulate_pso,
            render_root_sig,
            render_pso,
            view_ubo_resources,
            view_ubo_ptrs,
            params_ubo_resources,
            params_ubo_ptrs,
            params_stride,
            emitter_srv_base_slot,
            depth_srv_gpu,
        })
    }
}

// Allocate the per-emitter GPU state: a zero-initialized pool buffer (UAV),
// resting in UNORDERED_ACCESS.
pub(in crate::directx) fn build_emitter_gpu_state(
    alloc: &DeviceAllocator,
    record: &ParticleEmitterRecord,
) -> RenderResult<ParticleEmitterGpuState> {
    let device = alloc.device();
    let slots = u64::from(record.max_particles);
    let pool_bytes = slots * size_of::<GpuParticle>() as u64;

    // Default-heap UAV buffer for the pool. Created in COMMON (D3D12 always makes
    // committed buffers in COMMON regardless of the requested state); zero_default_
    // buffer leaves it in its UNORDERED_ACCESS resting state, then the encoder flips
    // it to NON_PIXEL_SHADER_RESOURCE around the render pass and back.
    let pool = create_uav_buffer(device, pool_bytes, D3D12_RESOURCE_STATE_COMMON)?;
    zero_default_buffer(alloc, &pool, pool_bytes)?;

    Ok(ParticleEmitterGpuState { pool })
}

// Zero-initialize a freshly-created (COMMON) default-heap buffer by uploading
// from a temporary upload-heap buffer through a one-shot command list. The target
// is transitioned COMMON → COPY_DEST for the copy and then to UNORDERED_ACCESS,
// its resting state for the per-frame compute passes.
pub(in crate::directx) fn zero_default_buffer(
    alloc: &DeviceAllocator,
    target: &ID3D12Resource,
    bytes: u64,
) -> RenderResult<()> {
    let device = alloc.device();
    let upload = alloc.alloc_buffer(
        bytes,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    // Zero the upload buffer via its persistent mapping.
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&raw mut ptr)) }
        .map_err(|e| map_hresult(e.code(), "zero_default_buffer: map upload"))?;
    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the source
    // is a separate allocation, so the ranges cannot overlap.
    unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, usize::saturating_from(bytes)) };
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe { upload.Unmap(0, None) };

    // One-shot copy command list. Pattern matches `upload_buffer` in texture.rs.
    let cmd_alloc: ID3D12CommandAllocator =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|e| map_hresult(e.code(), "zero_default_buffer: alloc"))?;
    let list: ID3D12GraphicsCommandList =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &cmd_alloc, None) }
            .map_err(|e| map_hresult(e.code(), "zero_default_buffer: list"))?;

    let to_copy_dest = transition_barrier(
        target,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: the new list is in the recording state, and `target` is live in COMMON.
    unsafe { list.ResourceBarrier(&[to_copy_dest]) };
    // SAFETY: `target` is in COPY_DEST after the barrier above, and the zeroed upload buffer
    // holds `bytes` bytes and stays alive until the fence wait below.
    unsafe { list.CopyBufferRegion(target, 0, &*upload, 0, bytes) };
    let back_to_uav = transition_barrier(
        target,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    // SAFETY: the command list is in the recording state, and every resource, descriptor and slice
    // these commands name is live for the call.
    unsafe { list.ResourceBarrier(&[back_to_uav]) };
    // SAFETY: the command list is live and in the recording state, which is what `Close` requires.
    unsafe { list.Close() }.map_err(|e| map_hresult(e.code(), "zero_default_buffer: close"))?;
    let cmd: ID3D12CommandList = windows::core::Interface::cast(&list)
        .map_err(|e| map_hresult(e.code(), "zero_default_buffer: cast"))?;
    // SAFETY: every command list in the submission is live and closed, and the slice outlives the
    // call.
    unsafe { alloc.queue().ExecuteCommandLists(&[Some(cmd)]) };

    // Wait for completion before returning so the upload buffer (about to go
    // out of scope) is not freed while still referenced. One-shot init only,
    // not on the hot path.
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|e| map_hresult(e.code(), "zero_default_buffer: fence"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    unsafe { alloc.queue().Signal(&fence, 1) }
        .map_err(|e| map_hresult(e.code(), "zero_default_buffer: signal"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    if unsafe { fence.GetCompletedValue() } < 1 {
        let event =
            // SAFETY: an auto-reset, initially unsignaled event with no name and no security
            // attributes; the call borrows nothing.
            unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
                .map_err(|e| map_hresult(e.code(), "zero_default_buffer: event"))?;
        // SAFETY: the fence and the event were created from this device and are live for the call.
        unsafe { fence.SetEventOnCompletion(1, event) }
            .map_err(|e| map_hresult(e.code(), "zero_default_buffer: set event"))?;
        // SAFETY: `event` is the handle created above and is still open.
        unsafe { windows::Win32::System::Threading::WaitForSingleObject(event, u32::MAX) };
        // SAFETY: `event` was created above, the wait has returned, and it is closed exactly once.
        unsafe { windows::Win32::Foundation::CloseHandle(event) }.ok();
    }

    Ok(())
}

impl DxContext {
    // GPU descriptor handle for emitter `i`'s albedo SRV.
    pub(in crate::directx) fn emitter_albedo_srv_gpu(&self, i: usize) -> SrvSlot {
        let base = self
            .particle
            .resources
            .as_ref()
            .map_or(0, |s| s.emitter_srv_base_slot);
        SrvSlot::at(
            &self.descriptors.srv_heap,
            self.descriptors.srv_descriptor_size,
            base + i,
        )
    }

    // Advance the particle clock and every emitter's spawn run. Run once per
    // frame on `&mut self` before `record_frame`, so the sim and draw halves
    // record on separate workers against one frame. `None` when the pass is
    // inert (no pipeline built, or no emitter ever added).
    pub(in crate::directx) fn prepare_particle_pass(
        &mut self,
        elapsed: f32,
    ) -> Option<ParticleFrame> {
        self.particle.resources.as_ref()?;
        self.particle.system.prepare(elapsed)
    }

    // Fill this frame's slot of the `ParticleParams` upload ring for every live
    // emitter, on the main thread before the fan-out.
    pub(in crate::directx) fn write_particle_params(
        &self,
        frame_idx: usize,
        frame: &ParticleFrame,
    ) {
        let Some(resources) = self.particle.resources.as_ref() else {
            return;
        };
        for emitter in self.particle.system.emitters(frame) {
            let offset = usize::saturating_from(emitter.id as u64 * resources.params_stride);
            // SAFETY: init sized the persistent mapping for `MAX_EMITTERS` slots of
            // `params_stride` bytes, and emitter ids stay below `MAX_EMITTERS`.
            let dst = unsafe { resources.params_ubo_ptrs[frame_idx].add(offset) };
            // SAFETY: `dst` starts this emitter's UPLOAD-heap slot, which holds one
            // `ParticleParams`, and the source is a separate live value, so the ranges cannot
            // overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const emitter.params).cast::<u8>(),
                    dst,
                    size_of::<ParticleParams>(),
                );
            }
        }
    }

    // GPU address of emitter `id`'s `ParticleParams` slot in frame `frame_idx`'s ring.
    fn particle_params_gva(resources: &ParticleResources, frame_idx: usize, id: usize) -> u64 {
        com::gpu_va(&resources.params_ubo_resources[frame_idx])
            + id as u64 * resources.params_stride
    }

    // Encode the `ParticlesSim` node: dispatch the simulation kernel over each
    // live emitter's pool.
    //
    // The compute -> vertex hazard against the draw is the graph's, derived from
    // the `particle_pool` read the draw declares in the VERTEX stage.
    pub(in crate::directx) fn encode_particles_sim(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        frame: ParticleFrame,
    ) {
        let Some(resources) = self.particle.resources.as_ref() else {
            return;
        };

        // The pool is in UNORDERED_ACCESS already; the kernel reads + writes
        // through its root UAV. Each
        // emitter is independent so no UAV barrier is needed between dispatches
        // (resources are disjoint).
        // SAFETY: the command list is recording, and the simulate root signature is owned by
        // `resources`.
        unsafe { cmd.SetComputeRootSignature(&resources.simulate_root_sig) };
        // SAFETY: the simulate PSO is owned by `resources` and matches the root signature above.
        unsafe { cmd.SetPipelineState(&resources.simulate_pso) };
        for emitter in self.particle.system.emitters(&frame) {
            let params_gva = Self::particle_params_gva(resources, frame_idx, emitter.id);
            // SAFETY: `params_gva` addresses this emitter's slot in this frame's params ring.
            unsafe { cmd.SetComputeRootConstantBufferView(0, params_gva) };
            let pool_gva = com::gpu_va(&emitter.gpu.pool);
            // SAFETY: the emitter's pool is live and already in UNORDERED_ACCESS.
            unsafe { cmd.SetComputeRootUnorderedAccessView(1, pool_gva) };
            let groups = emitter.record.max_particles.div_ceil(64);
            // SAFETY: the simulate pipeline and both root arguments it reads are bound above.
            unsafe { cmd.Dispatch(groups, 1, 1) };
        }
    }

    // Encode the `ParticlesDraw` node: one alpha-blended camera-facing quad per
    // live particle of every visible emitter, into the scene spine the graph has
    // already put in RENDER_TARGET for this pass's declared write. The vertex
    // shader's structured-buffer SRV reads the pool `encode_particles_sim` wrote;
    // the graph derives that `UNORDERED_ACCESS` -> `NON_PIXEL_SHADER_RESOURCE`
    // transition from the declared read, and its end-of-frame restore is what
    // returns each pool to the unordered-access state it rests in.
    pub(in crate::directx) fn encode_particles_draw(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        frame: &ParticleFrame,
        vp: [[f32; 4]; 4],
        frustum: &Frustum,
        reactive: ReactiveWrite,
    ) {
        let Some(resources) = self.particle.resources.as_ref() else {
            return;
        };

        // Visibility-cull per emitter, for the draw alone: the simulation ticked
        // every live pool so off-screen emitters stay in a realistic mid-life
        // state for when the camera turns back. Tombstoned (None) slots are
        // always invisible.
        let mut visible = self.particle.system.visible(frame, frustum).peekable();
        let any_visible = visible.peek().is_some();

        // Camera basis for camera-facing billboards: rows 0 and 1 of the view
        // matrix's 3×3 are the world-space right and up vectors (the view
        // matrix is column-major, so we read those rows out element-wise).
        let v = self.state.view.matrix;
        let cam_right = [v[0][0], v[1][0], v[2][0]];
        let cam_up = [v[0][1], v[1][1], v[2][1]];
        let view_uni = ParticleView {
            vp,
            cam_right,
            pre_exposure: self.pre_exposure.value(),
            cam_up,
            _pad1: 0.0,
        };
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const view_uni).cast::<u8>(),
                resources.view_ubo_ptrs[frame_idx],
                size_of::<ParticleView>(),
            );
        }
        let view_gva = com::gpu_va(&resources.view_ubo_resources[frame_idx]);

        if any_visible {
            let rtvs = [
                self.hdr_scene_rtv(),
                self.targets.reactive_mask.rtv(reactive),
            ];

            let w = self.targets.extent.render_width;
            let h = self.targets.extent.render_height;
            let viewport = D3D12_VIEWPORT {
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
            // SAFETY: the command list is recording, and `rtvs` is a local array of two live
            // RTVs that outlives the call.
            unsafe { cmd.OMSetRenderTargets(2, Some(rtvs.as_ptr()), false, None) };
            // SAFETY: the viewport slice is a local that outlives the call.
            unsafe { cmd.RSSetViewports(&[viewport]) };
            // SAFETY: the scissor slice is a local that outlives the call.
            unsafe { cmd.RSSetScissorRects(&[scissor]) };
            // SAFETY: setting the topology only needs a recording command list.
            unsafe {
                cmd.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
                );
            }
            // SAFETY: the render PSO is owned by `resources` and outlives the recording.
            unsafe { cmd.SetPipelineState(&resources.render_pso) };
            // SAFETY: the render root signature is owned by `resources` and matches the PSO above.
            unsafe { cmd.SetGraphicsRootSignature(&resources.render_root_sig) };
            // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
            unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
            // SAFETY: `view_gva` addresses this frame's view constants written above.
            unsafe { cmd.SetGraphicsRootConstantBufferView(0, view_gva) };
            // Main depth is already in a shader-resource state: the graph
            // declares this pass's depth read and emits the transition.
            // SAFETY: the depth SRV is a live slot in the heap set above.
            unsafe { cmd.set_graphics_srv_table(4, resources.depth_srv_gpu) };

            for emitter in visible {
                let params_gva = Self::particle_params_gva(resources, frame_idx, emitter.id);
                let albedo_srv_gpu = self.emitter_albedo_srv_gpu(emitter.id);
                // SAFETY: `params_gva` addresses this emitter's slot in this frame's params ring.
                unsafe { cmd.SetGraphicsRootConstantBufferView(1, params_gva) };
                let pool_gva = com::gpu_va(&emitter.gpu.pool);
                // SAFETY: the emitter's pool is live, and the graph moved it to a vertex-readable
                // state for this pass.
                unsafe { cmd.SetGraphicsRootShaderResourceView(2, pool_gva) };
                // SAFETY: the emitter's albedo SRV is a live slot in the bound heap.
                unsafe { cmd.set_graphics_srv_table(3, albedo_srv_gpu) };
                // SAFETY: the pipeline and every root argument the draw reads are bound above.
                unsafe { cmd.DrawInstanced(4, emitter.record.max_particles, 0, 0) };
                self.inc_draw_calls(1);
            }
        }
    }
}

// Runtime mutation (RenderBackend::add_emitter / remove_emitter)

// cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots,
// live in the concinnity binary. See the note on the analogous block in
// [directx/decal.rs].
impl DxContext {
    // Append a runtime emitter. Builds the particle pipelines + per-frame
    // uniform rings on first use (matching the init-time path) so a world
    // that never declared an emitter pays zero pipeline cost until the
    // first add. Reuses tombstoned slots from a prior `remove_emitter`
    // before growing the vec.
    pub(crate) fn add_emitter(&mut self, record: ParticleEmitterRecord) -> RenderResult<usize> {
        if self.particle.resources.is_none() {
            let resources = ParticleResources::new(
                &self.hw.alloc,
                self.particle.srv_base_slot,
                self.targets.hdr.msaa_samples,
                self.targets.main_depth_srv_gpu,
                self.hw.info_queue.as_ref(),
                self.hot_reload.enabled,
            )?;
            self.particle.resources = Some(resources);
        }
        let base_slot = self
            .particle
            .resources
            .as_ref()
            .map(|r| r.emitter_srv_base_slot)
            .ok_or_else(|| {
                RenderError::Other("add_emitter: particle pipeline unavailable".to_string())
            })?;

        let gpu_state = build_emitter_gpu_state(&self.hw.alloc, &record)?;
        let last_tex = self.scene.textures.len().saturating_sub(1);
        let tex_idx = record.texture_slot.min(last_tex);

        let id = self
            .particle
            .system
            .insert(record, gpu_state)
            .map_err(|_| {
                RenderError::Other(format!(
                    "add_emitter: MAX_EMITTERS ({MAX_EMITTERS}) exceeded"
                ))
            })?;

        let srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            // SAFETY: a property query on a live descriptor heap; it only reads.
            ptr: unsafe {
                self.descriptors
                    .srv_heap
                    .GetCPUDescriptorHandleForHeapStart()
            }
            .ptr + (base_slot + id) * self.descriptors.srv_descriptor_size,
        };
        write_texture_srv(&self.hw.device, &self.scene.textures[tex_idx], srv_cpu);
        Ok(id)
    }

    // Tombstone a runtime emitter slot. The id becomes invalid; the next
    // `add_emitter` may reuse it. The pool resource is dropped; the D3D12
    // driver keeps it alive until any in-flight command list that referenced it
    // completes, so this is safe to call mid-frame between encode passes.
    pub(crate) fn remove_emitter(&mut self, emitter_id: usize) -> RenderResult<()> {
        self.particle
            .system
            .remove(emitter_id)
            .map_err(|e| RenderError::Other(format!("remove_emitter: id {emitter_id} {e}")))?;
        Ok(())
    }
}
