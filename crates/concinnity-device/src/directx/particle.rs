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
//!   3. Transitions visible pools to NON_PIXEL_SHADER_RESOURCE and rasterizes
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
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::particles::{ParticleEmitterRecord, ParticleSpawnState, spawn_seed};
use concinnity_core::render::reactive_mask::ReactiveWrite;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::com;
use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::descriptor_slot::DescriptorTables;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::error::map_hresult;
use crate::directx::pso::{Blend, GraphicsPso, Raster, compute_pso};
use crate::directx::reactive_mask::mask_target;
use crate::directx::root_sig::{RootSig, SamplerState, Visibility};
use crate::directx::texture::{
    HDR_FORMAT, create_uav_buffer, transition_barrier, write_texture_srv,
};

// GPU-compute particle system. `resources` (compute + render PSOs + per-frame
// uniform rings) is built lazily either at init (when
// the world declared >= 1 emitter) or on the first runtime `add_emitter`; it
// stays `None` when no emitter has ever existed. `records` and `emitter_state`
// are parallel `Vec<Option<...>>`s walked in lockstep by the per-frame dispatch,
// skipping `None` pairs; `free_slots` recycles vacated slots. `srv_base_slot` is
// the SRV-heap slot where emitter `i`'s albedo SRV lives (written by
// `add_emitter`). `last_elapsed` is the previous frame's `elapsed` (the diff is
// the frame `dt`); `frame_index` is mixed into the compute kernel's per-thread
// RNG seed. Both are interior-mutable because `record_frame` is `&self` and they
// are only touched on the render thread.
pub(in crate::directx) struct ParticleState {
    pub resources: Option<ParticleResources>,
    pub records: Vec<Option<ParticleEmitterRecord>>,
    pub emitter_state: Vec<Option<ParticleEmitterGpuState>>,
    pub free_slots: Vec<usize>,
    pub srv_base_slot: usize,
    pub last_elapsed: std::cell::Cell<f32>,
    pub frame_index: std::cell::Cell<u32>,
}

// Cap on the number of simultaneously-live particle emitters. The SRV heap
// reserves a fixed block of `MAX_EMITTERS` per-emitter albedo SRV slots at
// init, so runtime `add_emitter` past this many returns an error. Matches
// the storage shape of `MAX_DECALS`.
pub(in crate::directx) const MAX_EMITTERS: usize = 256;

// `GpuParticle` (one simulation-pool slot) and `ParticleView` (the render-pass
// view cbuffer) are GPU-free layout structs that live in `core::render`;
// re-export them so `crate::directx::particle::{GpuParticle,ParticleView}` are
// unchanged.
pub(in crate::directx) use concinnity_core::render::uniforms::GpuParticle;
pub(in crate::directx) use concinnity_core::render::uniforms::ParticleView;

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

// Per-emitter persistent GPU state: the particle pool and the CPU-side spawn
// state. Dropping releases the underlying COM resources; the D3D12 driver keeps
// them alive until any in-flight command list referencing them completes.
pub(in crate::directx) struct ParticleEmitterGpuState {
    // Particle pool: `record.max_particles` slots of `GpuParticle`, used as a
    // UAV by the compute kernel and as a structured-buffer SRV by the vertex
    // stage. Resting state: UNORDERED_ACCESS.
    pub pool: ID3D12Resource,
    // Turns `dt` and the emitter's `spawn_rate` into each dispatch's spawn
    // run. Interior-mutable because `record_frame` (which calls into
    // `prepare_particle_pass`) holds `&self`; the field is only touched on the
    // render thread.
    pub spawn_state: std::cell::RefCell<ParticleSpawnState>,
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
        let view_size = align256(std::mem::size_of::<ParticleView>() as u64);
        let mut view_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(FRAMES);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = alloc.alloc_buffer(
                view_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map particle view ubo"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        // Per-frame, per-emitter params ring. One CBV is 256-aligned, so size
        // each slot to align256(sizeof(ParticleParams)).
        let params_stride = align256(std::mem::size_of::<ParticleParams>() as u64);
        let params_total = params_stride * MAX_EMITTERS as u64;
        let mut params_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(FRAMES);
        let mut params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = alloc.alloc_buffer(
                params_total,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map particle params ubo"))?;
            params_ubo_ptrs.push(ptr as *mut u8);
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
    let slots = record.max_particles as u64;
    let pool_bytes = slots * std::mem::size_of::<GpuParticle>() as u64;

    // Default-heap UAV buffer for the pool. Created in COMMON (D3D12 always makes
    // committed buffers in COMMON regardless of the requested state); zero_default_
    // buffer leaves it in its UNORDERED_ACCESS resting state, then the encoder flips
    // it to NON_PIXEL_SHADER_RESOURCE around the render pass and back.
    let pool = create_uav_buffer(device, pool_bytes, D3D12_RESOURCE_STATE_COMMON)?;
    zero_default_buffer(alloc, &pool, pool_bytes)?;

    Ok(ParticleEmitterGpuState {
        pool,
        spawn_state: std::cell::RefCell::new(ParticleSpawnState::default()),
    })
}

// Zero-initialize a freshly-created (COMMON) default-heap buffer by uploading
// from a temporary upload-heap buffer through a one-shot command list. The target
// is transitioned COMMON → COPY_DEST for the copy and then to UNORDERED_ACCESS,
// its resting state for the per-frame compute passes.
fn zero_default_buffer(
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
    unsafe { upload.Map(0, None, Some(&mut ptr)) }
        .map_err(|e| map_hresult(e.code(), "zero_default_buffer: map upload"))?;
    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the source
    // is a separate allocation, so the ranges cannot overlap.
    unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes as usize) };
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
    // SAFETY: the command list is in the recording state, and every resource, descriptor and slice
    // these commands name is live for the call.
    unsafe {
        list.ResourceBarrier(&[to_copy_dest]);
        list.CopyBufferRegion(target, 0, &*upload, 0, bytes);
    }
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

// One live emitter's GPU addresses for this frame: its slot of the per-frame
// `ParticleParams` ring and its persistent pool. Resolved
// once in `prepare_particle_pass` so neither encode half re-borrows the emitter
// state to find them.
struct EmitterFrameData {
    params_gva: u64,
    pool_gva: u64,
}

// The per-frame particle inputs `prepare_particle_pass` derives for the two
// read-only encode halves to consume.
pub(in crate::directx) struct ParticleFrame {
    // Per-emitter addresses, parallel to `records`; `None` for a tombstone. The
    // frame's `dt`, RNG seed and spawn run are already in the
    // `ParticleParams` slot each entry points at, so neither half recomputes
    // them.
    emitters: Vec<Option<EmitterFrameData>>,
}

impl DxContext {
    // GPU descriptor handle for emitter `i`'s albedo SRV.
    pub(in crate::directx) fn emitter_albedo_srv_gpu(&self, i: usize) -> SrvSlot {
        let base = self
            .particle
            .resources
            .as_ref()
            .map(|s| s.emitter_srv_base_slot)
            .unwrap_or(0);
        SrvSlot::at(
            &self.descriptors.srv_heap,
            self.descriptors.srv_descriptor_size,
            base + i,
        )
    }

    // Mutating prelude for the particle pass, run once on the main thread
    // before the render-graph fan-out: advance the frame `dt` (against
    // `particle.last_elapsed`), the monotonic `particle.frame_index`, and each
    // emitter's spawn state, and fill this frame's slot of the `ParticleParams`
    // upload ring. Returns the
    // [`ParticleFrame`] both encode halves consume, or `None` when the pass is
    // inert (no pipeline built, or every slot tombstoned).
    //
    // Splitting it out is what lets the sim and the draw record into separate
    // per-pass command lists on workers: each frame's accumulator advance has to
    // happen exactly once, and the two halves have to see the same spawn runs.
    // `&self` because every DirectX encoder is; the per-frame mutable state is
    // in `Cell`s. Mirrors `vulkan::VkContext::prepare_particle_pass`.
    pub(in crate::directx) fn prepare_particle_pass(
        &self,
        frame_idx: usize,
        elapsed: f32,
    ) -> Option<ParticleFrame> {
        let resources = self.particle.resources.as_ref()?;
        if self.particle.records.is_empty() || self.particle.emitter_state.is_empty() {
            return None;
        }

        let dt = (elapsed - self.particle.last_elapsed.get()).max(0.0);
        self.particle.last_elapsed.set(elapsed);
        let frame_index = self.particle.frame_index.get().wrapping_add(1);
        self.particle.frame_index.set(frame_index);

        let params_base_gva = com::gpu_va(&resources.params_ubo_resources[frame_idx]);
        let mut emitters: Vec<Option<EmitterFrameData>> =
            Vec::with_capacity(self.particle.records.len());

        for (i, (rec_slot, gpu_slot)) in self
            .particle
            .records
            .iter()
            .zip(self.particle.emitter_state.iter())
            .enumerate()
        {
            let (rec, gpu) = match (rec_slot.as_ref(), gpu_slot.as_ref()) {
                (Some(r), Some(g)) => (r, g),
                _ => {
                    emitters.push(None);
                    continue;
                }
            };
            let spawns = gpu.spawn_state.borrow_mut().take_spawns(dt, rec);

            // Write this frame's ParticleParams into the per-emitter params slot.
            let params = rec.params(dt, spawns, spawn_seed(frame_index, i));
            // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer
            // that init sized for this payload, and the source is a separate live value, so the
            // ranges cannot overlap.
            unsafe {
                let dst = resources.params_ubo_ptrs[frame_idx]
                    .add((i as u64 * resources.params_stride) as usize);
                std::ptr::copy_nonoverlapping(
                    &params as *const ParticleParams as *const u8,
                    dst,
                    std::mem::size_of::<ParticleParams>(),
                );
            }

            emitters.push(Some(EmitterFrameData {
                params_gva: params_base_gva + i as u64 * resources.params_stride,
                pool_gva: com::gpu_va(&gpu.pool),
            }));
        }
        Some(ParticleFrame { emitters })
    }

    // Encode the `ParticlesSim` node: dispatch the simulation kernel over each
    // live emitter's pool.
    //
    // The compute -> vertex hazard against the draw is the graph's, derived from
    // the `particle_pool` read the draw declares in the VERTEX stage.
    pub(in crate::directx) fn encode_particles_sim(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame: &ParticleFrame,
    ) {
        let Some(resources) = self.particle.resources.as_ref() else {
            return;
        };
        let frame_data = frame.emitters.as_slice();

        // The pool is in UNORDERED_ACCESS already; the kernel reads + writes
        // through its root UAV. Each
        // emitter is independent so no UAV barrier is needed between dispatches
        // (resources are disjoint).
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetComputeRootSignature(&resources.simulate_root_sig);
            cmd.SetPipelineState(&resources.simulate_pso);
        }
        for (i, data) in frame_data.iter().enumerate() {
            let Some(data) = data else {
                continue;
            };
            let Some(rec) = self.particle.records[i].as_ref() else {
                continue;
            };
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.SetComputeRootConstantBufferView(0, data.params_gva);
                cmd.SetComputeRootUnorderedAccessView(1, data.pool_gva);
                let groups = rec.max_particles.div_ceil(64);
                cmd.Dispatch(groups, 1, 1);
            }
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
        let frame_data = frame.emitters.as_slice();

        // Visibility-cull per emitter, for the draw alone: the simulation ticked
        // every live pool so off-screen emitters stay in a realistic mid-life
        // state for when the camera turns back. Tombstoned (None) slots are
        // always invisible.
        let visible: Vec<bool> = self
            .particle
            .records
            .iter()
            .map(|slot| match slot {
                Some(r) => {
                    let (mn, mx) = r.aabb();
                    frustum.intersects_aabb(mn, mx)
                }
                None => false,
            })
            .collect();
        let any_visible = visible.iter().any(|v| *v);

        // Camera basis for camera-facing billboards: rows 0 and 1 of the view
        // matrix's 3×3 are the world-space right and up vectors (the view
        // matrix is column-major, so we read those rows out element-wise).
        let v = self.state.view.matrix;
        let cam_right = [v[0][0], v[1][0], v[2][0]];
        let cam_up = [v[0][1], v[1][1], v[2][1]];
        let view_uni = ParticleView {
            vp,
            cam_right,
            _pad0: 0.0,
            cam_up,
            _pad1: 0.0,
        };
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const ParticleView as *const u8,
                resources.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<ParticleView>(),
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
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.OMSetRenderTargets(2, Some(rtvs.as_ptr()), false, None);
                let viewport = D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: w as f32,
                    Height: h as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                };
                cmd.RSSetViewports(&[viewport]);
                let scissor = RECT {
                    left: 0,
                    top: 0,
                    right: w as i32,
                    bottom: h as i32,
                };
                cmd.RSSetScissorRects(&[scissor]);
                cmd.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
                );
                cmd.SetPipelineState(&resources.render_pso);
                cmd.SetGraphicsRootSignature(&resources.render_root_sig);
                cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
                cmd.SetGraphicsRootConstantBufferView(0, view_gva);
                // Main depth is already in a shader-resource state: the graph
                // declares this pass's depth read and emits the transition.
                cmd.set_graphics_srv_table(4, resources.depth_srv_gpu);
            }

            for (i, data) in frame_data.iter().enumerate() {
                if !visible[i] {
                    continue;
                }
                let Some(data) = data else {
                    continue;
                };
                let Some(rec) = self.particle.records[i].as_ref() else {
                    continue;
                };
                let albedo_srv_gpu = self.emitter_albedo_srv_gpu(i);
                // SAFETY: the command list is in the recording state, and every resource,
                // descriptor and slice these commands name is live for the call.
                unsafe {
                    cmd.SetGraphicsRootConstantBufferView(1, data.params_gva);
                    cmd.SetGraphicsRootShaderResourceView(2, data.pool_gva);
                    cmd.set_graphics_srv_table(3, albedo_srv_gpu);
                    cmd.DrawInstanced(4, rec.max_particles, 0, 0);
                }
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

        // Reuse a tombstoned slot if available; otherwise grow the vec.
        let id = if let Some(slot) = self.particle.free_slots.pop() {
            self.particle.records[slot] = Some(record);
            self.particle.emitter_state[slot] = Some(gpu_state);
            slot
        } else {
            if self.particle.records.len() >= MAX_EMITTERS {
                return Err(RenderError::Other(format!(
                    "add_emitter: MAX_EMITTERS ({MAX_EMITTERS}) exceeded"
                )));
            }
            self.particle.records.push(Some(record));
            self.particle.emitter_state.push(Some(gpu_state));
            self.particle.records.len() - 1
        };

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
        let rec_slot = self.particle.records.get_mut(emitter_id).ok_or_else(|| {
            RenderError::Other(format!("remove_emitter: id {emitter_id} out of range"))
        })?;
        if rec_slot.is_none() {
            return Err(RenderError::Other(format!(
                "remove_emitter: id {emitter_id} already removed"
            )));
        }
        *rec_slot = None;
        if let Some(gpu_slot) = self.particle.emitter_state.get_mut(emitter_id) {
            *gpu_slot = None;
        }
        self.particle.free_slots.push(emitter_id);
        Ok(())
    }
}
