// D3D12 rendering context. Owns all GPU resources, the Win32 window, and input
// state. Mirrors the public API of VkContext / MtlContext so GraphicsSystem can
// drive all three backends identically.

use concinnity_core::components;
use concinnity_core::gfx::render_types;
use concinnity_core::gfx::render_types::*;
use concinnity_core::input::keymap::KeyMap;
use concinnity_core::input::snapshot::InputSnapshot;
use concinnity_core::profile;
use concinnity_core::render::backend;
use concinnity_core::render::backend::FrameParams;
use concinnity_core::render::backend_init;
use concinnity_core::render::error;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::hdr_output;
use concinnity_core::render::lights;
use concinnity_core::render::pass_timing;
use concinnity_core::render::planar_reflection;
use concinnity_core::render::reflection_probe;
use concinnity_core::render::render_graph;
use concinnity_core::render::scene_flow;
use concinnity_core::render::slot_rewrites;
use concinnity_core::window::display_mode;
use std::cell::RefCell;
use std::sync::OnceLock;
use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{GetCurrentThreadId, WaitForSingleObject};

use super::allocator::{DeviceAllocator, PooledBuffer, PooledTexture};
use super::auto_exposure::AutoExposureState;
use super::cull::CullState;
use super::decal::*;
use super::draw::main::InstanceBucketLayout;
use super::draw::shadow::ShadowState;
use super::draw::spot_shadow::SpotShadowState;
use super::fog::*;
use super::hot_reload::HotReloadState;
use super::particle::ParticleState;
use super::post::bloom::BloomState;
use super::post::gbuffer::GbufferResources;
use super::post::ssao::*;
use super::post::ssr::*;
use super::post::taa::*;
use super::post::upscale::UpscaleState;
use super::resources::geometry::MeshStreamState;
use super::resources::skinning::SkinnedState;
use super::resources::streaming::ChunkStreamState;
use super::texture::*;
use crate::directx::descriptor_slot::{SamplerSlot, SrvSlot};
use crate::win32::window;
use crate::win32::window::{WindowState, frame_tick, take_input_snapshot};

// Constants
pub(super) const FRAMES: usize = 3; // triple-buffered
// Constant buffer alignment required by D3D12.
pub(super) const CB_ALIGN: u64 = 256;

// Upper bound on the number of `SkinnedMesh` draws. The SRV heap is sized once
// at init, so a fixed block of `MAX_SKINNED_OBJECTS * 2` descriptors is
// reserved for skinned (albedo, normal) pairs even when no skinned mesh is
// declared (the reservation costs only descriptor-heap slots, mirroring the
// chunk-streaming SRV reservation). `upload_skinned` rejects worlds exceeding
// this count.
pub(super) const MAX_SKINNED_OBJECTS: usize = 64;

pub(super) fn align256(n: u64) -> u64 {
    (n + CB_ALIGN - 1) & !(CB_ALIGN - 1)
}

// Build the timestamp-query heap + readback buffer for the per-frame GPU
// time chip. Returns `(None, None, null, 0)` when the queue does not
// support timestamps or any resource allocation fails; `draw_frame` then
// leaves `gpu_frame_us` at zero. The readback buffer is persistently
// mapped (READBACK-heap pointers stay valid for the resource's lifetime,
// matching the auto-exposure readback pattern).
pub(super) fn build_timestamp_resources(
    alloc: &DeviceAllocator,
) -> (
    Option<ID3D12QueryHeap>,
    Option<PooledBuffer>,
    *const u64,
    u64,
) {
    let device = alloc.device();
    // SAFETY: a property query on a live command queue; it only reads.
    let frequency = unsafe { alloc.queue().GetTimestampFrequency() }.unwrap_or(0);
    if frequency == 0 {
        return (None, None, std::ptr::null(), 0);
    }
    // Heap holds one block of [whole_frame_start, whole_frame_end, then
    // PASS_COUNT (start, end) pairs] per in-flight frame. See
    // `concinnity_core::render::pass_timing` for the slot layout.
    //
    // Timing: `execute_graph` issues an EndQuery before and after each pass's
    // encode, and the resolve at the end of the command list copies the whole block
    // into the persistently-mapped readback buffer. The CPU reads the previous
    // frame's block at the top of `draw_frame`, after the matching fence wait gates
    // the GPU writes. SsaoPrepass and SsaoKernel are bundled inside their parent
    // encoder, and the FogFroxel / Upscale / Transparent / Raymarch arms are no-ops
    // here, so those slots stay zero and drop out of the on-screen chip.
    let heap_desc = D3D12_QUERY_HEAP_DESC {
        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
        Count: (pass_timing::SLOTS_PER_FRAME * FRAMES) as u32,
        NodeMask: 0,
    };
    let mut heap: Option<ID3D12QueryHeap> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    if let Err(e) = unsafe { device.CreateQueryHeap(&heap_desc, &mut heap) } {
        tracing::warn!("timestamp query heap create failed: {e}");
        return (None, None, std::ptr::null(), 0);
    }
    let readback = match alloc.alloc_buffer(
        pass_timing::FRAME_BLOCK_BYTES * FRAMES as u64,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    ) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("timestamp readback buffer create failed: {e}");
            return (None, None, std::ptr::null(), 0);
        }
    };
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    if let Err(e) = unsafe { readback.Map(0, None, Some(&mut ptr)) } {
        tracing::warn!("timestamp readback map failed: {e}");
        return (None, None, std::ptr::null(), 0);
    }
    (heap, Some(readback), ptr as *const u64, frequency)
}

// Per-pass GPU timestamp query state. `query_heap` + `readback` are `None`
// when the command queue does not expose a non-zero timestamp frequency; the
// per-pass chip then reports 0 us. See [`pass_timing`] for the slot
// helpers and [`build_timestamp_resources`] for construction.
pub(super) struct TimestampState {
    // Timestamp query heap with `SLOTS_PER_FRAME * FRAMES` slots: one block
    // per in-flight frame, each laid out as a whole-frame start/end pair
    // followed by one (start, end) pair per `PassId`.
    pub query_heap: Option<ID3D12QueryHeap>,
    // Persistently-mapped READBACK buffer paired with `query_heap`, holding
    // `FRAMES` blocks of `SLOTS_PER_FRAME` `u64` ticks each.
    pub readback: Option<PooledBuffer>,
    pub readback_ptr: *const u64,
    // Ticks per second from `ID3D12CommandQueue::GetTimestampFrequency`; zero
    // when timestamps are unsupported.
    pub frequency: u64,
}

// Off-screen HDR scene target. The main + instanced passes render linear-light
// HDR into `color`; the composite pass tonemaps it down onto the swapchain
// backbuffer. With MSAA on (`msaa_samples > 1`), `color` is the multisampled
// target and the per-frame loop resolves it into the single-sample `resolve`;
// with MSAA off `color` is single-sample and `resolve` is `None`.
// `resolve_rtv` is `Some` only when MSAA is on (the projected-decal pass renders
// into the resolved scene target; the MSAA-off path uses `color_rtv`).
// `srv_gpu` points at whichever target the composite pass samples.
pub(super) struct HdrState {
    pub color: ID3D12Resource,
    pub color_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub resolve: Option<ID3D12Resource>,
    pub resolve_rtv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub srv_gpu: SrvSlot,
    pub msaa_samples: u32,
}

// Rectangular area lights: the per-scene `AreaLightData` table indexed by
// `GpuLight.data_index`, plus the two LTC lookup tables the shading path
// samples. All three are static for the world's lifetime. The tables are
// scene-independent (fitted at build time), so they are uploaded even with no
// area light declared -- the shader simply never samples them.
pub(super) struct AreaLightState {
    pub buffer: PooledBuffer,
    #[expect(
        dead_code,
        reason = "owns the resource the LTC table descriptors point at"
    )]
    pub ltc_matrix: GpuResource,
    #[expect(
        dead_code,
        reason = "owns the resource the LTC table descriptors point at"
    )]
    pub ltc_magnitude: GpuResource,
    // Base of the 2-descriptor LTC table (matrix, then magnitude).
    pub ltc_table_gpu: SrvSlot,
}

// The main-pass constant buffers, grouped off the flat `DxContext`. Mirrors
// Vulkan's `VkUniforms`. The view, light and shadow buffers are all per-frame
// and persistently mapped. The COM resources auto-release on drop; only the
// persistent mappings need an explicit `unmap`.
pub(super) struct DxUniforms {
    pub view_ubo_resources: Vec<PooledBuffer>,
    pub view_ubo_ptrs: Vec<*mut u8>,
    // Per-frame-in-flight `LightUniforms` CBV ring. Every pass that binds the
    // light block (main, transparent, raymarch, planar) takes the frame's own
    // GPU address, so a live light change never has to drain the queue.
    pub light_ubo_resources: Vec<PooledBuffer>,
    pub light_ubo_ptrs: Vec<*mut u8>,
    // Which slots of the light ring still need this frame's values.
    pub light_dirty: std::cell::Cell<concinnity_core::render::frame_dirty::FrameDirty>,
    // Per-scene local-light storage buffer bound as a root SRV by the main
    // pass. Static: filled once at init from `BackendInit.local_lights` and
    // never rewritten (not persistently mapped, so it stays out of `unmap`).
    pub local_light_buffer: PooledBuffer,
    // The values the light ring carries. A live Ambient-slider or
    // directional-light change mutates this and re-arms `light_dirty`;
    // `record_frame` writes the frame's own slot, so no in-flight read is raced.
    pub light_uniforms: render_types::LightUniforms,
    pub shadow_ubo_resources: Vec<PooledBuffer>,
    pub shadow_ubo_ptrs: Vec<*mut u8>,
    // Per-frame CBVs holding `probe.set` (the parallax boxes + live count) bound
    // at root param [11] by the main pass. A `FRAMES` ring so a frame writes its
    // own slot without racing a prior frame's in-flight GPU read.
    pub probe_set_cbvs: Vec<PooledBuffer>,
    pub probe_set_cbv_ptrs: Vec<*mut u8>,
    // A static count-0 ProbeSet CBV the asynchronous capture binds at [11], so a
    // probe face render samples the sky (no probe feedback) and never reads the
    // live ring while `record_frame` rewrites it.
    pub probe_set_empty_cbv: PooledBuffer,
}

impl DxUniforms {
    // Re-arm every light-ring slot after a change to `light_uniforms`.
    pub(super) fn mark_lights_dirty(&self) {
        let mut dirty = self.light_dirty.get();
        dirty.mark_all();
        self.light_dirty.set(dirty);
    }

    // Whether `frame`'s light CBV still needs this frame's values, clearing it.
    pub(super) fn take_light_dirty(&self, frame: usize) -> bool {
        let mut dirty = self.light_dirty.get();
        let pending = dirty.take(frame);
        self.light_dirty.set(dirty);
        pending
    }

    // Unmap the persistent view + light + shadow CBV mappings. Called from
    // `DxContext::drop`; the COM resources themselves auto-release.
    pub(super) fn unmap(&self) {
        for res in self
            .view_ubo_resources
            .iter()
            .chain(self.light_ubo_resources.iter())
            .chain(self.shadow_ubo_resources.iter())
        {
            // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping
            // past this call.
            unsafe { res.Unmap(0, None) };
        }
    }
}

// Per-frame command infrastructure (allocators + lists), grouped off the flat
// `DxContext`. Mirrors Vulkan's `VkCommands`. The frame is split across two
// outer cmd lists + a per-pass pool so non-composite passes can record in
// parallel:
//
//   * `command_allocators` / `command_lists`: "start" outer cmd list. Holds the
//     timestamp pre-init at the top of every frame (D3D12 debug layer flags an
//     unwritten slot in the `ResolveQueryData` range, so every pass's pair is
//     pre-initialized here). FRAMES-sized.
//
//   * `pass_allocators` / `pass_cmd_lists`: per-pass pool. Sized
//     `FRAMES * PASS_COUNT` so each pass owns its own allocator + cmd list per
//     in-flight slot. Workers reset their own allocator + cmd list before
//     recording, so multiple workers can encode in parallel without contending.
//     Indexed as `frame_idx * PASS_COUNT + (PassId as usize)`. Passes the graph
//     never activates simply leave their slot untouched; it never enters the
//     submission list.
//
//   * `end_command_allocators` / `end_command_lists`: "end" outer cmd list.
//     Holds the composite pass + the final timestamp `EndQuery` +
//     `ResolveQueryData`. Submitted after every per-pass cmd list so the resolve
//     sees every prior pass's `EndQuery` writes. FRAMES-sized.
pub(super) struct DxCommands {
    pub command_allocators: Vec<ID3D12CommandAllocator>,
    pub command_lists: Vec<ID3D12GraphicsCommandList>,
    pub pass_allocators: Vec<ID3D12CommandAllocator>,
    pub pass_cmd_lists: Vec<ID3D12GraphicsCommandList>,
    pub end_command_allocators: Vec<ID3D12CommandAllocator>,
    pub end_command_lists: Vec<ID3D12GraphicsCommandList>,
}

// CPU/GPU frame synchronization, grouped off the flat `DxContext`. Mirrors
// Vulkan's `VkFrameSync` (D3D12 uses one monotonic fence + per-slot signaled
// values instead of per-frame semaphores). The ring cursor (`current_frame`)
// and the per-frame draw-call accumulator stay flat on `DxContext`. All COM /
// plain fields auto-release on drop; only `fence_event` needs an explicit
// `CloseHandle` (done in `DxContext::drop`).
pub(super) struct DxFrameSync {
    pub fence: ID3D12Fence,
    // Per-slot fence value last signaled for that slot's submission. Compared
    // against `fence.GetCompletedValue()` to gate the slot's allocator reset.
    pub fence_values: Vec<u64>,
    // Global monotonic counter feeding every Signal. Must be unique per
    // submission across all slots, otherwise the wait-before-reuse check can
    // be satisfied by another slot's signal of the same value. `Cell` because
    // `wait_idle` advances it through `&self` (trait-imposed signature).
    pub next_fence_value: std::cell::Cell<u64>,
    pub fence_event: windows::Win32::Foundation::HANDLE,
}

// Shared static-mesh geometry buffers + their views, grouped off the flat
// `DxContext`. Mirrors Vulkan's `VkGeometry`. The streamed-mesh + chunk
// sub-allocators stay in their own `MeshStreamState` / `ChunkStreamState`.
pub(super) struct DxGeometry {
    pub vertex_buffer: PooledBuffer,
    pub index_buffer: PooledBuffer,
    pub vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    pub index_buffer_view: D3D12_INDEX_BUFFER_VIEW,
}

// Instanced-prop clusters, grouped off the flat `DxContext`. Mirrors Vulkan's
// `VkInstanced`. Every instance folds into the GPU-driven cull records, so the
// only per-instance walk left is the spot shadow pass.
pub(super) struct DxInstanced {
    pub clusters: Vec<InstancedCluster>,
    // Whether any cluster declares LOD alternates. False skips the per-frame
    // per-instance LOD patch: without alternates the base slice written into
    // every frame's draw-args buffer at init is right for the world's life.
    pub any_lod: bool,
    // Per-cluster LOD-bucket layout for the current frame. Filled at the top of
    // `record_frame` by `build_instance_upload`. `RwLock` (not `RefCell`)
    // because the parallel-encoding executor fans the passes onto rayon workers
    // that all `read()` this slice while encoding; the single writer runs on the
    // main thread before fan-out.
    pub bucket_layouts: std::sync::RwLock<Vec<Vec<InstanceBucketLayout>>>,
}

impl DxInstanced {
    pub(super) fn new(clusters: Vec<InstancedCluster>) -> Self {
        Self {
            any_lod: concinnity_core::gfx::lod::any_cluster_has_lod(&clusters),
            // One outer Vec entry per cluster; populated each frame by
            // `build_instance_upload` from `lod_buckets(cam_pos)`. The
            // inner Vec is the bucket order (LOD0 -> LODN) for that
            // cluster. Empty rows for clusters that never have visible
            // instances stay empty.
            bucket_layouts: std::sync::RwLock::new(vec![Vec::new(); clusters.len()]),
            clusters,
        }
    }
}

// The shader-visible descriptor heaps and the static sampler handles, grouped
// off the flat `DxContext`. The SRV heap's slot map is `layout`, which every
// pass addresses its descriptors through.
pub(super) struct DxDescriptors {
    // CBV/SRV/UAV descriptor heap (shader-visible). The slot map is
    // `init/heap_layout.rs`.
    pub srv_heap: ID3D12DescriptorHeap,
    pub srv_descriptor_size: usize,
    // The resolved slot of every block in `srv_heap`. The flat deduplicated
    // bindless pool at `layout.flat_pool_base_slot` (`[albedo SRVs..] ++
    // [normal SRVs..]`) repeats once per frame in flight, `flat_pool_len` slots
    // per copy: the bindless main pass and the RT hit shader address the current
    // frame's copy by a flat index, and the streaming-residency rewrite
    // re-points the one SRV per swapped pool slot in each copy as its frame
    // fence-waits. The reflection-probe cube block and the per-bake convolution
    // descriptors live here too. See [`super::probe`] and
    // [`super::probe_prefilter`].
    pub layout: super::init::heap_layout::SrvHeapLayout,
    pub flat_pool_len: usize,
    // Sampler heap (shader-visible). Slots:
    //   [0] shadow comparison (s0)   [1] linear repeat (s1)
    //   [2] cube linear-clamp + mip linear (s2)   [3] text linear-clamp
    pub sampler_heap: ID3D12DescriptorHeap,
    pub shadow_sampler_gpu: SamplerSlot,
    pub linear_sampler_gpu: SamplerSlot,
    pub text_sampler_gpu: SamplerSlot,
}

// The scene draw list plus the record counts that partition the GPU-driven
// bindless buffers.
pub(super) struct DrawState {
    pub objects: Vec<DrawObject>,
    // The last compiled frame graph, keyed by the `FrameGraphInputs` it was built
    // from. `build_frame_graph` is a pure function of those inputs (which change
    // only when a feature toggles or a target resizes), so a frame whose inputs
    // match the cached key reuses the compiled graph instead of rebuilding it.
    // `RefCell` because `record_frame` is &self; taken out before
    // `execute_graph` and put back after so a steady scene compiles the graph
    // once and reuses it thereafter.
    pub graph_cache: RefCell<Option<(render_graph::FrameGraphInputs, render_graph::CompiledGraph)>>,
    // Build-time `objects` count. Streamed chunks are appended past this, so a
    // draw index >= `draw.n_objects` identifies a chunk.
    pub n_objects: usize,
    // Instanced-cluster instances folded into the GPU-driven bindless pass as
    // per-object `GpuObjectData` records after the `draw.n_objects` static records.
    pub n_instances: usize,
    // Runtime record reserve folded into the GPU-driven bindless pass BETWEEN the
    // instances and the skinned tail: the cull buffers reserve
    // `[draw.n_objects + draw.n_instances, +draw.n_runtime)` at init. It holds
    // both kinds of object that appear after init -- streamed `VoxelWorld` chunks
    // (capacity = the worst-case resident window) and spawned clones (capacity =
    // `MAX_CLONE_DRAWS`) -- because neither can join the build-time BVH and both
    // already have their geometry in the shared VB/IB. Resident ones are packed
    // into this region each frame (`build_object_buffer` / `build_draw_args_buffer`)
    // and drawn by the static+instance prefix `ExecuteIndirect`; the unused tail
    // is disabled. Fixed at init, unlike `draw.n_skinned` (whose tail base sits
    // past this reserve).
    pub n_runtime: usize,
    // Skinned draw objects folded into the GPU-driven bindless pass: each
    // occupies a `GpuObjectData` / `GpuDrawArgs` record at buffer index
    // `skinned_record_base() + k`, past the runtime reserve, drawn (as rigid deformed geometry) by the
    // main pass's 2nd `ExecuteIndirect`. The cull / object / draw-args / indirect
    // buffers reserve these slots at init (capacity threaded through `new`); this
    // count is set in `upload_skinned` once the skinned geometry is resident, so
    // it stays 0 (and `cull_count()` excludes the reserved tail) when no skinned
    // mesh loads.
    pub n_skinned: usize,
}

impl DrawState {
    pub(super) fn new(objects: Vec<DrawObject>, n_instances: usize, n_chunk_max: usize) -> Self {
        let n_objects = objects.len();
        Self {
            n_objects,
            objects,
            graph_cache: RefCell::new(None),
            n_instances,
            // Runtime record reserve (fixed at init): the worst-case
            // resident streamed-chunk window plus the runtime-clone cap. The
            // cull buffers reserve `[n_objects + n_instances, +n_runtime)`;
            // resident chunks and spawned clones are folded in per frame and
            // the unused tail is disabled.
            n_runtime: n_chunk_max + clone_reserve(n_objects),
            // Set in `upload_skinned` once skinned geometry is resident; the
            // cull buffers reserve the tail at init via the threaded
            // `n_skinned` capacity, but `cull_count()` reads this runtime
            // count.
            n_skinned: 0,
        }
    }
}

// The frame's view state, snapped from `FrameParams` at the top of `draw_frame`.
pub(super) struct ViewState {
    pub clear_color: [f32; 4],
    // Scene-transition fade to black in [0, 1], applied in the composite pass.
    // Backend-owned rather than a `PostProcessParams` field so a settings push
    // cannot reset an in-flight fade. Stays out of `view.clear_color`: folding it
    // there would tint only the pixels no geometry covers, and would mismatch the
    // color target's baked D3D12 optimized clear value every frame.
    pub scene_fade: f32,
    // The viewport view mode: the main passes read it for the wireframe pipeline
    // variant and the unlit shade flag, the composite for its channel
    // visualization.
    pub mode: concinnity_core::gfx::view_modes::ViewMode,
    // The frame's show flags; `record_frame` masks the seeded graph inputs with
    // both these and `mode`.
    pub show: concinnity_core::gfx::view_modes::ShowFlags,
    // The frame's camera far plane, for the composite's depth-channel
    // normalization.
    pub far: f32,
    pub matrix: [[f32; 4]; 4],
    // Rows of the sky's inverse rotation, uploaded into every uniform block
    // whose pass samples the environment cubemaps.
    pub sky_rot: [[f32; 4]; 3],
}

impl ViewState {
    pub(super) fn new(clear_color: [f32; 4]) -> Self {
        Self {
            clear_color,
            scene_fade: 0.0,
            mode: Default::default(),
            show: Default::default(),
            far: 1.0,
            matrix: concinnity_core::transform::IDENTITY,
            sky_rot: concinnity_core::sky::SkyOrientation::IDENTITY_ROWS,
        }
    }
}

// Scene-captured reflection probes and the staggered bake that fills them.
// See [`super::probe`].
pub(super) struct ProbeState {
    // Placements (declared `ReflectionProbe` assets or an auto-seeded grid).
    // Indexed in order by the staggered capture pass; one cube is baked per
    // placement.
    pub placements: Vec<reflection_probe::ProbePlacement>,
    // Staggered bake cursor over `placements`: a not-yet-baked probe falls back
    // to the sky until its turn, so no single frame pays the whole capture.
    pub bake_queue: reflection_probe::ProbeBakeQueue,
    // Per-frame probe set (parallax boxes + live count) bound to the forward /
    // SSR / RT shaders. `EMPTY` until a bake installs a cube; distinct from
    // `env_map` so the skybox + diffuse irradiance keep the sky.
    pub set: concinnity_core::render::uniforms::ProbeSet,
    // The probe whose six cube faces are currently rendering on the GPU (one at a
    // time, spread one face per frame). Owns the reserved-ring-slot capture
    // resources until its faces have landed in the capture cube.
    pub rendering: Option<super::probe::RenderingBake>,
    // The prior probe whose capture is convolving into its cube on the GPU, one
    // destination mip per frame.
    pub prefiltering: Option<super::probe::PrefilteringBake>,
    // The three convolution kernels and their root signatures, built at init under
    // the same gate the bake needs. `None` disables baking.
    pub prefilter: Option<super::probe_prefilter::ProbePrefilterPipelines>,
    // One baked prefilter cube per installed probe, aligned with `set` (index `i`
    // is placement `i`). Distinct from `env_map`; sampled only by the specular
    // reflection term.
    pub maps: Vec<super::probe::ProbeCube>,
}

impl ProbeState {
    // Empty until `set_reflection_probes` supplies placements (declared or
    // auto-seeded).
    pub(super) fn new(prefilter: Option<super::probe_prefilter::ProbePrefilterPipelines>) -> Self {
        Self {
            placements: Vec::new(),
            bake_queue: reflection_probe::ProbeBakeQueue::new(0),
            set: concinnity_core::render::uniforms::ProbeSet::EMPTY,
            rendering: None,
            prefiltering: None,
            prefilter,
            maps: Vec::new(),
        }
    }
}

// Stall-free texture streaming. A streamed slot swap replaces the pool resource
// immediately but cannot rewrite the per-frame flat-pool SRV copies while their
// frames' lists are pending; `stream.pool_rewrites` carries the slot to each frame's
// copy right after its fence wait (`apply_streamed_texture_rewrites`). The
// replaced resource and the upload's transients are parked on `retires` against
// the monotonic `frame` tick and released `FRAMES + 1` ticks later: by then every
// copy has been re-pointed, every list recorded against the old resource has
// retired, and the tick's fence wait covers the upload submission itself.
pub(super) struct StreamState {
    pub pool_rewrites: slot_rewrites::SlotRewriteQueue,
    pub frame: u64,
    pub retires: Vec<super::texture::StreamedUploadRetire>,
}

impl StreamState {
    pub(super) fn new() -> Self {
        Self {
            pool_rewrites: slot_rewrites::SlotRewriteQueue::new(FRAMES),
            frame: 0,
            retires: Vec::new(),
        }
    }
}

// The swapchain, its back buffers + RTV heap, and the presentation pacing.
pub(super) struct SwapchainState {
    pub handle: IDXGISwapChain3,
    pub back_buffers: Vec<ID3D12Resource>,
    pub rtv_heap: ID3D12DescriptorHeap,
    pub rtv_descriptor_size: usize,
    // Swapchain RTV format captured at init. Stored so the composite + text PSO
    // rebuilds during shader hot-reload can target the same format the PSOs were
    // originally created against.
    pub format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    // 1 (vsync, lock to refresh) or 0 (uncapped).
    pub present_sync_interval: u32,
    // True when the swapchain was created with
    // DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING (vsync off + tearing supported); it
    // selects the tearing present flag and must be mirrored in ResizeBuffers.
    pub allow_tearing: bool,
    // Back-buffer index of the most recently presented frame. `None` until the
    // first `draw_frame` presents. The headless `screenshot` path copies this
    // buffer (the one currently on screen); `GetCurrentBackBufferIndex` after a
    // present already points at the next buffer to render into, so the captured
    // index must be recorded at present time. Mirrors
    // `VkContext::swapchain.last_present_index`.
    pub last_present_index: Option<usize>,
}

// The two resolutions the pipeline runs at. `render_*` is what the scene is
// rendered at: the G-buffer + SSAO + SSR + Hi-Z + raymarch targets are sized to
// it. When temporal upscaling is active it is `output_* * upscale_quality.scale()`,
// strictly smaller than the drawable; FSR reconstructs the drawable resolution
// into the upscaler's output texture, which bloom + composite then sample.
// `output_*` is the drawable (swapchain) resolution: the back buffers, the bloom
// mip chain, the upscaler's output texture, and the composite/text pass run at
// this size. They are equal whenever temporal upscaling is off, and the renderer
// then behaves as a single-resolution pipeline.
pub(super) struct Extents {
    pub render_width: u32,
    pub render_height: u32,
    pub output_width: u32,
    pub output_height: u32,
}

// The main depth buffer and the DSV heap that also holds the cascade, spot and
// G-buffer depth views. `dsv` indexes into the heap; the depth resource itself
// is never read back.
pub(super) struct DepthState {
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub resource: ID3D12Resource,
    pub heap: ID3D12DescriptorHeap,
}

// HUD text pass: its root signature, its PSO (`None` until the first frame that
// publishes text), and the per-frame-slot persistent upload buffers for
// transient text geometry. Each upload slot's cursor resets and its buffer
// (re)maps inside the ring's `reserve`, which the composite pass calls once the
// frame fence confirms the GPU is done with slot i.
pub(super) struct TextState {
    pub root_sig: ID3D12RootSignature,
    pub pso: Option<ID3D12PipelineState>,
    pub upload: super::upload_ring::UploadRing,
    // Held only to keep the text-atlas textures resident; the SRV handles
    // below are what the text pass actually binds.
    #[expect(
        dead_code,
        reason = "held to keep the text atlases resident; the pass binds the SRV handles"
    )]
    pub atlas_textures: Vec<GpuResource>,
    pub atlas_srv_gpus: Vec<SrvSlot>,
}

// Composite (post-process) pass: fullscreen-triangle tonemap of the HDR scene
// target onto the swapchain backbuffer.
pub(super) struct CompositeState {
    pub root_sig: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

// Per-frame counters and the D3D12 validation sink.
#[derive(Default)]
pub(super) struct Diagnostics {
    // Render statistics for the most recent frame: draw-call and object counts
    // (filled by `draw_frame`) plus VRAM bytes pulled from the adapter. Lives in
    // a `Cell` because the per-pass increments happen through the `&self`
    // `record_frame` path. Surfaced to the profiler overlay via `render_stats`.
    pub frame_stats: std::cell::Cell<profile::RenderStats>,
    // CPU-side accumulator for this frame's draw calls. The Metal + Vulkan
    // executors use the same pattern: encoders (which may run in parallel) bump
    // this atomic via `inc_draw_calls`; the main thread drains it into
    // `diagnostics.frame_stats.draw_calls` at the end of every frame. `AtomicU32` because
    // `Cell<RenderStats>` is single-threaded and would race when workers encode
    // in parallel.
    pub draw_calls_accum: std::sync::atomic::AtomicU32,
}

// The render-resolution scene targets: the HDR color target, the main depth
// buffer and its sampling SRV, the two pipeline resolutions, and the render
// graph's transient pool. The resize path rebuilds all of it.
pub(super) struct DxTargets {
    pub hdr: HdrState,
    pub depth: DepthState,
    // GPU handle of the main-depth SRV. Written at init (and rewritten on
    // resize) into a single reserved heap slot every depth-sampling decoration
    // pass binds; the lazily-built line pass needs it past init.
    pub main_depth_srv_gpu: SrvSlot,
    pub extent: Extents,
    // Backing store for the render graph's transient render targets (the
    // resources the aliasing planner manages). Owns each managed transient as a
    // placed resource on an `ID3D12Heap`; features read them back by label and
    // the executor's barrier registry resolves them the same way.
    pub transient_pool: super::transient_pool::TransientResourcePool,
}

// The world's scene assets: the IBL cubes and color-grading LUT, the area-light
// tables, and the shared static-mesh geometry. None of it depends on the
// swapchain.
pub(super) struct DxSceneAssets {
    // IBL resources. The fragment shader always samples these; when no
    // EnvironmentMap was supplied, both are 1x1 gray fallback cubes and
    // ViewUniforms::prefilter_mip_count is 0, so the shader draws the gradient
    // sky and the flat albedo ambient term instead of IBL.
    pub env_map: EnvironmentMapTextures,
    // 3D color-grading LUT sampled in the composite pass. Holds the declared
    // `ColorLut` payload baked into a Texture3D, or a 2x2x2 identity LUT when
    // the world declares none. Resolution-independent, so it is never rebuilt.
    pub color_lut: GpuResource,
    pub area_light: AreaLightState,
    pub geometry: DxGeometry,
    // Shared texture pool (kept alive). Every texture -- albedo, normal map,
    // emissive/ORM, terrain secondary -- lives here once at its handle, matching
    // Metal/Vulkan, so `DrawObject::texture_slot` and a real `normal_map_slot`
    // index directly into it. `_fallback_textures` holds only the reserved
    // pair past the last real texture -- flat-normal for a normal-less draw,
    // then white for an albedo-less one; real normal maps and albedos are
    // entries in `textures`. Held so the flat pool's fallback SRVs stay
    // resident; nothing reads the resources back.
    pub textures: Vec<PooledTexture>,
    pub _fallback_textures: Vec<PooledTexture>,
}

// The hardware ray-tracing scene: the acceleration structures and the policy
// that keeps them current as the draw set changes.
pub(super) struct DxRayTracing {
    // BLAS/TLAS + geometry table. `Some` only when RT reflections are on, the
    // GPU reports the DXR tier, and the build succeeded.
    pub accel: Option<super::raytrace::RtAccelData>,
    // How the acceleration structure is kept current as props move (the
    // launch's `--rt-dynamic` request; `Auto` by default).
    pub dynamic_mode: concinnity_core::render::rt_geom::RtDynamicMode,
    // Whether skinned meshes join the BVH (the launch's `--rt-skinned-geometry`
    // request; in by default). Clear it and the BVH covers static + instanced
    // geometry only, isolating the skinned trace path.
    pub skinned_geometry: bool,
    // Set when a runtime change altered the RT-relevant draw set (a cloned prop,
    // a streamed chunk added/removed) since the last update. Consumed once per
    // frame by `rt_dynamic_update`, which folds the change into the BLAS head
    // (`RtAccelData::refresh_topology`) -- reusing every unchanged BLAS and
    // building only the new ones -- rather than ignoring it (the `Auto` dirty
    // check only watches transforms of the prior set) or rebuilding every BLAS.
    pub topology_dirty: bool,
    // Total static vertices uploaded at init (the shared VB element count); the
    // acceleration-structure build needs it to size the hit-shader vertex SSBO.
    // Static-geometry rebuilds are not reflected (a pre-existing RT topology
    // limitation).
    pub static_vertex_count: usize,
}

// The device layer every per-world resource is built on: device, queue,
// allocator, adapter and window, plus the capabilities and display settings
// negotiated with them. The window is declared last so every COM object that
// presents to it releases first.
pub(super) struct DxHardware {
    pub device: ID3D12Device,
    pub command_queue: ID3D12CommandQueue,
    // Placement pool every persistent buffer and CPU-uploaded texture is
    // suballocated from, rather than one committed resource each. See
    // `directx/allocator.rs`.
    pub alloc: DeviceAllocator,
    // IDXGIAdapter3 for `QueryVideoMemoryInfo`, the VRAM chip's source. `None`
    // when the adapter does not expose the v3 interface; the chip then reads
    // `0 MB`.
    pub adapter: Option<IDXGIAdapter3>,
    // D3D12 validation message sink (`Some` only when validation=true).
    pub info_queue: Option<ID3D12InfoQueue>,
    // Whether the GPU reports the DXR 1.1 tier. Gates the live RT-reflections
    // toggle: an enable on a non-DXR GPU no-ops with a warning, mirroring the
    // init fallback to SSR.
    pub rt_capable: bool,
    // The resolved HDR-output mode. (The DXGI format is in `swapchain.format`.)
    pub hdr_mode: hdr_output::HdrOutputMode,
    // The swapchain config (ring depth + HDR request) this context was built
    // with, reported by `hot_swap_config` so a live editor reload reuses this
    // hardware only when the new world's config still matches.
    pub swapchain_config: backend_init::SwapchainConfig,
    // The user's chosen fullscreen display mode, held on the monitor while the
    // window is in (borderless) fullscreen and restored on exit / drop.
    // Reconciled once per frame in `window_closed`.
    pub fullscreen_display: crate::win32::display_mode::FullscreenDisplayMode,
    // Win32 window. `Option` so a live `cn editor` world reload can MOVE the
    // window (and its live cursor / menu / keymap state) into the rebuilt
    // context; a normal context always holds `Some`. Access via `win`/`win_mut`.
    pub win_state: Option<Box<WindowState>>,
}

impl DxHardware {
    // The hardware an outgoing context hands its successor on a live editor
    // `reload_world`: COM clones of the device, queue, adapter and info queue,
    // with the window and fullscreen restore state moved out. The successor
    // places into a fresh allocator; the outgoing world releases into its own.
    pub(super) fn hand_over(&mut self) -> RenderResult<Self> {
        Ok(Self {
            win_state: Some(self.win_state.take().ok_or_else(|| {
                RenderError::Other("apply_world_reload: window already taken".into())
            })?),
            fullscreen_display: std::mem::replace(
                &mut self.fullscreen_display,
                crate::win32::display_mode::FullscreenDisplayMode::new(),
            ),
            alloc: DeviceAllocator::new(&self.device, &self.command_queue, FRAMES),
            device: self.device.clone(),
            command_queue: self.command_queue.clone(),
            adapter: self.adapter.clone(),
            info_queue: self.info_queue.clone(),
            rt_capable: self.rt_capable,
            hdr_mode: self.hdr_mode,
            swapchain_config: self.swapchain_config,
        })
    }

    // Maximum extended-range multiplier on the HDR path, `None` on SDR.
    pub(super) fn max_edr(&self) -> Option<f32> {
        match self.hdr_mode {
            hdr_output::HdrOutputMode::Hdr { max_edr, .. } => Some(max_edr),
            hdr_output::HdrOutputMode::Sdr => None,
        }
    }

    // HDR encoding (scRGB-linear vs PQ) on the HDR path, `None` on SDR.
    pub(super) fn hdr_encoding(&self) -> Option<hdr_output::HdrEncoding> {
        match self.hdr_mode {
            hdr_output::HdrOutputMode::Hdr { encoding, .. } => Some(encoding),
            hdr_output::HdrOutputMode::Sdr => None,
        }
    }
}

pub(crate) struct DxContext {
    pub(super) swapchain: SwapchainState,

    // Render-resolution scene targets. See [`DxTargets`].
    pub(super) targets: DxTargets,

    // Temporal upscaling (AMD FidelityFX FSR3). See [`UpscaleState`].
    pub(super) upscale: UpscaleState,

    // Shadow map resources. See [`ShadowState`].
    pub(super) shadow: ShadowState,

    // Spot shadow map resources. See [`SpotShadowState`].
    pub(super) spot_shadow: SpotShadowState,

    // Scene assets. See [`DxSceneAssets`].
    pub(super) scene: DxSceneAssets,

    // Shader-visible descriptor heaps + samplers + scene texture pools. See
    // `DxDescriptors`.
    pub(super) descriptors: DxDescriptors,
    // Draw list + cull inputs + folded record counts. See [`DrawState`].
    pub(super) draw: DrawState,

    // Streamed-mesh byte-range sub-allocators. See [`MeshStreamState`].
    pub(super) mesh_stream: MeshStreamState,

    // Chunk-streaming byte-range sub-allocators + slot recycling. See
    // [`ChunkStreamState`].
    pub(super) chunk_stream: ChunkStreamState,

    // Skinned (skeletally animated) mesh rendering. All `None` / empty until
    // `upload_skinned` runs.
    pub(super) skinned: SkinnedState,

    // Constant buffers (view + shadow per-frame persistent-mapped, light once).
    // See `DxUniforms`.
    pub(super) uniforms: DxUniforms,

    // Root signatures + PSOs
    // GPU-driven cull + main pass. All `Some`/non-empty only when the world has
    // anything to drive. See [`CullState`].
    pub(super) cull: CullState,
    // Clustered light binning: the compute pipeline (built only when the world
    // has local lights), the per-cluster light-index buffer, and the per-frame
    // `ClusterParams` constant buffers. See [`LightCullState`].
    pub(super) light_cull: super::light_cull::LightCullState,
    pub(super) text: TextState,
    pub(super) composite: CompositeState,

    // Bloom mip chain + pipelines. The mips are shared across frame slots; the
    // command queue runs frames serially, so a frame's bloom writes never race
    // a prior frame's composite read.
    pub(super) bloom: BloomState,
    // Post-process tunables (bloom / exposure / vignette). Drives whether the
    // bloom chain runs and feeds the bloom-prefilter + composite root constants.
    pub(super) post_process: render_types::PostProcessParams,

    // Unified geometry G-buffer pre-pass. `Some` whenever any screen-space
    // consumer (SSR, SSGI, SSAO, TAA, or temporal upscaling) is enabled: one
    // jittered traversal writes view normal+depth, roughness, and motion into
    // one MRT that all those consumers read, replacing the separate SSR / SSAO
    // / velocity geometry pre-passes. See [`GbufferResources`].
    pub(super) gbuffer: Option<GbufferResources>,
    // Per-record validity of the GPU-filled model-history ring. A record whose
    // occupant changed carries `NO_HISTORY` in its draw args, which sends the
    // G-buffer pre-pass to its current model instead of a stranger's.
    // `RefCell` because the draw-args build runs off `&self`.
    pub(super) model_history: RefCell<concinnity_core::render::model_history::ModelHistory>,

    // Temporal anti-aliasing. `Some` only when `PostProcessConfig.aa_mode` is set;
    // when `None` the history resolve and the projection jitter are skipped and
    // the composite samples the HDR scene target directly.
    pub(super) taa: Option<TaaResources>,

    // The descriptor slots every shared fullscreen post pass allocates its
    // targets from. Held once for the backend rather than reserved per effect in
    // `init/heap_layout.rs`.
    pub(super) post: super::post::descriptors::PostDescriptors,

    // SSAO (GTAO). See [`SsaoState`].
    pub(super) ssao: SsaoState,

    // SSR. `Some` when `PostProcessConfig.ssr` is set, or when SSGI is on
    // (SSGI reuses the depth + normal pre-pass G-buffer). The resolve half
    // (`ssr.resolve`) is `Some` only when SSR itself is authored on; with it
    // off the TAA / bloom / composite passes sample `hdr_srv_gpu` directly as
    // the scene color. When the resolve is on it writes into
    // `ssr.resolve.output`, whose SRV becomes the scene the post stack consumes
    // (see `scene_srv_for_post`).
    pub(super) ssr: Option<SsrResources>,

    // SSGI. `Some` only when `PostProcessConfig.indirect_lighting` is `ssgi`.
    // A hemisphere-gather + depth-aware-blur composite that bleeds nearby lit
    // surfaces' color onto one another, additively on top of the IBL ambient.
    // Reuses the SSR pre-pass G-buffer (so `ssr` is also `Some` whenever this
    // is); the render-graph `PassId::Ssgi` node is gated on `ssgi.is_some()`.
    pub(super) ssgi: Option<super::post::ssgi::SsgiResources>,

    // Roughness-aware reflection composite: the SSR / RT resolve writes reflected
    // radiance + weight, then this blurs it by surface roughness (a reduced-res blur
    // pass) and composites it over the scene into its own output -- the scene the
    // post stack consumes via `scene_srv_for_post`. `Some` when SSR resolve or RT is
    // authored (both feed it).
    pub(super) reflection_composite:
        Option<super::post::reflection_composite::ReflectionCompositeResources>,

    // Hardware ray-traced reflections (DXR). `rt_reflections` (output target +
    // RtParams UBO + root sig + flat/textured PSOs) and `rt.accel` (BLAS/TLAS +
    // geometry table) are both `Some` only when the world enables
    // `ray_traced_reflections`, the GPU supports the DXR tier, and the DXC
    // compile + acceleration-structure build succeeded; otherwise the graph
    // falls back to `SsrResolve`. RT occupies the `SsrResolve` slot, reuses the
    // SSR pre-pass G-buffer (forced on), and its output becomes the scene the
    // post stack consumes via `scene_srv_for_post`. `FrameGraphInputs::
    // rt_reflections_enabled` is gated on both being `Some`.
    pub(super) rt_reflections: Option<super::post::rt_reflections::RtReflectionsResources>,
    // The ray-tracing scene. See [`DxRayTracing`].
    pub(super) rt: DxRayTracing,

    // Projected decals. See [`DecalState`].
    pub(super) decal: DecalState,

    // World-space line pass state: the resources, built on the first frame
    // that publishes lines. See [`super::line::LineState`].
    pub(super) lines: super::line::LineState,

    // Raymarched SDF volumes. `Some` when the world declares at least one
    // `SdfVolume`. Render-graph `PassId::Raymarch` is gated on
    // `Self::raymarch_enabled()` so worlds with no visible SDF skip the slot
    // entirely.
    pub(super) raymarch: Option<super::raymarch::RaymarchResources>,

    // The shared `PassId::Transparent` slot and its two producers, translucent
    // glass panes and water surfaces. `Some` only when the world declared a
    // `GlassPanel` or a `WaterSurface`; with neither the field stays `None` and
    // the pass is skipped. Render-graph inclusion is gated on
    // `Self::transparent_enabled()`. Mirrors src/metal's transparent encoder.
    pub(super) transparent: Option<super::transparent::TransparentResources>,

    // Planar reflections for the transparent pass's flat reflectors: a per-frame
    // mirror render of the scene reflected across each distinct reflector plane
    // (a water surface's rest plane, a glass pane's plane), sampled by the
    // shaders at screen UV (sharper + scene-correct vs the box-projected probe
    // cube). `Some` only when the world has reflectors assigned to a planar slot.
    // Driven inline at the head of the transparent pass
    // (`encode_planar_reflections`).
    pub(super) planar_reflection: Option<super::planar::PlanarReflectionSet>,

    // Volumetric fog. See [`FogState`].
    pub(super) fog: FogState,

    // GPU-compute particle system. See [`ParticleState`].
    pub(super) particle: ParticleState,

    // Per-frame command allocators + lists (start / per-pass / end). See
    // `DxCommands`.
    pub(super) commands: DxCommands,
    pub(super) diagnostics: Diagnostics,
    // CPU/GPU frame synchronization (one monotonic fence + per-slot values).
    // See `DxFrameSync`.
    pub(super) frame_sync: DxFrameSync,
    pub(super) current_frame: usize,

    pub(super) stream: StreamState,

    // Instanced-prop per-frame upload buffers + LOD buckets. See `DxInstanced`.
    pub(super) instanced: DxInstanced,
    pub(super) view: ViewState,
    // Lazily-built wireframe twin of the main-pass pipeline; empty until the
    // first Wireframe frame. See [`super::wireframe`].
    pub(super) wireframe: super::wireframe::DxWireframe,

    // Auto-exposure (EV adaptation) state. See [`AutoExposureState`].
    pub(super) auto_exposure: AutoExposureState,

    // Per-pass GPU timestamp queries. Read at the top of `draw_frame` after
    // the matching fence wait so the CPU sees a fully committed block.
    pub(super) timestamps: TimestampState,

    // Shader hot-reload state. See [`HotReloadState`].
    pub(super) hot_reload: HotReloadState,
    // The world default Shader's compiled programs, `None` for the engine's
    // own. Kept past init so the built-in shader reload rebuilds bucket 0 from
    // the world's pair rather than the engine's.
    pub(super) world_shader: Option<concinnity_core::components::ShaderPrograms>,

    // Fixed descriptor-heap slots for the live-toggleable Quality effects (TAA /
    // SSAO / SSR / SSGI / RT-reflection output). Minted once at init (the slots
    // are reserved unconditionally) and stashed so `apply_quality_settings` can
    // build a launched-off feature into its slot without re-deriving the heap
    // layout.
    pub(super) quality_slots: super::quality::QualitySlotHandles,

    // Scene-captured reflection probes. See [`ProbeState`].
    pub(super) probe: ProbeState,

    // Device, queue, allocator and window. Declared last so every resource
    // above releases before the device and the window go.
    pub(super) hw: DxHardware,
}

// uniforms.view_ubo_ptrs are host-mapped and only touched on the render thread.
// text_upload's RefCells + map pointers are single-threaded.
// SAFETY: `DxContext` owns COM objects and host-mapped pointers that are only ever touched from the
// thread that built it, which `debug_assert_main_thread` enforces at every mutation entry point.
// Moving the whole context to another thread is therefore safe as long as it stays main-thread-only
// there, and nothing in it is shared across threads (hence `Send` without `Sync`).
unsafe impl Send for DxContext {}

// Win32 thread id of the thread that built the context. `DxContext::new` runs
// on the main thread and records it here; `debug_assert_main_thread` checks
// every mutation entry point against it.
static MAIN_THREAD_ID: OnceLock<u32> = OnceLock::new();

// Record the calling thread as the main (render) thread. Called once from
// `DxContext::new`, which always runs on the main thread.
pub(super) fn record_main_thread() {
    // SAFETY: a read of the calling thread's own id; it borrows nothing.
    let _ = MAIN_THREAD_ID.set(unsafe { GetCurrentThreadId() });
}

// Debug-only guard that the caller is on the main thread.
//
// The `unsafe impl Send for DxContext` above is sound only because the context
// is touched from the main thread alone: the Win32 window message pump and
// D3D12 command-queue submission are both thread-affine, and the parallel
// encoder fan-out only ever shares `&self` read-only. The `RenderBackend`
// mutation entry points (reached through the boxed trait object) had nothing
// proving this, so scheduling `GraphicsSystem` off the main thread would
// silently race the window/queue instead of failing. This makes that mistake
// panic loudly in debug builds and compiles to nothing in release. `entry` is
// the offending method name, for the message. Mirrors
// `metal/context.rs::debug_assert_main_thread`.
#[inline]
#[track_caller]
pub(super) fn debug_assert_main_thread(entry: &str) {
    debug_assert!(
        MAIN_THREAD_ID
            .get()
            // SAFETY: a read of the calling thread's own id; it borrows nothing.
            .is_none_or(|&main| unsafe { GetCurrentThreadId() } == main),
        "{entry} must be called from the main thread: DxContext is main-thread-only \
         (see `unsafe impl Send for DxContext`); driving GraphicsSystem off the main \
         thread races the Win32 window + D3D12 command submission",
    );
}

impl DxContext {
    pub(crate) fn draw_frame(&mut self, params: FrameParams<'_>) -> error::RenderResult<()> {
        let FrameParams {
            elapsed,
            fov_y_radians,
            near,
            far,
            cam_pos,
            text_calls,
            lines,
            world_hidden,
            view_mode,
            show,
            sky_rot,
        } = params;
        // Snapped for the passes recorded below (the wireframe pipeline
        // variant, the unlit shade flag, the composite's channel visualization
        // + depth normalization) and for the graph-input mask in record_frame.
        self.view.mode = view_mode;
        self.view.show = show;
        self.view.far = far;
        self.view.sky_rot = sky_rot;
        self.apply_pending_rebuilds()?;

        let frame = self.current_frame;

        let gpu_wait = self.wait_frame_slot(frame)?;
        self.service_background_work(elapsed, near, far, frame);
        let timings = self.read_gpu_timings(frame);
        self.begin_frame_stats(&gpu_wait, timings);

        // Flush any D3D12 validation messages from the previous frame.
        self.flush_validation();

        // Frame command-list pipeline (parallel encoding):
        //
        //   start_cmd  : pre-init timestamps only (closed up-front)
        //   per-pass   : one cmd list per non-composite pass, recorded
        //                in parallel by rayon workers inside execute_graph
        //   end_cmd    : Composite + final timestamp + ResolveQueryData
        //                 + per-frame restore barriers (composed by
        //                 execute_graph's main-thread composite arm + the
        //                 tail of record_frame)
        //
        // ExecuteCommandLists is called once with the whole topological
        // sequence so the GPU sees them in submission order.
        let start_cmd = self.record_frame_start(frame)?;

        // Line resources: built on the first frame that publishes lines, so
        // the graph gate inside `record_frame` can see them live this same
        // frame and a world that never draws a line never compiles them.
        self.ensure_line_pipeline(!lines.is_empty());

        self.update_shadow_schedule(cam_pos, fov_y_radians, near, far);

        // 2. Open the END cmd list (Composite + final timestamp +
        //    ResolveQueryData + per-frame restore barriers). The
        //    executor's main-thread Composite arm encodes onto this
        //    cmd list; `close_end_list` appends the final timestamp +
        //    resolve after `record_frame` returns.
        // SAFETY: the fence for this frame slot was already waited on, so no submission still
        // references what is being reset.
        unsafe { self.commands.end_command_allocators[frame].Reset() }
            .map_err(|e| super::error::map_hresult(e.code(), "end allocator reset"))?;
        let end_cmd = &self.commands.end_command_lists[frame];
        // SAFETY: the fence for this frame slot was already waited on, so no submission still
        // references what is being reset.
        unsafe { end_cmd.Reset(&self.commands.end_command_allocators[frame], None) }
            .map_err(|e| super::error::map_hresult(e.code(), "end cmd reset"))?;

        // SAFETY: a property query on a live COM object; it only reads.
        let back_idx = unsafe { self.swapchain.handle.GetCurrentBackBufferIndex() } as usize;
        let back_buffer = self.swapchain.back_buffers[back_idx].clone();
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let rtv_base = unsafe { self.swapchain.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let back_buffer_rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr + back_idx * self.swapchain.rtv_descriptor_size,
        };

        // 3. record_frame fans non-composite passes onto rayon workers
        //    (each records into its own cmd list from the per-pass pool)
        //    and dispatches Composite + per-frame restore barriers onto
        //    `end_cmd`. Returns the per-pass cmd lists in topological
        //    pass order.
        let pass_cmd_lists = self.record_frame(
            crate::directx::draw::RecordFrameTargets {
                cmd: end_cmd,
                back_buffer: &back_buffer,
                back_buffer_rtv,
                frame_idx: frame,
            },
            crate::directx::draw::RecordFrameView {
                elapsed,
                fov_y_radians,
                near,
                far,
                cam_pos,
                text_calls,
                lines,
            },
            crate::directx::draw::RecordFrameResolution {
                width: self.targets.extent.render_width.max(1),
                height: self.targets.extent.render_height.max(1),
                output_width: self.targets.extent.output_width.max(1),
                output_height: self.targets.extent.output_height.max(1),
            },
            world_hidden,
        )?;

        // Step the TAA accumulation ring: what this frame wrote is next frame's
        // history. After `record_frame` rather than inside it, so the write slot
        // is stable across the whole graph, and here rather than beside the
        // jitter tick because stepping the ring needs `&mut self`.
        if let Some(taa) = self.taa.as_mut() {
            taa.pass.advance();
        }

        self.finish_frame_stats();
        self.close_end_list(frame)?;
        self.submit_and_present(start_cmd, &pass_cmd_lists, back_idx, frame, gpu_wait)
    }

    // Drain any queued D3D12 validation messages and emit them via tracing.
    pub(super) fn flush_validation(&self) {
        if let Some(ref iq) = self.hw.info_queue {
            drain_info_queue(iq);
        }
    }

    pub(crate) fn update_view(&mut self, matrix: [[f32; 4]; 4]) {
        self.view.matrix = matrix;
    }

    // Update the model matrices of the given draw objects, one
    // `(slot, matrix)` entry per changed object. Out-of-range slots have no
    // effect.
    pub(crate) fn update_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]) {
        for &(index, model) in updates {
            if let Some(obj) = self.draw.objects.get_mut(index as usize) {
                obj.model = model;
            }
        }
    }

    pub(crate) fn update_visibility(&mut self, index: usize, visible: bool) {
        if let Some(obj) = self.draw.objects.get_mut(index) {
            obj.visible = visible;
        }
    }

    // Retire a draw object for a despawned entity: clear `visible` (drops it
    // from the main / shadow / velocity passes) and `resident` (drops it from
    // the ray-tracing BLAS / geometry-table rebuild), so it leaves no ghost in
    // any pass, then return its slot to the free list so the next runtime clone
    // (or streamed chunk) recycles it. The geometry buffers stay allocated.
    // No-op if the index is out of range.
    pub(crate) fn retire_draw_object(&mut self, index: usize) {
        if let Some(obj) = self.draw.objects.get_mut(index) {
            obj.visible = false;
            obj.resident = false;
            // Slot recycling lives in the engine's draw-slot allocator; only
            // the runtime-append region recycles here (`reuses_build_slots` is
            // false because the init-time cull BVH and RT `object_indices` key
            // fixed build-time slots and cannot refit).
        }
    }

    pub(crate) fn set_fade(&mut self, fade: f32) {
        self.view.scene_fade = fade.clamp(0.0, 1.0);
    }

    // Render statistics for the most recent `draw_frame`, for the profiler
    // overlay. `gpu_frame_us` is filled at the top of each `draw_frame` from
    // the timestamp pair this slot resolved on its previous trip through the
    // ring (so a `FRAMES`-stale window, matching Metal's "frame or two
    // stale" reading).
    pub(crate) fn render_stats(&self) -> profile::RenderStats {
        self.diagnostics.frame_stats.get()
    }

    // Current GPU memory residency in bytes. `Local` is the dedicated VRAM
    // budget on a discrete GPU and the local-process system-memory budget on
    // an integrated GPU; either way, `CurrentUsage` is what the HUD's
    // "VRAM N MB" chip reports. Zero when the adapter does not expose the
    // v3 interface (pre-WDDM 2.0).
    pub(super) fn query_vram_bytes(&self) -> u64 {
        let Some(adapter) = self.hw.adapter.as_ref() else {
            return 0;
        };
        let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a query on a live COM object; the descriptor it reads and the out-
        // parameters it fills are live locals that outlive the call.
        unsafe {
            adapter.QueryVideoMemoryInfo(
                0,
                windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                &mut info,
            )
        }
        .map_or(0, |_| info.CurrentUsage)
    }

    // Shared atomic clone of the shader-reload flag, or `None` when the
    // context was built without hot-reload. Surfaced through the
    // `RenderBackend` trait so the debug server's
    // `reload-shaders` command can flip it from a non-render thread.
    pub(crate) fn shader_reload_pending(
        &self,
    ) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(std::sync::Arc::clone)
    }

    // True when the render graph should include `PassId::Raymarch`. Wraps
    // the `raymarch.any_visible()` check so callers don't have to know
    // the resource is `Option`. Drives `FrameGraphInputs::raymarch_enabled`
    // in `record_frame::seed_inputs`.
    pub(super) fn raymarch_enabled(&self) -> bool {
        self.raymarch
            .as_ref()
            .map(|r| r.any_visible())
            .unwrap_or(false)
    }

    // True when the render graph should include `PassId::RtReflections` (and
    // omit `SsrResolve`). Both the resolve resources and the acceleration
    // structure must be live; either being absent (DXR unsupported, DXC missing,
    // accel build failed, or an empty scene) falls the graph back to SSR. Drives
    // `FrameGraphInputs::rt_reflections_enabled` in `record_frame::seed_inputs`
    // and the `scene_srv_for_post` precedence.
    pub(super) fn rt_reflections_active(&self) -> bool {
        self.rt_reflections.is_some() && self.rt.accel.is_some()
    }

    // True when a reflection resolve (SSR resolve or RT reflections) runs this
    // frame. RT takes precedence at the graph level, so at most one resolve
    // runs; either feeds the same composite. Single-sources the predicate that
    // `scene_srv_for_post` and the glass pass use to pick the scene-with-
    // reflections target, and that the forward shader reads (via
    // `ViewUniforms::reflections_enabled`) to hand glossy dielectric specular to
    // that resolve instead of double-counting the forward probe reflection.
    pub(super) fn reflection_resolve_active(&self) -> bool {
        self.rt_reflections_active() || self.ssr.as_ref().and_then(|s| s.resolve.as_ref()).is_some()
    }

    // The single-sample scene target the render graph drives as `hdr_resolve`:
    // the resolve target with MSAA on, `hdr.color` itself with MSAA off. Every
    // pass that writes the spine finds it already in RENDER_TARGET, so a pass
    // that additionally needs it in some other state (the MSAA resolve, a
    // refraction snapshot) transitions from there and back within its own body.
    pub(in crate::directx) fn hdr_scene_target(&self) -> &ID3D12Resource {
        self.targets
            .hdr
            .resolve
            .as_ref()
            .unwrap_or(&self.targets.hdr.color)
    }

    // Render-target view of the spine `hdr_scene_target` returns. Every
    // decoration pass on the hdr_resolve chain binds this as its sole RTV.
    pub(in crate::directx) fn hdr_scene_rtv(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        match self.targets.hdr.resolve_rtv {
            Some(rtv) => rtv,
            None => self.targets.hdr.color_rtv,
        }
    }

    // The frame's unlit flag for ViewUniforms, from the viewport view mode.
    pub(super) fn shade_mode(&self) -> f32 {
        if self.view.mode == concinnity_core::gfx::view_modes::ViewMode::Unlit {
            1.0
        } else {
            0.0
        }
    }

    // True when the transparent pass traces a per-pixel RT reflection this frame:
    // RT is live AND every live producer's RT pipelines built (DXR + DXC).
    // Single-sources the decision so the two consumers agree: `encode_transparent`
    // selects the RT trace, and `graph_exec` skips the planar mirror re-render (RT
    // supersedes planar). They MUST gate on the same predicate -- if RT is live but
    // a producer's RT pipelines failed to build, that producer falls back to the
    // probe/planar path, so the planar resolve must still be rendered for it to
    // sample (gating the skip on `rt_reflections_active()` alone would leave it
    // sampling a stale resolve).
    pub(super) fn rt_transparent_active(&self) -> bool {
        self.rt_reflections_active()
            && self
                .transparent
                .as_ref()
                .is_some_and(|t| t.rt_pipelines_ready())
    }

    // True when the transparent pass has to render its planar mirrors this frame.
    // Water takes the mirror over its own trace wherever it holds a slot (see
    // `water.slang`), so a visible water surface keeps the re-render alive even
    // while the trace is live; a glass-only world under a live trace skips it as
    // before. Shared with the other backends through
    // `planar_reflection::planar_pass_needed`.
    pub(super) fn planar_pass_needed(&self) -> bool {
        planar_reflection::planar_pass_needed(
            self.planar_reflection.is_some(),
            self.transparent
                .as_ref()
                .is_some_and(|t| t.water_planar_slot_live()),
            self.rt_transparent_active(),
        )
    }

    // True when the render graph should include `PassId::Transparent`. Wraps
    // the `transparent.any_visible()` check plus this frame's see-through mesh
    // visibility, which lives in `draw.objects` rather than in a static record;
    // drives `FrameGraphInputs::transparent_enabled` in
    // `record_frame::seed_inputs`.
    pub(super) fn transparent_enabled(&self) -> bool {
        self.transparent.as_ref().is_some_and(|t| t.any_visible()) || self.mesh_glass_visible()
    }

    // Whether a material opted into Layer 2 see-through glass AND the device can
    // drive it (the mesh pipelines built). Independent of `rt.accel`, so it
    // answers "would the see-through path run if RT is on" -- used at the RT-BLAS
    // build, which must exclude the meshes it will reroute before the
    // acceleration structure it gates on exists. Data-driven: see-through is
    // opt-in per `Material::see_through`, so a scene with no see-through material
    // never engages Layer 2 and its transparent glass stays Layer 1 (opaque, low
    // roughness, reflective).
    pub(super) fn seethrough_meshes_enabled(&self) -> bool {
        self.transparent
            .as_ref()
            .is_some_and(|t| t.mesh_pipelines_ready())
    }

    // Whether the see-through mesh (Layer 2) path is live this frame: the
    // pipelines built AND the pass can trace (`rt_transparent_active`, which
    // needs the TLAS). When false, those meshes render opaque + reflective in the
    // main pass (Layer 1) and the producer / opaque-skip / BLAS-exclude all stay
    // inert. Mirrors `MtlContext::mesh_glass_active`.
    pub(super) fn mesh_glass_active(&self) -> bool {
        self.seethrough_meshes_enabled() && self.rt_transparent_active()
    }

    // Whether any see-through mesh would actually draw this frame. Only then does
    // the mesh producer contribute, so the graph's Transparent node is not
    // scheduled for a world whose glass is all hidden or evicted.
    fn mesh_glass_visible(&self) -> bool {
        self.mesh_glass_active()
            && self.transparent.as_ref().is_some_and(|t| {
                t.seethrough_mesh_indices().iter().any(|&i| {
                    self.draw
                        .objects
                        .get(i)
                        .is_some_and(|o| o.visible && o.resident)
                })
            })
    }

    // Bump this frame's CPU-issued draw-call counter. Called from each draw
    // site in the shadow, main, decal, composite, and text passes. Mirrors
    // `MtlContext::frame_stats.draw_calls += 1`; fullscreen post-process
    // passes (SSAO, SSR, TAA, bloom) are not counted per the `RenderStats`
    // doc comment.
    pub(super) fn inc_draw_calls(&self, n: u32) {
        // Bump the atomic accumulator so worker threads encoding in
        // parallel don't race. Drained into `diagnostics.frame_stats.draw_calls`
        // by `draw_frame` at the end of every frame.
        self.diagnostics
            .draw_calls_accum
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }
}

impl DxContext {
    // The window state. Always `Some` on a live context; `None` only on the
    // outgoing context of a `reload_world` (which has moved the window into its
    // successor and calls no window method afterward), so the unwrap never fires.
    #[inline]
    pub(super) fn win(&self) -> &WindowState {
        self.hw
            .win_state
            .as_ref()
            .expect("DxContext window state present")
    }

    #[inline]
    pub(super) fn win_mut(&mut self) -> &mut WindowState {
        self.hw
            .win_state
            .as_mut()
            .expect("DxContext window state present")
    }

    pub(crate) fn window_closed(&mut self) -> bool {
        // Message pump + cursor window-exit / fullscreen-confinement refresh +
        // the Resolution-mode reconcile, shared with the Vulkan Windows window.
        // Partial field borrow (not `win_mut`) so `fullscreen_display` can be
        // borrowed disjointly in the same call.
        frame_tick(
            self.hw
                .win_state
                .as_mut()
                .expect("DxContext window state present"),
            &mut self.hw.fullscreen_display,
        )
    }

    pub(crate) fn wait_idle(&self) {
        // Signal a new fence value and wait until the GPU reaches it.
        let val = self.frame_sync.next_fence_value.get();
        self.frame_sync.next_fence_value.set(val + 1);
        // SAFETY: the fence and the event were created from this device and are live for the call.
        if unsafe { self.hw.command_queue.Signal(&self.frame_sync.fence, val) }.is_ok()
            // SAFETY: the fence and the event were created from this device and are live for the
            // call.
            && unsafe { self.frame_sync.fence.GetCompletedValue() } < val
            // SAFETY: the fence and the event were created from this device and are live for the
            // call.
            && let Ok(()) = unsafe {
                self.frame_sync
                    .fence
                    .SetEventOnCompletion(val, self.frame_sync.fence_event)
            }
        {
            // SAFETY: the event handle was created in `DxContext::new` and lives as long as the
            // context, and the wait borrows nothing else.
            unsafe { WaitForSingleObject(self.frame_sync.fence_event, u32::MAX) };
        }
    }

    pub(crate) fn request_cursor_capture(&mut self) {
        // Don't grab the cursor immediately. A freshly spawned window may not be
        // focused yet, and clipping + hiding the system cursor before the user
        // has interacted with the window is jarring; it also diverges from the
        // Vulkan/GLFW backend, where a disabled cursor only engages once the
        // window gains focus. Instead arm the click-to-capture path: the first
        // left-click in the content area grabs the cursor, the same as
        // recapturing after Escape or a focus loss (see `wnd_proc`'s
        // `WM_LBUTTONDOWN` arm).
        self.win_mut().recapture_on_click = true;
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered in the helper, so calling
    // it every frame with the same value is cheap.
    pub(crate) fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        window::set_ui_cursor_hidden(self.win_mut(), hidden);
    }

    // Whether the real cursor has left the window so the renderer should stop
    // drawing the in-engine UI cursor (windowed / borderless). Recomputed each
    // frame by `update_ui_cursor_confinement` in `window_closed`; false while
    // captured or in fullscreen (which confines the cursor instead).
    pub(crate) fn cursor_outside_window(&self) -> bool {
        self.win().cursor_outside_window
    }

    // A togglable menu coexists with a captured camera; see
    // `RenderBackend::set_menu_mode`. The wnd_proc reads this flag to route
    // Escape to the ECS and suppress click-to-recapture.
    pub(crate) fn set_menu_mode(&mut self, on: bool) {
        self.win_mut().menu_mode = on;
    }

    // Edge-triggered capture: capture for camera control, release while a menu
    // is open. GraphicsSystem calls this each frame in menu mode. Unlike the
    // startup `request_cursor_capture` (which arms click-to-capture), closing the menu
    // recaptures immediately so the camera resumes without an extra click.
    pub(crate) fn set_camera_capture(&mut self, capture: bool) {
        if capture == self.win().cursor_captured {
            return;
        }
        if capture {
            let hwnd = self.win().hwnd;
            window::capture_cursor(hwnd, self.win_mut());
        } else {
            window::release_cursor(self.win_mut());
        }
    }

    // Turn display sync (vsync) on or off at runtime. Only the present sync
    // interval changes (1 = lock to refresh, 0 = uncapped); the present flags
    // gate ALLOW_TEARING on this interval. The swapchain's ALLOW_TEARING flag
    // is fixed at creation, so true tearing is available only when the
    // swapchain was created with vsync off; turning vsync off later still
    // presents uncapped (interval 0) but without the tearing flag if the
    // swapchain lacks it.
    pub(crate) fn set_vsync(&mut self, on: bool) {
        self.swapchain.present_sync_interval = if on { 1 } else { 0 };
    }

    // Switch window mode / resize at runtime (windowed / borderless / fullscreen
    // and content-size presets). The Win32 work lives in `window.rs`; the resize
    // path picks up the resulting WM_SIZE.
    pub(crate) fn set_window_mode(&mut self, mode: components::WindowMode) {
        window::set_window_mode(self.win_mut(), mode);
    }

    pub(crate) fn set_window_size(&mut self, width: u32, height: u32) {
        window::set_window_size(self.win_mut(), width, height);
    }

    // The display modes (resolution + refresh rate) of the monitor the window
    // sits on, feeding the Resolution settings row (the caller dedups + sorts).
    pub(crate) fn display_modes(&self) -> Vec<display_mode::DisplayMode> {
        crate::win32::display_mode::enumerate(self.win().hwnd)
    }

    // The mode the window's monitor is currently running (what the Resolution
    // row shows before the user ever picks one).
    pub(crate) fn current_display_mode(&self) -> Option<display_mode::DisplayMode> {
        crate::win32::display_mode::current(self.win().hwnd)
    }

    // Remember the display mode to hold while the window is in fullscreen.
    // Applied by the per-frame reconcile in `window_closed` (which also
    // restores the desktop mode on leaving fullscreen), so a choice made in
    // any window mode takes effect when fullscreen is (or becomes) active.
    pub(crate) fn set_display_mode(&mut self, mode: display_mode::DisplayMode) {
        self.hw.fullscreen_display.set_desired(mode);
    }

    // Replace the live post-process tunables, pushed to the bloom + composite
    // shaders each frame. The composite's display-output flags are not part of
    // the payload, so the EDR path negotiated at init survives every push.
    pub(crate) fn update_post_process(&mut self, tunables: render_types::PostProcessTunables) {
        self.post_process.set_tunables(tunables);
    }

    // Set the live ambient (IBL) light scale (the Ambient slider). It lives in
    // `LightUniforms`, which rides a per-frame-in-flight CBV ring, so this
    // mutates the CPU-side copy and re-arms every slot; `record_frame` writes
    // the frame's own slot. Edge-triggered: a no-op when the value is unchanged
    // (e.g. an init push with no persisted override).
    pub(crate) fn set_ambient_intensity(&mut self, value: f32) {
        if self.uniforms.light_uniforms.ambient_intensity == value {
            return;
        }
        self.uniforms.light_uniforms.ambient_intensity = value;
        self.uniforms.mark_lights_dirty();
    }

    // Replace the live directional lights. The cascade shadow direction and the
    // fog sun are derived from the first light on the CPU, so both are
    // re-derived here. Edge-triggered: an unchanged set touches nothing, and a
    // changed one only re-arms the light CBV ring.
    pub(crate) fn update_directional_lights(&mut self, lights: &[components::DirectionalLight]) {
        let (directional, num_directional) = lights::directional_light_data(lights);
        let uniforms = &mut self.uniforms.light_uniforms;
        if uniforms.directional == directional && uniforms.num_directional == num_directional {
            return;
        }
        uniforms.directional = directional;
        uniforms.num_directional = num_directional;
        self.shadow.light_dir = lights::sun_direction(&self.uniforms.light_uniforms);
        self.fog.sun_dir = self.shadow.light_dir;
        self.fog.sun_color = lights::sun_color(&self.uniforms.light_uniforms);
        self.uniforms.mark_lights_dirty();
    }

    // Set the live shadow cascade re-render cadence. The per-frame cascade split
    // reads `shadow.update` at the start of each draw (see draw_frame), so a
    // change takes effect on the next frame with no rebuild or allocation.
    pub(crate) fn set_shadow_update(&mut self, update: components::ShadowUpdate) {
        self.shadow.update = update;
    }

    // Set the live shadow distance (world units). The per-frame cascade-split
    // computation reads `shadow.distance` each draw (capped at the camera far
    // plane), so a change takes effect on the next frame with no allocation (it
    // sizes no GPU resource).
    pub(crate) fn set_shadow_distance(&mut self, distance: u32) {
        self.shadow.distance = distance;
    }

    // Set the live shadow cascade count (1..=4). The per-frame split + schedule
    // read `shadow.cascades` each draw; only the first `count` of the four slots
    // are rendered + sampled, so a change takes effect on the next frame with no
    // resize (the shadow-map array stays sized for the 4-cascade capacity).
    pub(crate) fn set_shadow_cascades(&mut self, count: u32) {
        self.shadow.cascades = count;
    }

    // Update the live scalar sub-tunables of the SSAO / SSR / SSGI / auto-exposure
    // passes without rebuilding anything. Each pass rebuilds its per-frame uniform
    // from these stored `*Settings` every draw (`settings.params(...)`), so
    // mutating the stored struct here is picked up on the next frame. Only a
    // feature whose resources are currently live has settings to mutate; the rest
    // are skipped (the value still persists for the next launch). SSAO / SSR /
    // auto-exposure are fully scalar, so they are replaced wholesale; SSGI keeps
    // its gather resolution / ray / step counts (those size the gather target or
    // ride `apply_quality_settings`), so only its scalar intensity / distance are
    // updated. The SSR settings live one level deeper than Metal's (inside the
    // optional `resolve` half), so a SSGI-only build with no resolve is skipped.
    pub(crate) fn update_quality_params(&mut self, q: backend::QualitySettings) {
        if let (Some(live), Some(res)) = (q.ssao, self.ssao.resources.as_mut()) {
            res.settings = live;
        }
        if let (Some(live), Some(res)) = (q.ssr, self.ssr.as_mut())
            && let Some(r) = res.resolve.as_mut()
        {
            r.settings = live;
        }
        if let (Some(live), Some(res)) = (q.ssgi, self.ssgi.as_mut()) {
            res.settings.intensity = live.intensity;
            res.settings.max_distance = live.max_distance;
        }
        if let (Some(live), Some(cur)) = (q.auto_exposure, self.auto_exposure.settings.as_mut()) {
            *cur = live;
        }
    }

    // Replace the runtime movement key map. The window message loop decodes
    // key events through it, so a settings-menu rebind takes effect immediately.
    pub(crate) fn set_keymap(&mut self, keymap: &KeyMap) {
        self.win_mut().key.set_keymap(keymap);
    }

    pub(crate) fn take_input(&mut self) -> InputSnapshot {
        take_input_snapshot(self.win_mut())
    }

    // Live window size for overlay (view-owned UI) scaling and cursor
    // hit-testing. Returns the drawable (swapchain) pixel size, which is the
    // attachment the composite + text pass writes and the space the UI shader
    // divides vertices by; WM_MOUSEMOVE reports the cursor in the same client
    // pixels, so the overlay forward / inverse transforms stay consistent.
    pub(crate) fn logical_size(&self) -> (f32, f32) {
        (
            self.targets.extent.output_width as f32,
            self.targets.extent.output_height as f32,
        )
    }

    // Device capability flags for the settings menu. RT reflects the DXR-tier
    // query made at init (`hw.rt_capable`).
    pub(crate) fn capabilities(&self) -> backend::DeviceCapabilities {
        backend::DeviceCapabilities {
            ray_tracing: self.hw.rt_capable,
            selectable_upscaler: true,
            // The cull BVH + RT tables key fixed build-time slot indices and
            // cannot refit; only the runtime-append region recycles (tracked
            // as the RT incremental topology parity item).
            reuses_build_slots: false,
            // Per-object material state is baked at build time here, so
            // `set_draw_material` has no implementation yet.
            rewrites_draws: false,
        }
    }

    // Coarse GPU performance profile for default-quality selection, read live
    // from the adapter description (vendor id + dedicated VRAM). `UNKNOWN` when
    // the adapter does not expose the v3 interface or the desc query fails.
    pub(crate) fn gpu_profile(&self) -> backend::GpuProfile {
        use concinnity_core::render::backend::{
            GpuClassInput, GpuProfile, GpuVendor, classify_tier,
        };
        let Some(adapter) = self.hw.adapter.as_ref() else {
            return GpuProfile::UNKNOWN;
        };
        // SAFETY: a property query on a live COM object; it only reads.
        let desc = match unsafe { adapter.GetDesc1() } {
            Ok(d) => d,
            Err(_) => return GpuProfile::UNKNOWN,
        };
        let vendor = match desc.VendorId {
            0x10DE => GpuVendor::Nvidia,
            0x1002 => GpuVendor::Amd,
            0x8086 => GpuVendor::Intel,
            _ => GpuVendor::Other,
        };
        let dedicated = desc.DedicatedVideoMemory as u64;
        // A discrete GPU has dedicated VRAM; an integrated part reports little or
        // none (and large shared system memory). A small floor keeps a few MB of
        // carve-out from reading as discrete.
        let discrete = dedicated >= (256u64 << 20);
        let tier = classify_tier(&GpuClassInput {
            vendor,
            memory_budget_bytes: dedicated,
            discrete,
            apple_family: 0,
        });
        GpuProfile {
            vendor,
            tier,
            memory_budget_bytes: dedicated,
            unified_memory: !discrete,
            discrete,
        }
    }
}

impl scene_flow::SceneControl for DxContext {
    fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
        self.update_visibility(draw_idx, visible);
    }
    fn set_fade(&mut self, fade: f32) {
        self.set_fade(fade);
    }
}

impl Drop for DxContext {
    fn drop(&mut self) {
        self.wait_idle();
        // Persist and release the pipeline library. `win_state` is `None` only
        // on the outgoing context of a `reload_world`, whose successor keeps
        // the device and the installed library.
        if self.hw.win_state.is_some() {
            super::pso_library::shutdown();
            // Also writes whatever shader artifacts were compiled lazily since
            // init's own checkpoint.
            crate::shader::runtime_cache::checkpoint();
        }
        // Restore cursor clip + visibility so the OS isn't left in a bad state
        // if the caller didn't release explicitly. `None` on the outgoing context
        // of a `reload_world` (the window moved to its successor), so guard it.
        if let Some(ws) = self.hw.win_state.as_mut() {
            window::release_cursor(ws);
        }
        // Unmap persistent CBV mappings (view + shadow).
        self.uniforms.unmap();
        self.light_cull.unmap();
        // SAFETY: the event was created in `DxContext::new`, is closed exactly once here, and
        // `wait_idle` at the top of `drop` retired every wait that used it.
        unsafe { CloseHandle(self.frame_sync.fence_event) }.ok();
        // The remaining COM objects (device, swapchain, heaps, etc.) are reference-
        // counted and released automatically when the struct fields are dropped.
    }
}

// Standalone version of the per-frame flush_validation so init paths can dump
// validation messages before bailing; without this, PSO/root-sig failures
// surface only as the bare `E_INVALIDARG` HRESULT from CreateGraphicsPipelineState.

pub(super) fn drain_info_queue(iq: &ID3D12InfoQueue) {
    // SAFETY: a property query on a live COM object; it only reads.
    let count = unsafe { iq.GetNumStoredMessages() };
    for i in 0..count {
        let mut len = 0usize;
        // SAFETY: `iq` is live and a null message pointer asks for the message
        // size alone, which lands in the live local `len`.
        if unsafe { iq.GetMessage(i, None, &mut len) }.is_err() {
            continue;
        }
        // A D3D12_MESSAGE is variable length: the fixed header followed by the
        // description text it points into. Back it with u64s rather than bytes so
        // the header lands on the alignment the struct needs.
        if len < std::mem::size_of::<D3D12_MESSAGE>() {
            continue;
        }
        let mut buf = vec![0u64; len.div_ceil(std::mem::size_of::<u64>())];
        let msg_ptr = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
        // SAFETY: `iq` is live, `len` is the size it just reported, and `msg_ptr`
        // addresses a live allocation of at least that many bytes.
        if unsafe { iq.GetMessage(i, Some(msg_ptr), &mut len) }.is_err() {
            continue;
        }
        // SAFETY: `GetMessage` filled `buf` with a message at least as long as the
        // header, and the u64 backing store is aligned for it. The borrow ends
        // before `buf` is dropped at the end of the iteration.
        let msg = unsafe { &*msg_ptr };
        let text = if msg.pDescription.is_null() {
            "(no description)".to_owned()
        } else {
            // SAFETY: `pDescription` is the non-null, NUL-terminated description
            // D3D12 wrote into `buf` alongside the header, so it stays live for
            // the copy this makes.
            unsafe { std::ffi::CStr::from_ptr(msg.pDescription as *const i8) }
                .to_string_lossy()
                .into_owned()
        };
        match msg.Severity {
            D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR => {
                tracing::error!(target: "d3d12", "{text}");
            }
            D3D12_MESSAGE_SEVERITY_WARNING => {
                tracing::warn!(target: "d3d12", "{text}");
            }
            _ => {
                tracing::debug!(target: "d3d12", "{text}");
            }
        }
    }
    // SAFETY: `iq` is live and every message read above has been copied out.
    unsafe { iq.ClearStoredMessages() };
}

// Wrap an init-path Result so that any D3D12 validation messages queued
// during the failing op are dumped to tracing before the error bubbles up.
pub(super) fn dump_on_err<T, E>(
    info_queue: Option<&ID3D12InfoQueue>,
    r: Result<T, E>,
) -> Result<T, E> {
    if r.is_err()
        && let Some(iq) = info_queue
    {
        drain_info_queue(iq);
    }
    r
}
