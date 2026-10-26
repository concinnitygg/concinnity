// D3D12 rendering context. Owns all GPU resources, the Win32 window, and input
// state. Mirrors the public API of VkContext / MtlContext so GraphicsSystem can
// drive all three backends identically.

use crate::gpu_wait::GpuWait;
use concinnity_core::math::{SaturatingFrom, narrow_f32};
use concinnity_engine::gfx::render_types::{
    InstancedCluster, LightUniforms, PostProcessParams, PostProcessTunables,
};
use concinnity_engine::input::keymap::KeyMap;
use concinnity_engine::input::snapshot::InputSnapshot;
use concinnity_engine::render::backend::FrameParams;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::buffers::retire_pool::RetirePool;
use concinnity_engine::render::frame::plan::{FrameDevice, FramePlan, camera_inputs, drive_frame};
use concinnity_engine::render::post::reflection_path::ReflectionPath;
use concinnity_engine::render::probe::book::ProbeBook;
use concinnity_engine::render::scene_state::SceneState;
use concinnity_engine::render::shader::reload::ShaderReload;
use concinnity_engine::render::{
    backend, backend::init, backend::pass_timing, light::lights, output::hdr, slots::rewrites,
};
use concinnity_engine::window::display_mode;
use concinnity_engine::{components, profile};
use std::cell::{Cell, RefCell};
use std::sync::OnceLock;
use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{GetCurrentThreadId, WaitForSingleObject};

use super::allocator::{DeviceAllocator, PooledBuffer, PooledTexture};
use super::auto_exposure::AutoExposureState;
use super::cull::CullState;
use super::decal::DecalState;
use super::draw::shadow::ShadowState;
use super::draw::spot_shadow::SpotShadowState;
use super::fog::FogState;
use super::geometry_upload::{GeometryDest, GeometryUploads};
use super::particle::ParticleState;
use super::post::bloom::BloomResources;
use super::post::gbuffer::GbufferResources;
use super::post::ssao::SsaoState;
use super::post::ssr::SsrResources;
use super::post::taa::TaaResources;
use super::post::upscale::UpscaleState;
use super::resources::skinning::SkinnedState;
use super::texture::{EnvironmentMapTextures, GpuResource, StreamedUploadRetire};
use crate::directx::descriptor_slot::{SamplerSlot, SrvSlot};
use crate::win32::window;
use crate::win32::window::{WindowState, frame_tick, take_input_snapshot};

// Constants
// Swapchain back buffers. Indexed by `GetCurrentBackBufferIndex`, independent of
// the frames-in-flight ring depth (`DxHardware::frames`).
pub(super) const BACK_BUFFERS: usize = 3;
// Constant buffer alignment required by D3D12.
pub(super) const CB_ALIGN: u64 = 256;

pub(super) const fn align256(n: u64) -> u64 {
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
    // `concinnity_engine::render::backend::pass_timing` for the slot layout.
    //
    // Timing: `execute_graph` issues an EndQuery before and after each pass's
    // encode, and the resolve at the end of the command list copies the whole block
    // into the persistently-mapped readback buffer. The CPU reads the previous
    // frame's block at the top of `draw_frame`, after the matching fence wait gates
    // the GPU writes. SsaoDepth and SsaoKernel are bundled inside their parent
    // encoder, and the FogFroxel / Upscale / Transparent / Raymarch arms are no-ops
    // here, so those slots stay zero and drop out of the on-screen chip.
    let heap_desc = D3D12_QUERY_HEAP_DESC {
        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
        Count: u32::saturating_from(pass_timing::SLOTS_PER_FRAME * alloc.frames_in_flight()),
        NodeMask: 0,
    };
    let mut heap: Option<ID3D12QueryHeap> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    if let Err(e) = unsafe { device.CreateQueryHeap(&raw const heap_desc, &raw mut heap) } {
        tracing::warn!("timestamp query heap create failed: {e}");
        return (None, None, std::ptr::null(), 0);
    }
    let readback = match alloc.alloc_buffer(
        pass_timing::FRAME_BLOCK_BYTES * alloc.frames_in_flight() as u64,
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
    if let Err(e) = unsafe { readback.Map(0, None, Some(&raw mut ptr)) } {
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
    // Timestamp query heap with one block of `SLOTS_PER_FRAME` slots per
    // in-flight frame, each laid out as a whole-frame start/end pair followed by
    // one (start, end) pair per `PassId`.
    pub query_heap: Option<ID3D12QueryHeap>,
    // Persistently-mapped READBACK buffer paired with `query_heap`, holding
    // one block of `SLOTS_PER_FRAME` `u64` ticks per frame in flight.
    pub readback: Option<PooledBuffer>,
    pub readback_ptr: *const u64,
    // Ticks per second from `ID3D12CommandQueue::GetTimestampFrequency`; zero
    // when timestamps are unsupported.
    pub frequency: u64,
    // When the input each frame slot's last frame showed was sampled, on the
    // host clock, read back beside that frame's timestamps.
    pub input_sampled_us: Vec<Option<u64>>,
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
    // Per-scene local-light storage buffer bound as a root SRV by the main
    // pass. Static: filled once at init from `BackendInit.local_lights` and
    // never rewritten (not persistently mapped, so it stays out of `unmap`).
    pub local_light_buffer: PooledBuffer,
    // The values the light ring carries, which slots of it are stale, and the
    // local lights' reach. `record_frame` writes the frame's own slot, so no
    // in-flight read is raced.
    pub lights: concinnity_engine::render::light::state::LightState,
    pub shadow_ubo_resources: Vec<PooledBuffer>,
    pub shadow_ubo_ptrs: Vec<*mut u8>,
    // Per-frame CBVs holding the live probe count, bound at root param [11] by
    // the main pass. A frames-in-flight ring so a frame writes its
    // own slot without racing a prior frame's in-flight GPU read.
    pub probe_set_cbvs: Vec<PooledBuffer>,
    pub probe_set_cbv_ptrs: Vec<*mut u8>,
    // A static count-0 ProbeSet CBV the asynchronous capture binds at [11], so a
    // probe face render samples the sky (no probe feedback) and never reads the
    // live ring while `record_frame` rewrites it.
    pub probe_set_empty_cbv: PooledBuffer,
}

impl DxUniforms {
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
//     pre-initialized here). One per frame in flight.
//
//   * `pass_allocators` / `pass_cmd_lists`: per-pass pool. Sized
//     `frames * PASS_COUNT` so each pass owns its own allocator + cmd list per
//     in-flight slot. Workers reset their own allocator + cmd list before
//     recording, so multiple workers can encode in parallel without contending.
//     Indexed as `frame_idx * PASS_COUNT + (PassId as usize)`. Passes the graph
//     never activates simply leave their slot untouched; it never enters the
//     submission list.
//
//   * `end_command_allocators` / `end_command_lists`: "end" outer cmd list.
//     Holds the composite pass + the final timestamp `EndQuery` +
//     `ResolveQueryData`. Submitted after every per-pass cmd list so the resolve
//     sees every prior pass's `EndQuery` writes. One per frame in flight.
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
    pub next_fence_value: Cell<u64>,
    pub fence_event: windows::Win32::Foundation::HANDLE,
}

// Shared static-mesh geometry buffers + their views, grouped off the flat
// `DxContext`. Mirrors Vulkan's `VkGeometry`.
pub(super) struct DxGeometry {
    pub vertex_buffer: PooledBuffer,
    pub index_buffer: PooledBuffer,
    pub vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    pub index_buffer_view: D3D12_INDEX_BUFFER_VIEW,
}

// Instanced-prop clusters, grouped off the flat `DxContext`. Mirrors Vulkan's
// `VkInstanced`. Every instance folds into the GPU-driven cull records, so no
// pass walks the instances on the CPU.
pub(super) struct DxInstanced {
    pub clusters: Vec<InstancedCluster>,
    // Whether any cluster declares LOD alternates. False skips the per-frame
    // per-instance LOD patch: without alternates the base slice written into
    // every frame's draw-args buffer at init is right for the world's life.
    pub any_lod: bool,
}

impl DxInstanced {
    pub(super) fn new(clusters: Vec<InstancedCluster>) -> Self {
        Self {
            any_lod: concinnity_engine::gfx::lod::any_cluster_has_lod(&clusters),
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

// Scene-captured reflection probes and the staggered bake that fills them.
// See [`super::probe`].
pub(super) struct ProbeState {
    // Placements (declared `ReflectionProbe` assets or an auto-seeded grid), the
    // record of every installed probe (the live count the forward / SSR / RT /
    // transparent shaders read) and the staggered bake's queue: a not-yet-baked
    // probe falls back to the sky until its turn.
    pub book: ProbeBook,
    // The cube array the bake writes a cube of per placement, and the per-frame
    // record buffers. Distinct from `env_map` so the background + diffuse
    // irradiance keep the sky.
    pub gpu: super::probe_set::ProbeSetGpu,
    // At most one probe whose six cube faces are rendering on the GPU (one face
    // per frame, into the reserved-ring-slot capture resources) and one prior
    // probe whose capture is convolving into its cube, one destination mip per
    // frame.
    pub bake: super::probe::DxProbeBake,
    // The three convolution kernels and their root signatures, built at init under
    // the same gate the bake needs. `None` disables baking.
    pub prefilter: Option<super::probe_prefilter::ProbePrefilterPipelines>,
}

impl ProbeState {
    // Empty until `set_reflection_probes` supplies placements (declared or
    // auto-seeded).
    pub(super) fn new(
        prefilter: Option<super::probe_prefilter::ProbePrefilterPipelines>,
        gpu: super::probe_set::ProbeSetGpu,
    ) -> Self {
        Self {
            book: ProbeBook::new(),
            gpu,
            bake: super::probe::DxProbeBake::default(),
            prefilter,
        }
    }
}

// Stall-free texture streaming. A streamed slot swap replaces the pool resource
// immediately but cannot rewrite the per-frame flat-pool SRV copies while their
// frames' lists are pending; `stream.pool_rewrites` carries the slot to each frame's
// copy right after its fence wait (`apply_streamed_texture_rewrites`). The
// replaced resource and the upload's transients are parked on `retires` against
// the monotonic `frame` tick and released `retire_depth` ticks later: by then
// every copy has been re-pointed, every list recorded against the old resource
// has retired, and the tick's fence wait covers the upload submission itself.
pub(super) struct StreamState {
    pub pool_rewrites: rewrites::SlotRewriteQueue,
    pub frame: u64,
    pub retires: RetirePool<StreamedUploadRetire>,
    // `frames + 1`: a swap lands between frames, after the previous frame's
    // submit, so the first frame fence that covers the upload submission is the
    // one signaled by the NEXT draw, waited `frames` ticks after that draw's tick.
    pub retire_depth: u64,
}

impl StreamState {
    pub(super) fn new(frames: usize) -> Self {
        Self {
            pool_rewrites: rewrites::SlotRewriteQueue::new(frames),
            frame: 0,
            retires: RetirePool::new(),
            retire_depth: frames as u64 + 1,
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
    pub format: Common::DXGI_FORMAT,
    // 1 (vsync, lock to refresh) or 0 (uncapped).
    pub present_sync_interval: u32,
    // True when the swapchain was created with
    // DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING (vsync off + tearing supported); it
    // selects the tearing present flag and must be mirrored in ResizeBuffers.
    pub allow_tearing: bool,
    // Signaled while DXGI may queue another present; the frame start waits on
    // it. Owned here and closed in `DxContext::drop`.
    pub frame_latency: windows::Win32::Foundation::HANDLE,
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
    pub frame_stats: Cell<profile::RenderStats>,
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
    // The reactive mask the particle and transparent passes write.
    pub reactive_mask: super::reactive_mask::ReactiveMask,
}

// The world's scene assets: the IBL cubes and color-grading LUT, the area-light
// tables, and the shared static-mesh geometry. None of it depends on the
// swapchain.
pub(super) struct DxSceneAssets {
    // IBL resources. The fragment shader always samples these; when no
    // EnvironmentMap was supplied, both are 1x1 gray fallback cubes and
    // ViewUniforms::prefilter_mip_count is 0, so the shader uses the flat albedo
    // ambient term instead of IBL.
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
    pub dynamic_mode: concinnity_engine::render::rt_accel::geom::RtDynamicMode,
    // Whether the per-frame BVH update is failing, so a failure is logged once
    // per streak rather than every frame.
    pub update_streak: concinnity_engine::render::rt_accel::FailureStreak,
    // BVHs dropped mid-run (a refresh removed the last geometry), held until no
    // frame still in flight can trace them, timed against `retire_tick`, which
    // advances once per frame.
    pub retired: RetirePool<super::raytrace::RtAccelData>,
    pub retire_tick: u64,
    // Whether skinned meshes join the BVH (the launch's `--rt-skinned-geometry`
    // request; in by default). Clear it and the BVH covers static + instanced
    // geometry only, isolating the skinned trace path.
    pub skinned_geometry: bool,
    // The compute-skinning pipeline (`rt_skin`) skinned geometry joins the BVH
    // through, built with the RT pass. `None` when the kernel failed to compile.
    pub skin: Option<super::raytrace::SkinPipeline>,
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
    // Whether the GPU reports the DXR 1.1 tier. Reported as the ray-tracing
    // capability and checked again before any RT build.
    pub rt_capable: bool,
    // The resolved HDR-output mode. (The DXGI format is in `swapchain.format`.)
    pub hdr_mode: hdr::HdrOutputMode,
    // The swapchain config (ring depth + HDR request) this context was built
    // with, reported by `hot_swap_config` so a live editor reload reuses this
    // hardware only when the new world's config still matches.
    pub swapchain_config: init::SwapchainConfig,
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
            alloc: DeviceAllocator::new(
                &self.device,
                &self.command_queue,
                self.alloc.frames_in_flight(),
            ),
            device: self.device.clone(),
            command_queue: self.command_queue.clone(),
            adapter: self.adapter.clone(),
            info_queue: self.info_queue.clone(),
            rt_capable: self.rt_capable,
            hdr_mode: self.hdr_mode,
            swapchain_config: self.swapchain_config,
        })
    }

    // Frames in flight: the depth of every per-frame ring, fixed for this
    // hardware's lifetime (a change rebuilds it through `hot_swap_config`).
    pub(super) fn frames(&self) -> usize {
        self.alloc.frames_in_flight()
    }

    // Maximum extended-range multiplier on the HDR path, `None` on SDR.
    pub(super) fn max_edr(&self) -> Option<f32> {
        match self.hdr_mode {
            hdr::HdrOutputMode::Hdr { max_edr, .. } => Some(max_edr),
            hdr::HdrOutputMode::Sdr => None,
        }
    }

    // HDR encoding (scRGB-linear vs PQ) on the HDR path, `None` on SDR.
    pub(super) fn hdr_encoding(&self) -> Option<hdr::HdrEncoding> {
        match self.hdr_mode {
            hdr::HdrOutputMode::Hdr { encoding, .. } => Some(encoding),
            hdr::HdrOutputMode::Sdr => None,
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
    // The CPU-side scene: draw list, view, skinned slots, model history and
    // streamed-geometry placement. See [`SceneState`].
    pub(super) state: SceneState,
    // Scratch the graph executor refills each frame: the per-resource barrier
    // targets, the per-pass aliasing barriers and the per-pass recording slots.
    // Only the allocations carry over; taken during `execute_graph` and put back
    // after.
    pub(super) graph_scratch: Cell<Option<super::graph_exec::DxGraphScratch>>,
    // The frame's per-pass command lists in submission order, and the list the
    // submit call takes, kept across frames so building them allocates nothing.
    pub(super) pass_lists: Cell<Vec<ID3D12GraphicsCommandList>>,
    pub(super) submission: Vec<Option<ID3D12CommandList>>,

    // Geometry writes staged for the next copy submit. See [`GeometryUploads`].
    pub(super) geometry_uploads: RefCell<GeometryUploads>,

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

    // The shared bloom chain and the view of the pool's `bloom_top` it writes.
    // The octaves are shared across frame slots; the command queue runs frames
    // serially, so a frame's bloom writes never race a prior frame's composite
    // read. `None` only while a resize rebuilds it.
    pub(super) bloom: Option<BloomResources>,
    // Post-process tunables (bloom / exposure / vignette). Drives whether the
    // bloom chain runs and feeds the bloom-prefilter + composite root constants.
    pub(super) post_process: PostProcessParams,
    // The scale the frame's HDR chain is written at, advanced once per frame.
    pub(super) pre_exposure: concinnity_engine::gfx::auto_exposure::PreExposure,

    // Unified geometry G-buffer pre-pass. `Some` whenever any screen-space
    // consumer (SSR, SSGI, SSAO, TAA, or temporal upscaling) is enabled: one
    // jittered traversal writes view normal+depth, roughness, and motion into
    // one MRT that all those consumers read, replacing the separate SSR / SSAO
    // / velocity geometry pre-passes. See [`GbufferResources`].
    pub(super) gbuffer: Option<GbufferResources>,

    // Temporal anti-aliasing. `Some` only when `AntiAliasing.mode` is set;
    // when `None` the history resolve and the projection jitter are skipped and
    // the composite samples the HDR scene target directly.
    pub(super) taa: Option<TaaResources>,

    // The descriptor slots every shared fullscreen post pass allocates its
    // targets from. Held once for the backend rather than reserved per effect in
    // `init/heap_layout.rs`.
    pub(super) post: super::post::descriptors::PostDescriptors,
    // Feature state a settings change replaced, freed once no frame in flight
    // can name it.
    pub(super) retired_effects: concinnity_engine::render::buffers::retire_pool::TickedRetirePool<
        super::quality::RetiredEffect,
    >,

    // SSAO (GTAO). See [`SsaoState`].
    pub(super) ssao: SsaoState,

    // SSR. `Some` when `Reflections.screen_space` is set, or when SSGI is on
    // (SSGI reuses the depth + normal pre-pass G-buffer). The resolve half
    // (`ssr.resolve`) is `Some` only when SSR itself is authored on; with it
    // off the TAA / bloom / composite passes sample `hdr_srv_gpu` directly as
    // the scene color. When the resolve is on it writes into
    // `ssr.resolve.output`, whose SRV becomes the scene the post stack consumes
    // (see `scene_srv_for_post`).
    pub(super) ssr: Option<SsrResources>,

    // SSGI. `Some` only when `IndirectLighting.technique` is `ssgi`.
    // A hemisphere-gather + depth-aware-blur composite that bleeds nearby lit
    // surfaces' color onto one another, additively on top of the IBL ambient.
    // Reuses the SSR pre-pass G-buffer (so `ssr` is also `Some` whenever this
    // is); the render-graph `PassId::Ssgi` node is gated on `ssgi.is_some()`.
    pub(super) ssgi: Option<super::post::ssgi::SsgiResources>,

    // The reactive mask's disocclusion mark. `Some` while an upscaler that
    // reads the mask runs over the G-buffer pre-pass.
    pub(super) disocclusion: Option<super::post::disocclusion::DxDisocclusionPass>,

    // Roughness-aware reflection composite: the SSR / RT resolve writes reflected
    // radiance + weight, then this blurs it by surface roughness (a reduced-res blur
    // pass) and composites it over the scene into its own output -- the scene the
    // post stack consumes via `scene_srv_for_post`. `Some` when SSR resolve or RT is
    // authored (both feed it).
    pub(super) reflection_composite:
        Option<super::post::reflection_composite::DxReflectionCompositePass>,

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

    // The environment drawn behind the opaque scene. See `directx/sky.rs`.
    pub(super) sky: super::sky::DxSky,
    // The world's atmosphere, when it declares one: the sky, the haze and the
    // ambient light. See `directx/atmosphere.rs`.
    pub(super) atmosphere: Option<super::atmosphere::DxAtmosphere>,
    // The simulated oceans' cascade arrays, which every water draw binds.
    // `Some` whenever the world declared a `WaterSurface`.
    pub(super) water_sim: Option<super::water_sim::DxOceanSim>,
    // The wake's history, and this frame's block and lane once a frame that
    // runs it has advanced it.
    pub(super) wake: concinnity_engine::render::environment::water::wake::WakeHistory,
    pub(super) wake_frame: Option<concinnity_engine::render::environment::water::wake::WakeFrame>,

    // The shared `PassId::Transparent` slot and its two producers, translucent
    // glass panes and water surfaces. `Some` only when the world declared a
    // `GlassPanel` or a `WaterSurface`; with neither the field stays `None` and
    // the pass is skipped. Render-graph inclusion follows the shared
    // frame-graph gates. Mirrors src/metal's transparent encoder.
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

    // The grass field, when the world grows one and the GPU-driven main pass
    // exists to draw it in. See [`super::grass`].
    pub(super) grass: Option<super::grass::GrassResources>,
    // How this frame's view resolves the grass: where it coarsens, thins and
    // hands over to the canopy. `None` without grass.
    pub(super) grass_lod: Cell<Option<concinnity_engine::render::environment::grass::GrassLodView>>,
    // What the canopy block is filled from: the grass's far fields and the
    // planet's ground. `None` when the world has neither.
    pub(super) canopy_field: Option<concinnity_engine::render::environment::grass::CanopyField>,
    // The far field's canopy block, one copy per frame in flight. See
    // [`super::canopy`].
    pub(super) canopy: super::canopy::DxCanopy,

    // Per-frame command allocators + lists (start / per-pass / end). See
    // `DxCommands`.
    pub(super) commands: DxCommands,
    pub(super) diagnostics: Diagnostics,
    // CPU/GPU frame synchronization (one monotonic fence + per-slot values).
    // See `DxFrameSync`.
    pub(super) frame_sync: DxFrameSync,
    // Frame captures asked of `draw_frame`: their read-back buffers, the
    // captures awaiting their frame's fence, and the writer.
    pub(super) frame_capture: super::screenshot::DxFrameCaptures,
    pub(super) current_frame: usize,
    // `prepare_frame` already took the next frame's turn on the swapchain's
    // frame-latency waitable, which counts, so the draw must not wait on it again.
    pub(super) frame_latency_reserved: bool,

    pub(super) stream: StreamState,

    // Instanced-prop per-frame upload buffers + LOD buckets. See `DxInstanced`.
    pub(super) instanced: DxInstanced,
    // Lazily-built wireframe twin of the main-pass pipeline; empty until the
    // first Wireframe frame. See [`super::wireframe`].
    pub(super) wireframe: super::wireframe::DxWireframe,

    // Auto-exposure (EV adaptation) state. See [`AutoExposureState`].
    pub(super) auto_exposure: AutoExposureState,

    // Per-pass GPU timestamp queries. Read at the top of `draw_frame` after
    // the matching fence wait so the CPU sees a fully committed block.
    pub(super) timestamps: TimestampState,

    // Shader hot-reload state. See [`ShaderReload`].
    pub(super) hot_reload: ShaderReload,
    // The world default Shader's compiled programs, `None` for the engine's
    // own. Kept past init so the built-in shader reload rebuilds bucket 0 from
    // the world's pair rather than the engine's.
    pub(super) world_shader: Option<components::ShaderPrograms>,

    // Fixed descriptor-heap slots for the live-toggleable features outside the
    // post block (the G-buffer and the glass reflection render targets). Minted
    // once at init (the slots are reserved unconditionally) and stashed so a
    // settings rebuild can build a launched-off feature into its slot without
    // re-deriving the heap layout.
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
#[expect(
    clippy::let_underscore_must_use,
    reason = "the main thread is the same for every context a process makes"
)]
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
    #[expect(
        clippy::large_types_passed_by_value,
        reason = "keeps the `RenderBackend::draw_frame` signature it serves"
    )]
    pub(crate) fn draw_frame(&mut self, params: FrameParams<'_>) -> RenderResult<()> {
        drive_frame(self, params, concinnity_host::clock::monotonic_micros)
    }

    // Drain any queued D3D12 validation messages and emit them via tracing.
    pub(super) fn flush_validation(&self) {
        if let Some(ref iq) = self.hw.info_queue {
            drain_info_queue(iq);
        }
    }

    // Render statistics for the most recent `draw_frame`, for the profiler
    // overlay. `gpu_frame_us` is filled at the top of each `draw_frame` from
    // the timestamp pair this slot resolved on its previous trip through the
    // ring (so a frames-in-flight-stale window, matching Metal's "frame or two
    // stale" reading).
    pub(crate) fn render_stats(&self) -> profile::RenderStats {
        profile::RenderStats {
            spans: self.state.spans.get(),
            ..self.diagnostics.frame_stats.get()
        }
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
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a query on a live COM object; the descriptor it reads and the out-
        // parameters it fills are live locals that outlive the call.
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &raw mut info) }
            .map_or(0, |()| info.CurrentUsage)
    }

    // Shared atomic clone of the shader-reload flag, or `None` when the
    // context was built without hot-reload. Surfaced through the
    // `RenderBackend` trait so the debug server's
    // `reload-shaders` command can flip it from a non-render thread.
    // True when the render graph should include `PassId::Raymarch`. Wraps
    // the `raymarch.any_visible()` check so callers don't have to know
    // the resource is `Option`. Drives `FrameGraphInputs::raymarch_enabled`
    // in `record_frame::seed_inputs`.
    pub(super) fn raymarch_enabled(&self) -> bool {
        self.raymarch
            .as_ref()
            .is_some_and(super::raymarch::RaymarchResources::any_visible)
    }

    // True when the render graph should include `PassId::RtReflections` (and
    // omit `SsrResolve`). Both the resolve resources and the acceleration
    // structure must be live; either being absent (DXR unsupported, DXC missing,
    // accel build failed, or an empty scene) falls the graph back to SSR. Drives
    // `FrameGraphInputs::rt_reflections_enabled` in `record_frame::seed_inputs`
    // and the `scene_srv_for_post` precedence.
    pub(super) fn rt_reflections_active(&self) -> bool {
        self.reflection_path().rt_trace()
    }

    // Which reflection stages run this frame, from the authored SSR resolve and
    // the live RT pass and BVH. This backend reads the composite's output only
    // while a resolve writes it, so the RT node never runs without a BVH.
    pub(super) fn reflection_path(&self) -> ReflectionPath {
        ReflectionPath::new(
            self.ssr.as_ref().is_some_and(|s| s.resolve.is_some()),
            self.rt_reflections.is_some(),
            self.rt.accel.is_some(),
        )
    }

    // True when a reflection resolve (SSR resolve or RT reflections) runs this
    // frame. RT takes precedence at the graph level, so at most one resolve
    // runs; either feeds the same composite. Single-sources the predicate that
    // `scene_srv_for_post` and the glass pass use to pick the scene-with-
    // reflections target, and that the forward shader reads (via
    // `ViewUniforms::reflections_enabled`) to hand glossy dielectric specular to
    // that resolve instead of double-counting the forward probe reflection.
    pub(super) fn reflection_resolve_active(&self) -> bool {
        self.reflection_path().resolves()
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
                .is_some_and(super::transparent::TransparentResources::rt_pipelines_ready)
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
            .is_some_and(super::transparent::TransparentResources::mesh_pipelines_ready)
    }

    // Whether the see-through mesh (Layer 2) path is live this frame: the
    // pipelines built AND the pass can trace (`rt_transparent_active`, which
    // needs the TLAS). When false, those meshes render opaque + reflective in the
    // main pass (Layer 1) and the producer / opaque-skip / BLAS-exclude all stay
    // inert. Mirrors `MtlContext::mesh_glass_active`.
    pub(super) fn mesh_glass_active(&self) -> bool {
        self.seethrough_meshes_enabled() && self.rt_transparent_active()
    }

    // Whether a see-through mesh is visible and its pipelines are built, whether
    // or not the pass can trace this frame. The graph's Transparent node takes
    // the mesh producer only when both hold, so a world whose glass is all
    // hidden or evicted does not schedule it.
    pub(super) fn see_through_mesh_visible(&self) -> bool {
        self.seethrough_meshes_enabled()
            && self.transparent.as_ref().is_some_and(|t| {
                t.seethrough_mesh_indices().iter().any(|&i| {
                    self.state
                        .draw
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

    // Submit every staged geometry write. Queue order puts the copies ahead of
    // any later submission, so nothing waits on them.
    pub(super) fn flush_geometry_uploads(&self) -> RenderResult<()> {
        self.geometry_uploads.borrow_mut().flush(
            &self.hw.device,
            &self.hw.command_queue,
            &GeometryDest {
                vertex: &self.scene.geometry.vertex_buffer,
                index: &self.scene.geometry.index_buffer,
            },
        )
    }

    pub(crate) fn wait_idle(&self) {
        // Staged geometry writes count as submitted work, so an idle GPU has
        // applied them.
        if let Err(e) = self.flush_geometry_uploads() {
            tracing::error!("geometry upload submit failed: {e}");
        }
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
            && unsafe {
                self.frame_sync
                    .fence
                    .SetEventOnCompletion(val, self.frame_sync.fence_event)
            }
            .is_ok()
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

    pub(crate) fn window_minimized(&self) -> bool {
        window::is_minimized(self.win().hwnd)
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
        self.swapchain.present_sync_interval = u32::from(on);
    }

    // Switch window mode / resize at runtime (windowed / borderless / fullscreen
    // and content-size presets). The Win32 work lives in `window.rs`; the resize
    // path picks up the resulting WM_SIZE.
    pub(crate) fn set_window_mode(&mut self, mode: components::WindowMode) {
        window::set_window_mode(self.win_mut(), mode);
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "`forward!` reaches it from the trait's `&mut self` method of the same name"
    )]
    pub(crate) fn set_window_size(&mut self, width: u32, height: u32) {
        window::set_window_size(self.win(), width, height);
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
    pub(crate) fn update_post_process(&mut self, tunables: PostProcessTunables) {
        self.post_process.set_tunables(tunables);
    }

    // Set the live ambient (IBL) light scale (the Ambient slider). It lives in
    // `LightUniforms`, which rides a per-frame-in-flight CBV ring, so this
    // mutates the CPU-side copy and re-arms every slot; `record_frame` writes
    // the frame's own slot. Edge-triggered: a no-op when the value is unchanged
    // (e.g. an init push with no persisted override).
    pub(crate) fn set_ambient_intensity(&mut self, value: f32) {
        self.uniforms.lights.set_ambient(value);
    }

    // Move the local lights in place. The light, area and spot tables and the
    // spot slice CBVs are single UPLOAD buffers every in-flight frame reads,
    // so the GPU is drained before they are rewritten; the cluster reach, the
    // spot frusta and the spot refresh schedule are re-derived, and the light
    // CBV ring is re-armed for the point array.
    pub(crate) fn move_local_lights(
        &mut self,
        data: &lights::LightData,
        uniforms: &LightUniforms,
    ) -> RenderResult<()> {
        if !self.uniforms.lights.same_local_count(&data.lights)
            || data.spot_shadows.len() != self.spot_shadow.count() as usize
        {
            return Err(RenderError::Other(
                "move_local_lights: the light count changed since init".into(),
            ));
        }
        self.wait_idle();
        if !data.lights.is_empty() {
            super::draw::upload_static_records(
                &self.uniforms.local_light_buffer,
                &data.lights,
                "local-light",
            )?;
        }
        if !data.area_lights.is_empty() {
            super::draw::upload_static_records(
                &self.scene.area_light.buffer,
                &data.area_lights,
                "area-light",
            )?;
        }
        if !data.spot_shadows.is_empty() {
            super::draw::upload_static_records(
                &self.spot_shadow.buffer,
                &data.spot_shadows,
                "spot-shadow",
            )?;
            super::init::shadow::write_spot_uniforms(
                &self.spot_shadow.ubo,
                &data.spot_shadows,
                self.spot_shadow.ubo_stride,
            )?;
        }
        self.spot_shadow.frusta = data
            .spot_shadows
            .iter()
            .map(concinnity_engine::render::shadow::spot::slice_frustum)
            .collect();
        self.state.planner.shadow.reset_spots();
        self.uniforms.lights.move_local(&data.lights, uniforms);
        Ok(())
    }

    // Replace the live directional lights. The cascade shadow direction and the
    // fog sun are derived from the first light on the CPU, so both are
    // re-derived here. Edge-triggered: an unchanged set touches nothing, and a
    // changed one only re-arms the light CBV ring.
    pub(crate) fn update_directional_lights(&mut self, lights: &[components::DirectionalLight]) {
        self.uniforms.lights.set_directional(lights);
    }

    pub(crate) fn set_shadow_cadence(&mut self, cadence: init::ShadowCadence) {
        self.state.planner.shadow.cadence = cadence;
    }

    // Update the live scalar sub-tunables of the SSAO / SSR / SSGI / auto-exposure
    // passes without rebuilding anything. Each pass rebuilds its per-frame uniform
    // from these stored `*Settings` every draw (`settings.params(...)`), so
    // mutating the stored struct here is picked up on the next frame. Only a
    // feature whose resources are currently live has settings to mutate; the rest
    // are skipped (the value still persists for the next launch). SSAO / SSR /
    // auto-exposure are fully scalar, so they are replaced wholesale; SSGI's trace
    // resolution sizes its targets and rides its rebuild with its ray
    // count, so only its scalar intensity / distance are updated. The SSR settings live one level deeper than Metal's (inside the
    // optional `resolve` half), so a SSGI-only build with no resolve is skipped.
    pub(crate) fn update_quality_params(
        &mut self,
        g: &concinnity_engine::settings::ResolvedGraphics,
    ) {
        if let (Some(live), Some(res)) = (g.ssao(), self.ssao.resources.as_mut()) {
            res.settings = live;
        }
        if let (Some(live), Some(res)) = (g.ssr(), self.ssr.as_mut())
            && let Some(r) = res.resolve.as_mut()
        {
            r.settings = live;
        }
        if let (Some(live), Some(res)) = (g.ssgi(), self.ssgi.as_mut()) {
            res.settings.intensity = live.intensity;
            res.settings.max_distance = live.max_distance;
        }
        if let (Some(live), Some(cur)) = (g.auto_exposure(), self.auto_exposure.adaptation.as_mut())
        {
            cur.settings = live;
        }
        self.refresh_gates(g);
    }

    // Replace the runtime movement key map. The window message loop decodes
    // key events through it, so a settings-menu rebind takes effect immediately.
    pub(crate) fn set_keymap(&mut self, keymap: KeyMap) {
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
        super::gpu_profile::capabilities(self.hw.rt_capable)
    }

    // Coarse GPU performance profile for default-quality selection, read live
    // from the adapter description (vendor id + dedicated VRAM). `UNKNOWN` when
    // the adapter does not expose the v3 interface or the desc query fails.
    pub(crate) fn gpu_profile(&self) -> backend::GpuProfile {
        let Some(adapter) = self.hw.adapter.as_ref() else {
            return backend::GpuProfile::UNKNOWN;
        };
        // SAFETY: a property query on a live COM object; it only reads.
        match unsafe { adapter.GetDesc1() } {
            Ok(desc) => super::gpu_profile::adapter_profile(&desc),
            Err(_) => backend::GpuProfile::UNKNOWN,
        }
    }
}

// What `begin` acquired for a frame: its in-flight slot, how long it waited
// for it, and the "start" command list it closed.
pub(crate) struct DxFrame {
    frame: usize,
    gpu_wait: GpuWait,
    start_cmd: ID3D12GraphicsCommandList,
}

impl FrameDevice for DxContext {
    type Frame = DxFrame;

    fn frame_scene(&mut self) -> &mut SceneState {
        &mut self.state
    }

    fn draws_atmosphere(&self) -> bool {
        self.atmosphere.is_some()
    }

    fn prepare(&mut self, params: &FrameParams<'_>) -> RenderResult<()> {
        self.frame_capture.arm(params.capture);
        if let Some(atmo) = self.atmosphere.as_mut() {
            atmo.frame = params.atmosphere;
        }
        self.apply_pending_rebuilds()?;
        // A moved world's decals and grass move with it whatever happens to
        // history.
        if let Some(rebase) = &params.rebase {
            self.decal.set.rebase(rebase);
            self.rebase_grass(rebase);
            self.rebase_transparent_water(rebase);
            self.particle.system.rebase(rebase);
        }
        Ok(())
    }

    // The TAA and SSGI rings, and the upscaler's history on its next dispatch.
    fn reset_history(&mut self) {
        if let Some(taa) = self.taa.as_mut() {
            taa.pass.reset_history();
        }
        if let Some(ssgi) = self.ssgi.as_mut() {
            ssgi.reset_history();
        }
        if let Some(disocclusion) = self.disocclusion.as_mut() {
            disocclusion.reset_history();
        }
        if let Some(upscaler) = &self.upscale.backend {
            upscaler.request_history_reset();
        }
    }

    fn begin(&mut self, params: &FrameParams<'_>) -> RenderResult<Option<DxFrame>> {
        // Nothing a minimized window draws can be seen; park the frame until
        // it is restored. `window_closed` keeps pumping messages meanwhile.
        if self.window_minimized() {
            return Ok(None);
        }
        self.size_grass()?;

        let frame = self.current_frame;

        let gpu_wait = self.wait_frame_slot(frame)?;
        self.retire_frame_captures();
        self.reserve_frame_capture();
        // Staged geometry writes go ahead of everything this frame submits.
        self.flush_geometry_uploads()?;
        self.service_background_work(narrow_f32(params.elapsed), frame);
        let timings = self.read_gpu_timings(frame);
        let gpu_done = self.read_gpu_done(frame);
        self.timestamps.input_sampled_us[frame] = params.input_sampled_us;
        self.begin_frame_stats(gpu_wait, timings, gpu_done);

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
        let start_cmd = self.record_frame_start(frame, params.world_hidden)?;

        // Line resources: built on the first frame that publishes lines, so
        // the plan's graph gate can see them live this same frame and a world
        // that never draws a line never compiles them.
        self.ensure_line_pipeline(!params.lines.is_empty());

        Ok(Some(DxFrame {
            frame,
            gpu_wait,
            start_cmd,
        }))
    }

    fn plan(&mut self, _: &DxFrame, params: &FrameParams<'_>) -> RenderResult<FramePlan> {
        self.plan_frame(crate::directx::draw::PlanArgs {
            camera: camera_inputs(&self.state, params),
            elapsed: narrow_f32(params.elapsed),
            render_size: [
                self.targets.extent.render_width.max(1),
                self.targets.extent.render_height.max(1),
            ],
            lines: params.lines,
            world_hidden: params.world_hidden,
        })
    }

    fn execute(
        &mut self,
        frame: DxFrame,
        plan: &FramePlan,
        frame_params: &FrameParams<'_>,
    ) -> RenderResult<()> {
        let FrameParams {
            elapsed: elapsed_f64,
            water_time,
            fov_y_radians,
            near,
            cam_pos,
            text_calls,
            lines,
            world_hidden,
            grass_benders,
            water_stirs,
            particle_bursts,
            ..
        } = *frame_params;
        self.prepare_water_wake(cam_pos, water_time, water_stirs);
        let DxFrame {
            frame,
            gpu_wait,
            start_cmd,
        } = frame;
        // Reflection-probe count + records into this frame's ring CBV and records
        // buffer, which every probe-reading pass binds. A ring (one of each per
        // frame) so this write never races a prior frame's in-flight GPU read.
        self.upload_probe_set(frame)?;
        // The material parameter table's copy for this frame, after a change.
        if let Some(params) = self.cull.material_params.as_mut() {
            params.upload(frame);
        }

        let particle = self.prepare_particle_pass(narrow_f32(elapsed_f64), particle_bursts, frame);

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
        let recorded = self.record_frame(
            crate::directx::draw::RecordFrameTargets {
                cmd: end_cmd,
                back_buffer: &back_buffer,
                back_buffer_rtv,
                frame_idx: frame,
            },
            crate::directx::draw::RecordFrameView {
                elapsed: elapsed_f64,
                water_time,
                fov_y_radians,
                near,
                cam_pos,
                text_calls,
                lines,
                grass_benders,
                particle,
            },
            crate::directx::draw::RecordFrameResolution {
                width: self.targets.extent.render_width.max(1),
                height: self.targets.extent.render_height.max(1),
                output_width: self.targets.extent.output_width.max(1),
                output_height: self.targets.extent.output_height.max(1),
            },
            plan,
            world_hidden,
        )?;

        let end_cmd = self.commands.end_command_lists[frame].clone();
        let capture = self.record_frame_capture(&end_cmd, &back_buffer);

        self.finish_frame_stats();
        let waited_before_submit = gpu_wait.micros();
        let started = std::time::Instant::now();
        let mut pass_lists = recorded.pass_cmd_lists;
        let submitted = self.close_end_list(frame).and_then(|()| {
            self.submit_and_present(&start_cmd, &pass_lists, back_idx, frame, gpu_wait)
        });
        // The lists stay alive in their pool; only this frame's references drop.
        pass_lists.clear();
        self.pass_lists.set(pass_lists);
        let mut stats = self.diagnostics.frame_stats.get();
        // The present's block on the display already counts as GPU wait.
        let present_wait = stats.gpu_wait_us.saturating_sub(waited_before_submit);
        stats.submit_present_us =
            concinnity_host::clock::micros_since(started).saturating_sub(present_wait);
        self.diagnostics.frame_stats.set(stats);
        // A submitted frame's capture waits for the fence value it signaled; a
        // failed one is reported as a dropped frame by the caller.
        if let Some(capture) = capture
            && submitted.is_ok()
        {
            self.frame_capture
                .park(self.frame_sync.fence_values[frame], capture);
        }
        // A frame that took the history prime but never reached the GPU leaves
        // the ring unwritten, so the next frame primes instead.
        if submitted.is_err() && recorded.primed_model_history {
            self.state.model_history.get_mut().request_prime();
        }
        submitted
    }

    // What the TAA and SSGI accumulations wrote is next frame's history. After
    // `record_frame` rather than inside it, so the TAA write slot is stable
    // across the whole graph.
    fn finish(&mut self, plan: &FramePlan) {
        if plan.motion_read
            && let Some(taa) = self.taa.as_mut()
        {
            taa.pass.advance();
        }
        if let Some(ssgi) = self.ssgi.as_mut() {
            ssgi.advance();
        }
        if let Some(disocclusion) = self.disocclusion.as_mut() {
            disocclusion.advance();
        }
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
        // SAFETY: the swapchain handed this handle out in `build_swapchain`, it is closed exactly
        // once here, and no frame is waiting on it after `wait_idle`.
        unsafe { CloseHandle(self.swapchain.frame_latency) }.ok();
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
        if unsafe { iq.GetMessage(i, None, &raw mut len) }.is_err() {
            continue;
        }
        // A D3D12_MESSAGE is variable length: the fixed header followed by the
        // description text it points into. Back it with u64s rather than bytes so
        // the header lands on the alignment the struct needs.
        if len < size_of::<D3D12_MESSAGE>() {
            continue;
        }
        let mut buf = vec![0u64; len.div_ceil(size_of::<u64>())];
        let msg_ptr = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
        // SAFETY: `iq` is live, `len` is the size it just reported, and `msg_ptr`
        // addresses a live allocation of at least that many bytes.
        if unsafe { iq.GetMessage(i, Some(msg_ptr), &raw mut len) }.is_err() {
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
            unsafe { std::ffi::CStr::from_ptr(msg.pDescription.cast::<i8>()) }
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
