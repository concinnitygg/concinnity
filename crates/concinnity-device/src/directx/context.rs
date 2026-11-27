// D3D12 rendering context. Owns all GPU resources, the Win32 window, and input
// state. Mirrors the public API of VkContext / MtlContext so GraphicsSystem can
// drive all three backends identically.

use crate::gpu_wait::GpuWait;
use concinnity_core::math::{SaturatingFrom, narrow_f32};
use concinnity_engine::gfx::render_types::{
    InstancedCluster, PostProcessParams, ReceiverBounds, ShadowUniforms,
};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{BackendProbe, FrameParams, RenderBackend, WindowControl};
use concinnity_engine::render::buffers::retire_pool::RetirePool;
use concinnity_engine::render::frame::plan::{FrameDevice, FramePlan};
use concinnity_engine::render::post::reflection_path::ReflectionPath;
use concinnity_engine::render::scene_state::SceneState;
use concinnity_engine::render::shader::reload::ShaderReload;
use concinnity_engine::render::{
    backend, backend::init, backend::pass_timing, output::hdr, slots::rewrites,
};
use concinnity_engine::{components, profile};
use std::cell::{Cell, RefCell};
use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;

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
use crate::win32::platform_window::Win32Window;

// Constants
// Swapchain back buffers. Indexed by `GetCurrentBackBufferIndex`, independent of
// the frames-in-flight ring depth (`DxHardware::frames`).
pub(super) const BACK_BUFFERS: usize = 3;
// Constant buffer alignment required by D3D12.
pub(super) const CB_ALIGN: u64 = 256;

pub(super) const fn align256(n: u64) -> u64 {
    (n + CB_ALIGN - 1) & !(CB_ALIGN - 1)
}

// Where in each frame's shadow constant buffer the prop probe's block starts,
// after the frame's.
pub(super) const PROBE_SHADOW_OFFSET: u64 = align256(size_of::<ShadowUniforms>() as u64);

// Build the timestamp query heap and its persistently mapped readback buffer.
// All `None` when the queue has no timestamps or an allocation fails;
// `gpu_frame_us` then stays zero.
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
    // One block per frame in flight: the whole-frame pair, then a pair per timing
    // slot (see `pass_timing`). Each pass brackets its encode; the end list
    // resolves the block, read back after the fence wait. Passes bundled into
    // another encoder, or no-ops here, stay zero.
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

// The off-screen HDR scene target, tonemapped onto the back buffer by the
// composite. With MSAA, `color` is multisampled and resolves into `resolve`
// (and `resolve_rtv` is `Some`); without, `color` is single-sample.
// `srv_gpu` points at what the composite samples.
pub(super) struct HdrState {
    pub color: ID3D12Resource,
    pub color_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub resolve: Option<ID3D12Resource>,
    pub resolve_rtv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub srv_gpu: SrvSlot,
    pub msaa_samples: u32,
}

// Rectangular area lights: the `AreaLightData` table plus the two LTC lookup
// tables, all static. The tables are uploaded even with no area light.
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
// Vulkan's `Uniforms`. The view, light and shadow buffers are all per-frame
// and persistently mapped. The COM resources auto-release on drop; only the
// persistent mappings need an explicit `unmap`.
pub(super) struct Uniforms {
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
    // Per frame, the frame's `ShadowUniforms` and at `PROBE_SHADOW_OFFSET`
    // the prop probe's.
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

impl Uniforms {
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

// Per-frame command lists: a start list for the timestamp pre-init, one list
// per (frame, pass) for parallel recording, and an end list for Composite and
// the timestamp resolve.
pub(super) struct Commands {
    pub command_allocators: Vec<ID3D12CommandAllocator>,
    pub command_lists: Vec<ID3D12GraphicsCommandList>,
    pub pass_allocators: Vec<ID3D12CommandAllocator>,
    pub pass_cmd_lists: Vec<ID3D12GraphicsCommandList>,
    pub end_command_allocators: Vec<ID3D12CommandAllocator>,
    pub end_command_lists: Vec<ID3D12GraphicsCommandList>,
}

// Frame synchronization: one monotonic fence with per-slot values.
// `fence_event` needs a `CloseHandle` in `DxContext::drop`.
pub(super) struct FrameSync {
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
// `DxContext`.
pub(super) struct Geometry {
    pub vertex_buffer: PooledBuffer,
    pub index_buffer: PooledBuffer,
    pub vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    pub index_buffer_view: D3D12_INDEX_BUFFER_VIEW,
}

// Instanced-prop clusters, grouped off the flat `DxContext`. Every instance
// folds into the GPU-driven cull records, so no pass walks the instances on the
// CPU.
pub(super) struct InstancedState {
    pub clusters: Vec<InstancedCluster>,
    // Whether any cluster declares LOD alternates. False skips the per-frame
    // per-instance LOD patch: without alternates the base slice written into
    // every frame's draw-args buffer at init is right for the world's life.
    pub any_lod: bool,
}

impl InstancedState {
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
pub(super) struct Descriptors {
    // CBV/SRV/UAV descriptor heap (shader-visible). The slot map is
    // `init/heap_layout.rs`.
    pub srv_heap: ID3D12DescriptorHeap,
    pub srv_descriptor_size: usize,
    // Every block's slot in `srv_heap`. The flat bindless pool repeats once per
    // frame in flight, so streaming re-points each copy as its frame's fence
    // waits. Also holds the probe cube block and the convolution descriptors.
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
    // The cube array the bake writes a cube of per placement, and the per-frame
    // record buffers. Distinct from `env_map` so the background + diffuse
    // irradiance keep the sky.
    pub gpu: super::probe_set::ProbeSetGpu,
    // At most one probe whose six cube faces are rendering on the GPU (one face
    // per frame, into the reserved-ring-slot capture resources) and one prior
    // probe whose capture is convolving into its cube, one destination mip per
    // frame.
    pub bake: super::probe::ProbeBake,
    // The three convolution kernels and their root signatures, built at init under
    // the same gate the bake needs. `None` disables baking.
    pub prefilter: Option<super::probe_prefilter::ProbePrefilterPipelines>,
    // The world's prop probe, captured whole every frame it rides.
    pub prop: super::prop_probe::PropProbe,
}

impl ProbeState {
    // Empty until `set_reflection_probes` supplies placements (declared or
    // auto-seeded).
    pub(super) fn new(
        prefilter: Option<super::probe_prefilter::ProbePrefilterPipelines>,
        gpu: super::probe_set::ProbeSetGpu,
    ) -> Self {
        Self {
            gpu,
            bake: super::probe::ProbeBake::default(),
            prefilter,
            prop: super::prop_probe::PropProbe::default(),
        }
    }
}

// Stall-free texture streaming. A slot swap re-points each frame's flat-pool
// SRV copy after that frame's fence wait (`pool_rewrites`), and parks the old
// resource and upload transients on `retires` for `retire_depth` ticks, so
// nothing a pending list or the upload still reads is released.
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
    // Ahead of `handle`, so the buffers' references to the swapchain are gone
    // by the time it is released.
    pub back_buffers: Vec<ID3D12Resource>,
    pub handle: IDXGISwapChain3,
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
    // The back buffer most recently presented, `None` before the first present;
    // `screenshot` copies it. Recorded at present, since the current index has
    // already moved on.
    pub last_present_index: Option<usize>,
}

// The two resolutions: `render_*` for the scene and its screen-space targets,
// reduced under temporal upscaling; `output_*` for the swapchain, bloom, the
// upscaler output and composite. Equal without upscaling.
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

// HUD text: its root signature, its PSO (`None` until text is first
// published), and per-frame upload buffers for text geometry.
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
    // Draw calls the pass encoders bump; atomic because passes record on workers.
    pub draw_calls_accum: std::sync::atomic::AtomicU32,
}

// The render-resolution scene targets: the HDR color target, the main depth
// buffer and its sampling SRV, the two pipeline resolutions, and the render
// graph's transient pool. The resize path rebuilds all of it.
pub(super) struct Targets {
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
pub(super) struct SceneAssets {
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
    pub geometry: Geometry,
    // Shared texture pool: every texture at its handle, so draw slots index it
    // directly. `_fallback_textures` holds only the flat-normal and white
    // fallbacks past the last real texture, kept so their SRVs stay resident.
    pub textures: Vec<PooledTexture>,
    pub _fallback_textures: Vec<PooledTexture>,
}

// The hardware ray-tracing scene: the acceleration structures and the policy
// that keeps them current as the draw set changes.
pub(super) struct RtState {
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
    // Streamline bound to `device`, when it loaded. First, so its session
    // shuts down before the device is released.
    pub streamline: Option<super::streamline::DxStreamline>,
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
    // The window, moved into the successor context on `reload_world`; `None`
    // only on the outgoing context.
    pub window: Option<Win32Window>,
}

impl DxHardware {
    // The hardware an outgoing context hands its successor on a live editor
    // `reload_world`: COM clones of the device, queue, adapter and info queue,
    // with the window and fullscreen restore state moved out. The successor
    // places into a fresh allocator; the outgoing world releases into its own.
    pub(super) fn hand_over(&mut self) -> RenderResult<Self> {
        Ok(Self {
            window: Some(
                self.window.take().ok_or_else(|| {
                    RenderError::Other("reload_world: window already taken".into())
                })?,
            ),
            alloc: DeviceAllocator::new(
                &self.device,
                &self.command_queue,
                self.alloc.frames_in_flight(),
            ),
            streamline: self.streamline.clone(),
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

    // Render-resolution scene targets. See [`Targets`].
    pub(super) targets: Targets,

    // Temporal upscaling (AMD FidelityFX FSR3). See [`UpscaleState`].
    pub(super) upscale: UpscaleState,

    // Shadow map resources. See [`ShadowState`].
    pub(super) shadow: ShadowState,

    // Spot shadow map resources. See [`SpotShadowState`].
    pub(super) spot_shadow: SpotShadowState,

    // Scene assets. See [`SceneAssets`].
    pub(super) scene: SceneAssets,

    // Shader-visible descriptor heaps + samplers + scene texture pools. See
    // `Descriptors`.
    pub(super) descriptors: Descriptors,
    // The CPU-side scene: draw list, view, skinned slots, model history and
    // streamed-geometry placement. See [`SceneState`].
    pub(super) state: SceneState,
    // Scratch the graph executor refills each frame: the per-resource barrier
    // targets, the per-pass aliasing barriers and the per-pass recording slots.
    // Only the allocations carry over; taken during `execute_graph` and put back
    // after.
    pub(super) graph_scratch: Cell<Option<super::graph_exec::GraphScratch>>,
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
    // See `Uniforms`.
    pub(super) uniforms: Uniforms,

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

    // Unified G-buffer pre-pass. `Some` whenever any screen-space consumer (SSR,
    // SSGI, SSAO, TAA, upscaling) is on.
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

    // SSR, `Some` when authored or for SSGI's shared G-buffer. The resolve half is
    // `Some` only when SSR itself is authored; its output becomes the post stack's
    // scene (`scene_srv_for_post`).
    pub(super) ssr: Option<SsrResources>,

    // SSGI, `Some` when `IndirectLighting.technique` is `ssgi`: a hemisphere
    // gather added over the IBL ambient.
    pub(super) ssgi: Option<super::post::ssgi::SsgiResources>,

    // The reactive mask's disocclusion mark. `Some` while TAA or an upscaler
    // that reads the mask runs over the G-buffer pre-pass.
    pub(super) disocclusion: Option<super::post::disocclusion::DxDisocclusionPass>,

    // Roughness-aware reflection composite over the SSR / RT resolve, whose output
    // is the post stack's scene. `Some` when either resolve is authored.
    pub(super) reflection_composite:
        Option<super::post::reflection_composite::DxReflectionCompositePass>,

    // Hardware ray-traced reflections (DXR): the pass and the BVH, `Some` when RT
    // is on, the GPU supports the DXR tier, and the build succeeded; otherwise
    // the graph uses `SsrResolve`. RT takes that slot and reads the same G-buffer.
    pub(super) rt_reflections: Option<super::post::rt_reflections::RtReflectionsResources>,
    // The ray-tracing scene. See [`RtState`].
    pub(super) rt: RtState,

    // Projected decals. See [`DecalState`].
    pub(super) decal: DecalState,

    // World-space line pass state: the resources, built on the first frame
    // that publishes lines. See [`super::line::LineState`].
    pub(super) lines: super::line::LineState,

    // Raymarched SDF volumes. `Some` when the world declares at least one
    // `SdfVolume`. Render-graph `PassId::Raymarch` is gated on the frame
    // content's visible volumes, so worlds with no visible SDF skip the slot
    // entirely.
    pub(super) raymarch: Option<super::raymarch::RaymarchResources>,

    // The environment drawn behind the opaque scene. See `directx/sky.rs`.
    pub(super) sky: super::sky::SkyState,
    // The world's atmosphere, when it declares one: the sky, the haze and the
    // ambient light. See `directx/atmosphere.rs`.
    pub(super) atmosphere: Option<super::atmosphere::AtmosphereState>,
    // The simulated oceans' cascade arrays, which every water draw binds.
    // `Some` whenever the world declared a `WaterSurface`.
    pub(super) water_sim: Option<super::water_sim::OceanSim>,
    // The wake's history and this frame of it.
    pub(super) wake: concinnity_engine::render::environment::water::wake::WakeHistory,

    // The transparent pass's glass and water producers, `Some` when the world
    // declares either.
    pub(super) transparent: Option<super::transparent::TransparentResources>,

    // Planar reflections: one mirror render per distinct reflector plane, `Some`
    // when the world assigns reflectors a planar slot. Driven at the head of the
    // transparent pass.
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
    pub(super) canopy: super::canopy::CanopyState,

    // Per-frame command allocators + lists (start / per-pass / end). See
    // `Commands`.
    pub(super) commands: Commands,
    pub(super) diagnostics: Diagnostics,
    // CPU/GPU frame synchronization (one monotonic fence + per-slot values).
    // See `FrameSync`.
    pub(super) frame_sync: FrameSync,
    // Frame captures asked of `draw_frame`: their read-back buffers, the
    // captures awaiting their frame's fence, and the writer.
    pub(super) frame_capture: super::screenshot::FrameCaptures,
    pub(super) current_frame: usize,
    // `prepare_frame` already took the next frame's turn on the swapchain's
    // frame-latency waitable, which counts, so the draw must not wait on it again.
    pub(super) frame_latency_reserved: bool,

    pub(super) stream: StreamState,

    // Instanced-prop per-frame upload buffers + LOD buckets. See `InstancedState`.
    pub(super) instanced: InstancedState,
    // Lazily-built wireframe twin of the main-pass pipeline; empty until the
    // first Wireframe frame. See [`super::wireframe`].
    pub(super) wireframe: super::wireframe::Wireframe,

    // Auto-exposure (EV adaptation) state. See [`AutoExposureState`].
    pub(super) auto_exposure: AutoExposureState,
    // The shadow receiver reduction and its readback ring, built beside the
    // Hi-Z pyramid and the cascades.
    pub(super) shadow_receivers: Option<super::shadow_receivers::ShadowReceiversGpu>,
    // The view sampling and its readback ring, built beside the Hi-Z pyramid.
    pub(super) view_samples: Option<super::view_samples::ViewSamplesGpu>,

    // Per-pass GPU timestamp queries. Read at the top of `draw_frame` after
    // the matching fence wait so the CPU sees a fully committed block.
    pub(super) timestamps: TimestampState,

    // Shader hot-reload state. See [`ShaderReload`].
    pub(super) hot_reload: ShaderReload,
    // The world default Shader's compiled programs, `None` for the engine's
    // own. Kept past init so the built-in shader reload rebuilds bucket 0 from
    // the world's pair rather than the engine's.
    pub(super) world_shader: Option<components::ShaderPrograms>,

    // The heap slots reserved for live-toggleable features (G-buffer, glass
    // reflection targets), kept so a later enable builds into them.
    pub(super) quality_slots: super::quality::QualitySlotHandles,

    // Scene-captured reflection probes. See [`ProbeState`].
    pub(super) probe: ProbeState,

    // Device, queue, allocator and window. Declared last so every resource
    // above releases before the device and the window go.
    pub(super) hw: DxHardware,
}

impl DxContext {
    // Drain any queued D3D12 validation messages and emit them via tracing.
    pub(super) fn flush_validation(&self) {
        if let Some(ref iq) = self.hw.info_queue {
            drain_info_queue(iq);
        }
    }

    // GPU memory residency in bytes from the adapter's local budget, or zero on a
    // pre-WDDM 2.0 adapter.
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

    // True when the graph runs `RtReflections` instead of `SsrResolve`: both the
    // resolve and the acceleration structure are live.
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

    // True when a reflection resolve (SSR or RT) runs this frame; the predicate
    // `scene_srv_for_post` and the glass pass use to pick the scene target.
    pub(super) fn reflection_resolve_active(&self) -> bool {
        self.state.planner.reads().reflections
    }

    // The single-sample scene spine the graph drives as `hdr_resolve`, always in
    // RENDER_TARGET between passes; a pass needing another state transitions back.
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

    // True when the transparent pass traces this frame: RT is live and every live
    // producer's RT pipelines built. The graph's planar skip uses the same test,
    // so a producer still on its fallback keeps its mirror.
    pub(super) fn rt_transparent_active(&self) -> bool {
        self.rt_reflections_active()
            && self
                .transparent
                .as_ref()
                .is_some_and(super::transparent::TransparentResources::rt_pipelines_ready)
    }

    // Whether a material opts into see-through glass and the mesh pipelines built,
    // regardless of RT being live: the BLAS build must exclude these meshes before
    // the acceleration structure exists.
    pub(super) fn seethrough_meshes_enabled(&self) -> bool {
        self.transparent
            .as_ref()
            .is_some_and(super::transparent::TransparentResources::mesh_pipelines_ready)
    }

    // Whether see-through meshes draw here this frame: pipelines built and the pass
    // can trace. Otherwise they render opaque in the main pass.
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

    // Count CPU-issued draw calls; fullscreen post passes are not counted.
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
    pub(super) fn platform_window(&self) -> &Win32Window {
        self.hw
            .window
            .as_ref()
            .expect("DxContext window taken by reload_world")
    }

    #[inline]
    pub(super) fn platform_window_mut(&mut self) -> &mut Win32Window {
        self.hw
            .window
            .as_mut()
            .expect("DxContext window taken by reload_world")
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

    // The slot this frame reuses holds what the frame before it measured,
    // retired by the fence `begin` waited on.
    fn retired_receivers(&mut self, frame: &DxFrame) -> Option<ReceiverBounds> {
        self.shadow_receivers.as_ref()?.read(frame.frame)
    }

    fn measure_view(&mut self, frame: &DxFrame) {
        if let Some(cells) = self.view_samples.as_ref().and_then(|v| v.read(frame.frame)) {
            self.state.planner.view.measured(cells);
        }
    }

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
            self.state.decals.rebase(rebase);
            self.rebase_grass(rebase);
            self.rebase_transparent_water(rebase);
            self.state.particles.rebase(rebase);
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
        if let Some(composite) = self.reflection_composite.as_mut() {
            composite.reset_history();
        }
        if let Some(upscaler) = &self.upscale.backend {
            upscaler.request_history_reset();
        }
    }

    fn begin(&mut self, params: &FrameParams<'_>) -> RenderResult<Option<DxFrame>> {
        // Nothing a minimized window draws can be seen; park the frame until
        // it is restored. `window_closed` keeps pumping messages meanwhile.
        if self.platform_window().window_minimized() {
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
        self.begin_frame_stats(gpu_wait, &timings, gpu_done);

        // Flush any D3D12 validation messages from the previous frame.
        self.flush_validation();

        // The frame's lists: `start` (timestamp pre-init), one per non-composite pass
        // recorded in parallel, and `end` (Composite, final timestamp, resolve,
        // restores), submitted together in graph order.
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
        let render_size = [
            self.targets.extent.render_width.max(1),
            self.targets.extent.render_height.max(1),
        ];
        self.plan_frame(params, render_size)
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
            cam_pos,
            water_stirs,
            particle_bursts,
            ..
        } = *frame_params;
        self.prepare_water_wake(
            cam_pos,
            water_time,
            plan.passes
                .get(concinnity_engine::render::render_graph::PassId::WaterSim)
                .runs,
            water_stirs,
        );
        let DxFrame {
            frame,
            gpu_wait,
            start_cmd,
        } = frame;
        self.prepare_prop_probe(frame_params.prop_probe, plan.inputs.prop_probe_enabled);
        let planar = self.plan_mirrors(plan);
        // Reflection-probe count + records into this frame's ring CBV and records
        // buffer, which every probe-reading pass binds. A ring (one of each per
        // frame) so this write never races a prior frame's in-flight GPU read.
        self.upload_probe_set(frame)?;
        // The material parameter table's copy for this frame, after a change.
        if let Some(params) = self.cull.material_params.as_ref() {
            params.upload(&mut self.state.material_params, frame);
        }

        let particle = self.prepare_particle_pass(narrow_f32(elapsed_f64), particle_bursts, frame);

        // Open the end list for Composite, the final timestamp, the resolve and the
        // restore barriers.
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

        // Record the passes: non-composite ones on workers into their own lists,
        // Composite and the restores onto `end_cmd`. Returns the pass lists in graph
        // order.
        let recorded = self.record_frame(
            crate::directx::draw::RecordFrameTargets {
                cmd: end_cmd,
                back_buffer: &back_buffer,
                back_buffer_rtv,
                frame_idx: frame,
            },
            crate::directx::draw::RecordFrameResolution {
                width: self.targets.extent.render_width.max(1),
                height: self.targets.extent.render_height.max(1),
                output_width: self.targets.extent.output_width.max(1),
                output_height: self.targets.extent.output_height.max(1),
            },
            plan,
            frame_params,
            particle,
            planar,
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
        if let Some(composite) = self.reflection_composite.as_mut() {
            composite.advance();
        }
    }
}

impl Drop for DxContext {
    fn drop(&mut self) {
        self.wait_idle();
        // The upscaler first: a DLSS viewport is freed through the Streamline
        // session the hardware holds.
        self.upscale.backend = None;
        // Persist and release the pipeline library. `window` is `None` only
        // on the outgoing context of a `reload_world`, whose successor keeps
        // the device and the installed library.
        if self.hw.window.is_some() {
            super::pso_library::shutdown();
            // Also writes whatever shader artifacts were compiled lazily since
            // init's own checkpoint.
            crate::shader::runtime_cache::checkpoint();
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

impl BackendProbe for DxContext {
    // Device capability flags for the settings menu. RT reflects the DXR-tier
    // query made at init (`hw.rt_capable`).
    fn capabilities(&self) -> backend::DeviceCapabilities {
        super::gpu_profile::capabilities(self.hw.rt_capable)
    }

    // Coarse GPU performance profile for default-quality selection, read live
    // from the adapter description (vendor id + dedicated VRAM). `UNKNOWN` when
    // the adapter does not expose the v3 interface or the desc query fails.
    fn gpu_profile(&self) -> backend::GpuProfile {
        let Some(adapter) = self.hw.adapter.as_ref() else {
            return backend::GpuProfile::UNKNOWN;
        };
        // SAFETY: a property query on a live COM object; it only reads.
        match unsafe { adapter.GetDesc1() } {
            Ok(desc) => super::gpu_profile::adapter_profile(&desc),
            Err(_) => backend::GpuProfile::UNKNOWN,
        }
    }

    // Render statistics for the last `draw_frame`. `gpu_frame_us` is
    // `frames_in_flight` frames stale, like the other backends.
    fn render_stats(&self) -> profile::RenderStats {
        profile::RenderStats {
            spans: self.state.spans.get(),
            view: *self.state.planner.view.facts(),
            relevance_off: self.state.planner.relevance_off(),
            ..self.diagnostics.frame_stats.get()
        }
    }
}
