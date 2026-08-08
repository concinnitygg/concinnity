// src/backend.rs
//
// RenderBackend trait: the union of methods every graphics backend
// implements, dispatched dynamically by GraphicsSystem so the per-frame
// step + setup logic lives in one cfg-free copy instead of three.
//
// Each concrete backend (MtlContext / DxContext / VkContext) supplies a
// thin forwarder impl that delegates to the existing inherent methods
// (see metal/backend.rs, directx/backend.rs, vulkan/backend.rs).
//
// Two cross-backend signature variances are handled here:
//   - `upload_skinned`: Metal uses three shader payloads (vert + frag +
//     shadow); DX/VK use one (frag). The trait method takes all three;
//     DX/VK ignore the unused bytes.
//   - `setup_chunk_streaming`: Metal binds chunk textures per draw and
//     ignores the (texture_slot, normal_map_slot) args; DX/VK bake them
//     into a shared descriptor at setup time.
//
// `render_stats` is Metal-only today and has a default no-op impl so DX/VK
// don't need to override it.

use crate::auto_exposure::AutoExposureSettings;
use crate::backend_init::{BackendInit, ShaderBytes, SwapchainConfig};
use crate::error::{RenderError, RenderResult};
use crate::input::RenderInput;
use crate::keymap::KeyMap;
use crate::mesh_payload::{SkinnedVertex, Vertex};
use crate::profile::RenderStats;
use crate::render_types::{
    LineVertex, MaterialUniforms, PostProcessParams, SkinnedDrawObject, TextDrawCall,
};
use crate::rt_reflections::RtReflectionSettings;
use crate::scene_flow::SceneControl;
use crate::ssao::SsaoSettings;
use crate::ssgi::SsgiSettings;
use crate::ssr::SsrSettings;
use crate::volumetric_fog::FogSettings;

// Per-frame inputs for [`RenderBackend::draw_frame`]. `world_hidden` is set when
// an opaque menu backdrop covers the scene: the backend skips every world pass
// and presents only the overlay (`text_calls`) over a cleared target.
#[derive(Clone, Copy)]
pub struct FrameParams<'a> {
    pub elapsed: f32,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    pub cam_pos: [f32; 3],
    pub text_calls: &'a [TextDrawCall],
    // Expanded line ribbons (`lines::build_vertices`) for this frame's camera,
    // drawn depth-tested into the scene after the world passes. Empty on any
    // frame that submits no lines, which also drops the pass from the graph.
    pub lines: &'a [LineVertex],
    pub world_hidden: bool,
    // Viewport view mode + show flags for the frame (`ViewOverrides` when the
    // editor publishes one, defaults otherwise). Backends run their seeded
    // graph inputs through `render_graph::apply_view` and steer the composite
    // by the mode.
    pub view_mode: concinnity_core::gfx::view_modes::ViewMode,
    pub show: concinnity_core::gfx::view_modes::ShowFlags,
}

// One streamed chunk's geometry plus placement, supplied to
// [`RenderBackend::add_chunk_mesh`]. `frame` reclaims retired deferred frees
// before the chunk is placed in the streaming headroom.
#[derive(Clone, Copy)]
pub struct ChunkMesh<'a> {
    pub verts: &'a [Vertex],
    pub idxs: &'a [u16],
    pub model: [[f32; 4]; 4],
    pub texture_slot: usize,
    pub normal_map_slot: usize,
    pub material: MaterialUniforms,
    pub frame: u64,
}

// One draw slot's fresh geometry, supplied to
// [`RenderBackend::rebuild_static_geometry`] when an asset hot-reload
// changed its vertex / index count and the slot can no longer hold the new
// data in place. The backend rebuilds the entire shared vertex / index
// buffer; draws not named here keep their current geometry, copied byte-for-
// byte from the live buffers. `indices` are mesh-relative (0-based); the
// backend rebases them onto whatever new vertex region the draw lands in.
#[allow(dead_code)] // consumed by Metal's rebuild_static_geometry; no-op on DirectX / Vulkan.
pub struct DrawGeometryUpdate {
    pub draw_idx: usize,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    // One slice per additional LOD, ordered mip 0 → mip N-1. Each is
    // `(switch_distance, mesh-relative indices)`. Empty for meshes
    // declared `lod_levels <= 1`.
    pub lod_alternates: Vec<(f32, Vec<u16>)>,
}

// One skinned draw slot's fresh geometry, supplied to
// [`RenderBackend::rebuild_skinned_geometry`] when an asset hot-reload
// changed its vertex / index count and the slot can no longer hold the new
// data in its existing region of the shared skinned vertex / index buffers.
// The backend rebuilds both shared buffers; slots not named here keep their
// current geometry, copied byte-for-byte from the live buffers and re-based
// onto whatever new vertex region they land in. `indices` are mesh-relative
// (0-based); the backend rebases them onto the new vertex region.
#[allow(dead_code)] // consumed by Metal's rebuild_skinned_geometry; no-op on DirectX / Vulkan.
pub struct SkinnedDrawGeometryUpdate {
    pub skinned_index: usize,
    pub vertices: Vec<SkinnedVertex>,
    pub indices: Vec<u16>,
}

// The post-rebuild layout for one skinned slot, returned by
// [`RenderBackend::rebuild_skinned_geometry`] so the asset hot-reload
// helper can refresh its `SkinnedMeshSourceEntry`s'
// `vertex_base` / `vertex_count` / `index_count` to point at the new
// regions. Returned for every slot (both the ones whose geometry was
// replaced and the ones whose geometry was carried over) because the
// rebuild may have shifted every slot's `vertex_base`.
// Constructed only by the `cn debug` binary's skinned-rebuild reload pass;
// reads as dead under `cargo check --lib`.
#[allow(dead_code)]
pub struct SkinnedSlotLayout {
    pub skinned_index: usize,
    pub vertex_base: u16,
    pub vertex_count: usize,
    pub index_count: usize,
}

// The resolved per-feature quality settings for [`RenderBackend::apply_quality_settings`].
// `GraphicsSystem` derives these from its stored `PostProcessConfig` (with the
// user's persisted toggle overrides applied) whenever a Quality-group toggle
// changes, so the backend receives ready-to-use settings rather than re-deriving
// from the asset. Each `Option` mirrors the init-time gate: `None` means the
// feature is off and its passes / resources should be torn down; `Some` means it
// is on and its resources should exist. A backend without a live-rebuild path
// ignores this (the choice still persists and applies at the next launch).
#[allow(dead_code)] // fields read only by Metal's apply_quality_settings.
pub struct QualitySettings {
    // Temporal anti-aliasing on/off (the `Taa` anti-aliasing mode). The backend
    // additionally suppresses TAA while temporal upscaling is active (the scaler
    // does its own accumulation). The other anti-aliasing modes are the composite
    // FXAA edge filter, which rides `PostProcessParams.fxaa` (pushed via
    // `update_post_process`), not this pass-rebuild payload.
    pub taa: bool,
    pub ssao: Option<SsaoSettings>,
    pub ssr: Option<SsrSettings>,
    // Hardware ray-traced reflections. The backend further gates this on GPU
    // ray-tracing support, falling back to leaving it off when unsupported.
    pub rt_reflections: Option<RtReflectionSettings>,
    pub ssgi: Option<SsgiSettings>,
    // Per-axis divisor for the roughness-aware reflection blur target (the
    // reduced-resolution first pass of the SSR / RT reflection composite),
    // resolved from `PostProcessConfig.reflection_blur_resolution`. Every backend
    // sizes its blur target at render / this on a live reflection rebuild.
    pub reflection_blur_scale: u32,
    pub auto_exposure: Option<AutoExposureSettings>,
    // The authored exposure bias (stops) auto-exposure applies on top of its
    // adapted value; carried so a live auto-exposure enable matches init.
    pub auto_exposure_bias_ev: f32,
}

// GPU/device capability flags, queried from the backend once it is built.
// Surfaced so the settings menu can gray out (and make inert) toggles the
// device cannot honor -- e.g. ray-traced reflections on a GPU without hardware
// ray tracing. Mirrors an RHI-style capability set: a handful of bools held in
// memory and re-queried each launch, never persisted, so it is always correct
// for the current device + driver.
#[derive(Clone, Copy, Debug)]
pub struct DeviceCapabilities {
    // Hardware ray tracing for the RT-reflections pass: DXR 1.1 on DirectX, the
    // ray-query device extensions on Vulkan (and not under XeSS), and
    // `MTLDevice::supportsRaytracing` on Metal.
    pub ray_tracing: bool,
    // Whether the upscaler implementation is a choice (FSR3 / DLSS / XeSS)
    // rather than fixed. DirectX and Vulkan offer the selection; Metal always
    // upscales through MetalFX, so the row has nothing to pick.
    pub selectable_upscaler: bool,
}

impl DeviceCapabilities {
    // Every capability present. The trait default, so a backend that does not
    // report capabilities never wrongly disables a toggle (it keeps the prior
    // behavior: the feature no-ops with a warning on an incapable device).
    pub const ALL: Self = Self {
        ray_tracing: true,
        selectable_upscaler: true,
    };
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self::ALL
    }
}

// Coarse GPU vendor class, derived per backend from the adapter's reported
// vendor id (DirectX / Vulkan) or unified-memory / Apple-family signals (Metal).
// Used only to pick default quality and to gate vendor-specific options (e.g.
// which upscalers to offer); never persisted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuVendor {
    Apple,
    Nvidia,
    Amd,
    Intel,
    Other,
}

// Coarse performance class for default-quality selection, ordered low -> high so
// callers can compare with `>=`. Each backend maps its native signals (memory
// budget, discrete / integrated, Apple GPU family) onto this via `classify_tier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuTier {
    // Unknown hardware: the conservative default, never the top preset. Sorts
    // lowest so a comparison-based resolver treats it as the floor.
    Unknown,
    // Integrated / low-power GPU: the lowest quality tier.
    Integrated,
    // Older or small discrete GPU, or an Apple base M-series: entry quality.
    EntryDiscrete,
    // Mainstream discrete GPU, or an Apple Pro: mid quality.
    MidDiscrete,
    // Enthusiast discrete GPU, or an Apple Max / Ultra: high quality.
    HighDiscrete,
}

// A coarse, Copy snapshot of the active GPU's class, queried from the backend
// once it is built (mirrors `DeviceCapabilities`). Read at init to choose
// sensible default graphics quality; never persisted, re-queried each launch so
// it is always correct for the current device + driver. The GPU *name* is
// deliberately omitted (it is not `Copy`); a backend exposes the name separately
// when a UI needs it.
#[derive(Clone, Copy, Debug)]
pub struct GpuProfile {
    pub vendor: GpuVendor,
    pub tier: GpuTier,
    // Dedicated VRAM on a discrete GPU, or the recommended working-set on a
    // unified-memory GPU. 0 when the backend / driver cannot report it.
    pub memory_budget_bytes: u64,
    pub unified_memory: bool,
    pub discrete: bool,
}

impl GpuProfile {
    // Conservative fallback for a backend that does not report a profile:
    // unknown hardware picks the cautious baseline, never a high preset. The
    // opposite default from `DeviceCapabilities::ALL` -- a feature gate fails
    // open (assume capable, no-op with a warning if not), but quality
    // auto-config fails safe (assume modest, never overdrive a weak GPU).
    pub const UNKNOWN: Self = Self {
        vendor: GpuVendor::Other,
        tier: GpuTier::Unknown,
        memory_budget_bytes: 0,
        unified_memory: false,
        discrete: false,
    };
}

impl Default for GpuProfile {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

// The cheap signals every backend can gather about its GPU, mapped to a coarse
// `GpuTier` by one shared rule so the three backends classify consistently and
// the mapping is unit-testable without a GPU. The backends differ in what they
// can report (Apple exposes a GPU family; DirectX / Vulkan expose a VRAM figure
// and a discrete / integrated flag), so this carries the union and the rule
// uses whichever signals are present.
pub struct GpuClassInput {
    pub vendor: GpuVendor,
    pub memory_budget_bytes: u64,
    pub discrete: bool,
    // Apple GPU family generation rank (7 = M1 .. 10 = M4), or 0 for a non-Apple
    // GPU. Apple silicon classifies by generation; everything else by VRAM.
    pub apple_family: u8,
}

// The Apple GPU family generation rank a device name implies, or 0 when the name
// is not an Apple silicon GPU. Metal reads the rank straight off the device
// (`MTLDevice::supportsFamily`); Vulkan has no equivalent query, so a MoltenVK
// build recovers it from the reported device name ("Apple M2 Max"). Without it
// Apple silicon falls through `classify_tier`'s integrated branch and the two
// backends disagree on the same GPU. `M<n>` maps to `n + 6`, matching Metal's
// `MTLGPUFamily::Apple7` = M1.
pub fn apple_family_from_device_name(name: &str) -> u8 {
    let Some(rest) = name.strip_prefix("Apple M") else {
        return 0;
    };
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    match digits.parse::<u8>() {
        Ok(n) if n >= 1 => n.saturating_add(6),
        _ => 0,
    }
}

// Map the gathered GPU signals to a coarse performance tier. Apple silicon is
// classified by GPU family generation (family alone cannot separate base from
// Pro / Max / Ultra within a generation -- a working-set refinement can split
// them later); a non-Apple integrated / low-power GPU is the lowest tier; a
// discrete GPU is bucketed by dedicated VRAM. An unreporting device (no memory,
// not discrete) stays `Unknown` so the resolver uses the conservative baseline.
pub fn classify_tier(input: &GpuClassInput) -> GpuTier {
    const GB: u64 = 1 << 30;
    // Apple silicon: classify by GPU family generation.
    if input.vendor == GpuVendor::Apple && input.apple_family >= 7 {
        return match input.apple_family {
            7 => GpuTier::EntryDiscrete, // M1 class
            8 => GpuTier::MidDiscrete,   // M2 class
            _ => GpuTier::HighDiscrete,  // M3 / M4 and newer
        };
    }
    // Any non-Apple integrated / low-power GPU is the lowest tier (Apple silicon
    // is unified too, but it returned above via its family branch).
    if !input.discrete {
        return GpuTier::Integrated;
    }
    // Discrete GPU: bucket by dedicated VRAM.
    match input.memory_budget_bytes {
        0 => GpuTier::Unknown,
        b if b >= 12 * GB => GpuTier::HighDiscrete,
        b if b >= 6 * GB => GpuTier::MidDiscrete,
        _ => GpuTier::EntryDiscrete,
    }
}

// The set of operations GraphicsSystem performs on a graphics backend.
// Implementations are thin forwarders to the inherent methods on
// MtlContext / DxContext / VkContext.
//
// The asset / world.jsonl hot-reload mutators below (`update_color_lut`,
// `rebuild_*_geometry`, `clone_static_draw_object`, etc.) are provided no-op
// methods driven only by the `cn debug` binary's reload passes, so they have
// no call site under `cargo check --lib`. Allow dead code at the trait level
// rather than annotating each; the required interface methods are never
// subject to the lint, so this only covers the binary-driven provided methods.
#[allow(dead_code)]
pub trait RenderBackend: SceneControl + Send {
    // Window / input lifecycle.
    fn window_closed(&mut self) -> bool;
    fn capture_cursor(&mut self);
    fn take_input(&mut self) -> RenderInput;
    fn wait_idle(&self);

    // Per-frame drive. See [`FrameParams`] for the inputs.
    fn draw_frame(&mut self, params: FrameParams<'_>) -> RenderResult<()>;
    fn update_view(&mut self, matrix: [[f32; 4]; 4]);
    fn update_model(&mut self, index: usize, model: [[f32; 4]; 4]);

    // Retire a draw object: hide it from every pass (main, shadow, velocity)
    // and exclude it from the ray-tracing acceleration structure, so a
    // despawned entity's slot leaves no ghost. The slot's geometry buffers are
    // untouched, but the slot index is returned to the draw-slot free list so
    // the next `clone_static_draw_object` can recycle it. A no-op if the index
    // is out of range.
    fn retire_draw_object(&mut self, draw_idx: usize);

    // Skinning. `vert_bytes` and `shadow_bytes` are Metal-only payloads;
    // DX/VK ignore them.
    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u16],
        draw_objects: Vec<SkinnedDrawObject>,
        vert_bytes: &[u8],
        frag_bytes: &[u8],
        shadow_bytes: &[u8],
    ) -> RenderResult<()>;
    fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]);

    // Attach morph-target data to the skinned draw objects, called once after
    // `upload_skinned`: `morphs[i]` belongs to draw object `i` (instance
    // copies share their template's data via the `Arc`). Default no-op for a
    // backend without a morph deformation path.
    fn upload_skinned_morphs(
        &mut self,
        _morphs: Vec<Option<std::sync::Arc<crate::mesh_payload::PayloadMorphs>>>,
    ) {
    }

    // Push a skinned object's current morph-target weights, sampled by the
    // animation system each frame. A no-op when the index is out of range or
    // the object carries no morph targets.
    fn update_morph_weights(&mut self, _skinned_index: usize, _weights: &[f32]) {}

    // Runtime skinned spawn (pre-reserved instance pool): a backend pre-reserves
    // hidden bind-pose copies at load (`SkinnedMesh.max_instances`) and reveals
    // one per skinned SpawnRequest. The default no-op implementations are a
    // fallback for a backend that has not wired runtime skinned spawn, where a
    // skinned SpawnRequest finds nothing to claim and is dropped.

    // Seed the backend's skinned instance pool from `(template_index,
    // instance_index)` pairs built at load, where each instance is a hidden
    // bind-pose copy of its template. Lets a later `spawn_skinned_instance`
    // claim a copy without growing any GPU buffer.
    fn seed_skinned_instance_pool(&mut self, _reservations: Vec<(usize, usize)>) {}

    // Claim a free pre-reserved copy of the skinned object at
    // `template_skinned_index`, reveal it at `model`, reset its pose to bind,
    // and return the claimed slot's skinned index. `None` when the template
    // reserved no pool or the pool is exhausted.
    fn spawn_skinned_instance(
        &mut self,
        _template_skinned_index: usize,
        _model: [[f32; 4]; 4],
    ) -> Option<usize> {
        None
    }

    // Hide a live skinned instance and return its slot to the pool so a later
    // spawn can claim it. A no-op if the index is out of range or was not a
    // pre-reserved instance slot.
    fn retire_skinned_draw_object(&mut self, _skinned_index: usize) {}

    // Push a skinned object's model-to-world matrix (it animates in place
    // unless something moves it). Cheap: the per-frame cull rebuild (and the
    // legacy skinned draw) reads the object's model directly, so this just
    // writes the field. A no-op if the index is out of range.
    fn update_skinned_model(&mut self, _skinned_index: usize, _model: [[f32; 4]; 4]) {}

    // Texture streaming. Albedo and normal maps share one handle-indexed pool,
    // so every streamed texture (whatever its role) flows through these. The
    // image carries its GPU format and mip chain: RGBA8 regenerates mips on
    // upload, block-compressed formats upload their chain verbatim.
    fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String>;
    fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &crate::build::texture::TextureImage,
    ) -> RenderResult<()>;

    // Mesh streaming.
    fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
    fn upload_mesh(
        &mut self,
        draw_idx: usize,
        verts: &[Vertex],
        idxs: &[u16],
        frame: u64,
    ) -> RenderResult<()>;

    // Seed the streamed-mesh sub-allocators with one reserved headroom block
    // (byte ranges in the shared vertex / index buffers) instead of the
    // per-mesh build-time regions. Used by the shrinkable-seed path: the
    // streamed geometry is no longer baked into the buffers at build time, so
    // the renderer hands the allocators one contiguous block sized to the
    // cap-many resident meshes rather than the whole streamed set. Implemented
    // on Metal + DirectX + Vulkan. Default no-op: a backend without the
    // shrinkable seed keeps freeing each mesh's build-time region in
    // `setup_mesh_streaming`.
    fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
    ) {
        let _ = (vtx_offset, vtx_bytes, idx_offset, idx_bytes);
    }

    // Voxel-world chunk streaming. `texture_slot` and `normal_map_slot`
    // are ignored by Metal (it binds chunk textures per draw).
    fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
        texture_slot: usize,
        normal_map_slot: usize,
    ) -> RenderResult<()>;
    fn add_chunk_mesh(&mut self, mesh: ChunkMesh<'_>) -> RenderResult<usize>;
    fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
    fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String>;

    // Device capability flags, queried from the GPU once the backend is built.
    // Read by GraphicsSystem to gray out + disable settings rows the device
    // cannot honor. Default: all capable, so a backend that does not report
    // capabilities keeps every toggle live (the feature then no-ops with a
    // warning on an incapable device, as before).
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::ALL
    }

    // Coarse GPU performance profile, queried once the backend is built. Read at
    // init to pick default graphics quality on first launch. Default: `UNKNOWN`
    // (the conservative tier), so a backend that does not report a profile never
    // makes the resolver auto-select a high preset.
    fn gpu_profile(&self) -> GpuProfile {
        GpuProfile::UNKNOWN
    }

    // The overlay coordinate space: the window's content size in logical,
    // DPI-independent units (points on macOS, client pixels on Windows, window
    // coordinates on Linux). Every backend reports the cursor in these same
    // units, so UI hit-testing, text layout, and the overlay shader's divide to
    // NDC all share one space regardless of the backing scale. A backend
    // converts to attachment pixels only where a pixel rect is unavoidable,
    // through `fullscreen::clip_rect_to_scissor`.
    //
    // Default `(0.0, 0.0)` for a headless backend with no window.
    fn logical_size(&self) -> (f32, f32) {
        (0.0, 0.0)
    }
    // Metal-only diagnostics; default no-op for parity.
    fn render_stats(&self) -> RenderStats {
        RenderStats::default()
    }

    // Show or hide the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // independent of camera capture. Edge-triggered by the backend, so calling
    // it every frame with the same value is cheap. Default no-op: a backend
    // without a free-mode cursor hide leaves the system cursor visible (DX /
    // Vulkan today).
    fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        let _ = hidden;
    }

    // Whether the real cursor has left the window, so an in-engine UI cursor
    // should stop drawing (windowed / borderless). The backend confines the
    // cursor to the active screen while in fullscreen, so it reports `false`
    // there. Default `false` (inside): backends without window-bounds tracking
    // (DX / Vulkan today) always draw the in-engine cursor.
    fn cursor_outside_window(&self) -> bool {
        false
    }

    // Tell the backend a togglable menu (a Screen toggled by an Escape KeyBinding)
    // coexists with a captured camera. In this mode Escape routes to the ECS
    // (so the menu shows/hides) instead of releasing the cursor inline, and a
    // click never recaptures the cursor (it fires a UI action). Set once at
    // setup. Default no-op: backends without dynamic capture (DX / Vulkan today)
    // keep the static behavior.
    fn set_menu_mode(&mut self, on: bool) {
        let _ = on;
    }

    // Drive cursor capture from the menu state each frame: capture for camera
    // control, release while a menu is open. Edge-triggered by the backend.
    // Default no-op (DX / Vulkan): they keep their startup capture decision.
    fn set_camera_capture(&mut self, capture: bool) {
        let _ = capture;
    }

    // Supply the reflection-probe placements (from declared `ReflectionProbe`
    // assets, or empty to auto-seed from the scene bounds). The backend bakes a
    // cube per placement and samples the nearest for the specular reflection.
    // Pushed once after construction. Default no-op: backends without probe
    // support (DX / Vulkan today) keep the sky reflection.
    fn set_reflection_probes(&mut self, probes: &[crate::reflection_probe::ProbePlacement]) {
        let _ = probes;
    }

    // Turn display sync (vsync) on or off at runtime, applied to presentation.
    // Edge-triggered by the backend, so calling it with the unchanged value is
    // cheap. Default no-op: a backend that only honors vsync at init ignores
    // runtime changes.
    fn set_vsync(&mut self, on: bool) {
        let _ = on;
    }

    // Switch the window between windowed / borderless / fullscreen at runtime.
    // The change flows through the backend's normal resize path (no GPU rebuild
    // beyond the resize it triggers). Default no-op for backends without a
    // window (embedded / preview) or that don't yet implement it.
    fn set_window_mode(&mut self, mode: crate::assets::WindowMode) {
        let _ = mode;
    }

    // Resize the window's content area at runtime (meaningful in windowed mode).
    // Drives the same resize path as a user-dragged resize. Default no-op for
    // backends without a window or that don't yet implement it.
    fn set_window_size(&mut self, width: u32, height: u32) {
        let _ = (width, height);
    }

    // The display modes (pixel resolution + refresh rate) the display this
    // backend renders to supports, unshaped (the caller dedups + sorts).
    // Default empty: a backend that cannot enumerate (or has no window) makes
    // the Resolution row fall back to the static preset list.
    fn display_modes(&self) -> Vec<crate::display_mode::DisplayMode> {
        Vec::new()
    }

    // The mode the display is currently running, if the backend can read it.
    // Shown by the Resolution row when the user has never chosen a mode (the
    // display keeps its desktop mode until one is chosen). Default `None`.
    fn current_display_mode(&self) -> Option<crate::display_mode::DisplayMode> {
        None
    }

    // Select the display mode to hold while the window is in fullscreen. The
    // backend applies it whenever the window is (or becomes) fullscreen and
    // restores the display's original mode when the window leaves fullscreen
    // or shuts down; outside fullscreen the choice is only remembered. Default
    // no-op: a backend without mode switching leaves the display alone.
    fn set_display_mode(&mut self, mode: crate::display_mode::DisplayMode) {
        let _ = mode;
    }

    // Replace the live post-process parameters (bloom / exposure / vignette /
    // LUT blend). These are pushed to the bloom + composite shaders each frame,
    // so a change takes effect on the next draw with no allocation or pipeline
    // rebuild. Default no-op: a backend that only reads the params at init
    // ignores runtime changes (DirectX / Vulkan today).
    fn update_post_process(&mut self, params: PostProcessParams) {
        let _ = params;
    }

    // Set the live ambient (IBL) light scale. Unlike the post-process params
    // above, `ambient_intensity` lives in the shared `LightUniforms` (uploaded
    // each frame by the main lighting pass), so it takes its own setter rather
    // than `update_post_process`. Default no-op: only Metal mutates it live
    // today; DirectX / Vulkan keep the init-time value (they read it at init).
    fn set_ambient_intensity(&mut self, value: f32) {
        let _ = value;
    }

    // Push the gameplay movement key map. The backend resolves each canonical
    // `Key` to its native key code and decodes physical key events through the
    // map (instead of hardcoded keys), so a settings-menu rebind takes effect on
    // the next key event. Pushed once after the backend is built and again on
    // each rebind. Default no-op: a backend without keymap decode keeps its
    // built-in defaults.
    fn set_keymap(&mut self, keymap: &KeyMap) {
        let _ = keymap;
    }

    // Apply a change to the quality-feature toggles (TAA / SSAO / SSR / RT
    // reflections / SSGI / auto-exposure) live. Unlike the post-process params,
    // these gate render passes whose GPU resources (pipelines, render targets,
    // ray-tracing acceleration structures) are built once at init, so applying a
    // change rebuilds the affected resources in place rather than flipping a
    // uniform. Default no-op: a backend that only reads these at init ignores
    // runtime changes (DirectX / Vulkan today), so the choice persists and takes
    // effect at the next launch there.
    fn apply_quality_settings(&mut self, settings: QualitySettings) {
        let _ = settings;
    }

    // Set the shadow cascade re-render cadence live. The cascade scheduler reads
    // the policy at the start of each shadow pass, so a change takes effect on the
    // next draw with no pipeline rebuild or allocation (unlike the shadow map
    // resolution, which is sized once at init). Default no-op: a backend that only
    // reads the cadence at init keeps the init-time value (DirectX / Vulkan
    // today), so the choice persists and takes effect at the next launch there.
    fn set_shadow_update(&mut self, update: crate::assets::ShadowUpdate) {
        let _ = update;
    }

    // Set the shadow distance (world units the cascades cover, capped at the
    // camera far plane) live. The per-frame cascade-split computation reads it
    // each draw, so a change takes effect on the next frame with no allocation or
    // rebuild (it sizes no GPU resource, unlike the shadow map resolution).
    // Default no-op: a backend that only reads the distance at init keeps the
    // init-time value (DirectX / Vulkan today), so the choice persists and takes
    // effect at the next launch there.
    fn set_shadow_distance(&mut self, distance: u32) {
        let _ = distance;
    }

    // Set the live shadow cascade count (1..=4). The cascade-split math + the
    // re-render schedule read it each frame and only the first `count` cascades
    // are projected, rendered, and sampled (the array capacity stays 4), so a
    // change takes effect on the next frame with no resize or rebuild. Default
    // no-op: a backend that only reads the count at init keeps the init-time
    // value (DirectX / Vulkan today), so the choice persists and takes effect at
    // the next launch there.
    fn set_shadow_cascades(&mut self, count: u32) {
        let _ = count;
    }

    // Update the live scalar sub-tunables of the SSAO / SSR / SSGI / auto-exposure
    // passes (radius, intensity, distance, EV bounds, adaptation speed). Unlike
    // `apply_quality_settings`, this rebuilds nothing: each backend re-reads these
    // values from its stored `*Settings` structs into a per-frame uniform every
    // draw, so mutating them takes effect on the next frame with no pipeline /
    // target rebuild and no TAA-history reset. Only the fields of a feature that is
    // currently on are honoured (its settings are present); a value for an off
    // feature is ignored here and applies when the feature next turns on. The
    // structural sub-knobs (gather resolution, ray / step counts) are NOT live and
    // still ride `apply_quality_settings`. Default no-op: a backend that reads
    // these only at init keeps the init-time values (DirectX / Vulkan today), so
    // the choice persists and takes effect at the next launch there.
    fn update_quality_params(&mut self, settings: QualitySettings) {
        let _ = settings;
    }

    // Shared atomic flag the backend polls at frame start to trigger a
    // shader rebuild. `Some` only under `cn debug` on backends that ship
    // hot-reload (Metal today); `None` on production runs and on backends
    // that have not implemented hot-reload yet. The debug server reads this
    // to forward `reload-shaders` commands; the filesystem watcher writes
    // it directly. Default: `None`.
    fn shader_reload_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        None
    }

    // Replace the live colour-grading LUT with a fresh `size³` RGBA8 payload.
    // Driven by asset hot-reload (`cn debug` only). Default no-op: backends
    // that have not implemented the swap leave the LUT bound at whatever
    // payload was uploaded at init.
    fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        let _ = (size, data);
        Ok(())
    }

    // `(vertex_count, index_count)` for the static draw at `draw_idx`, or
    // `None` when the index is out of range / the backend does not expose
    // the field. Used by asset hot-reload to detect size-changing
    // reloads before attempting [`Self::update_mesh_geometry`], which
    // rejects size mismatches. Default returns `None`; backends that
    // implement the rebuild path also override this.
    fn draw_geometry_size(&self, draw_idx: usize) -> Option<(usize, usize)> {
        let _ = draw_idx;
        None
    }

    // Per-LOD-alternate index counts for the static draw at `draw_idx`,
    // ordered from LOD1 upward (LOD0 is reported by
    // [`Self::draw_geometry_size`]). Returns `None` when the index is out of
    // range or the backend does not expose its LOD layout. Used by asset
    // hot-reload alongside [`Self::draw_geometry_size`] to detect
    // size-changing reloads: a `.glb` that re-exports with a different LOD
    // breakdown queues the entry for [`Self::rebuild_static_geometry`]
    // instead of [`Self::update_mesh_geometry`]'s in-place write.
    fn draw_lod_index_counts(&self, draw_idx: usize) -> Option<Vec<usize>> {
        let _ = draw_idx;
        None
    }

    // Rebuild the shared static-mesh vertex + index buffers, replacing the
    // geometry of each `DrawGeometryUpdate.draw_idx` with the new
    // vertices / indices / LOD alternates. Draws not named in `changes`
    // keep their current geometry, copied byte-for-byte from the live
    // buffers. The slot's `vertex_count`, `index_count`, and
    // `lod_alternates` index offsets are rewritten as the new buffers are
    // laid out. Driven by asset hot-reload (`cn debug` only) when a
    // size-changing `.glb` re-export means the existing
    // [`Self::update_mesh_geometry`] in-place write no longer fits.
    // `wait_idle` first; the rebuild swaps the GPU buffers wholesale.
    // Default no-op: backends that have not implemented the rebuild
    // return `Ok(())` and the size-changing reload is logged + skipped at
    // the caller (the existing in-place path already errored on size
    // mismatch).
    fn rebuild_static_geometry(&mut self, changes: Vec<DrawGeometryUpdate>) -> RenderResult<()> {
        let _ = changes;
        Ok(())
    }

    // Replace a `SkinnedMesh` draw slot's vertex + index data in place.
    // Driven by asset hot-reload (`cn debug` only). Reuses the slot's
    // existing vertex region + index region in the shared skinned vertex /
    // index buffers (created once by [`Self::upload_skinned`]), so the new
    // geometry must match the slot's init-time vertex count + index count
    // and the new skeleton must keep the same joint count; pipelines stay
    // untouched, only the bytes change. `vertex_base` is the init-time
    // vertex offset (in vertex units) into the shared buffer; indices are
    // rebased onto it before writing. Default no-op.
    fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: usize,
        vertex_base: u16,
        verts: &[SkinnedVertex],
        idxs: &[u16],
    ) -> Result<(), String> {
        let _ = (skinned_index, vertex_base, verts, idxs);
        Ok(())
    }

    // Rebuild the shared skinned-mesh vertex + index buffers, replacing the
    // geometry of each `SkinnedDrawGeometryUpdate.skinned_index` with the
    // new vertices / indices. Slots not named in `changes` keep their
    // current geometry, copied byte-for-byte from the live buffers and
    // re-based onto the new vertex region they land in. Returns the
    // post-rebuild layout (one [`SkinnedSlotLayout`] per slot, in
    // `skinned_index` order) so the caller can refresh its source-map
    // `vertex_base` / `vertex_count` / `index_count` to point at the new
    // regions. Driven by asset hot-reload (`cn debug` only) when a
    // size-changing `.glb` re-export means the existing
    // [`Self::update_skinned_mesh_geometry`] in-place write no longer fits.
    // The backend `wait_idle`s first; the rebuild swaps the GPU buffers
    // wholesale. The skinned pipelines, shadow + velocity + SSAO + SSR
    // variants, and `skinned_draw_objects` slot metadata
    // (`texture_slot` / `normal_map_slot` / `material` / `joint_count`)
    // all stay untouched; only the `index_offset` / `index_count` on each
    // `SkinnedDrawObject` (and the buffers themselves) move. Default no-op
    // (returns an empty layout vec): backends that have not implemented
    // the rebuild leave the size-changing reload as logged + skipped at
    // the caller, the same behaviour as before, since the in-place path
    // already errored on size mismatch.
    fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<SkinnedDrawGeometryUpdate>,
    ) -> Result<Vec<SkinnedSlotLayout>, String> {
        let _ = changes;
        Ok(Vec::new())
    }

    // Update a skinned slot's joint count and resize the backend's per-slot
    // joint-matrix buffers to match. Driven by asset hot-reload (`cn debug`
    // only) when a re-imported `.glb`'s skeleton has a different joint
    // count than the slot was initialised with. Shrinking truncates the
    // per-slot Vec; growing seeds the new entries to identity so the slot
    // renders undeformed on the next `update_skinned_pose`. The skinned
    // shaders consume the joints buffer through a pointer (not a fixed-
    // size array) and use vertex-attribute-encoded joint indices, so no
    // pipeline or shader rebuild is required for a joint-count change;
    // only the CPU-side per-slot buffer and `SkinnedDrawObject.joint_count`
    // change. Default no-op: backends that have not implemented the resize
    // leave the skeleton-shape change logged + skipped at the caller.
    fn update_skinned_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        let _ = (skinned_index, new_joint_count);
        Ok(())
    }

    // Replace a `Mesh` draw slot's vertex + index data in place. Driven by
    // asset hot-reload (`cn debug` only). Reuses the slot's existing offset
    // in the shared vertex / index buffers, so the new geometry must match
    // the slot's init-time vertex count + index count; a size-changing
    // reload returns an error so the caller can queue
    // [`Self::rebuild_static_geometry`] instead, which repacks the shared
    // buffers. Each entry in
    // `lod_alternates` (`(switch_distance, mesh-relative indices)`) is
    // written to the matching slot's pre-allocated LOD index region; the
    // number of LODs and each LOD's index count must match the slot's
    // init-time layout, otherwise the call returns an error so the caller
    // can queue [`Self::rebuild_static_geometry`]. `switch_distance` is
    // re-stored per LOD so a JSON-side tweak to `lod_distances` propagates
    // without a process restart. Default no-op.
    fn update_mesh_geometry(
        &mut self,
        draw_idx: usize,
        verts: &[Vertex],
        idxs: &[u16],
        lod_alternates: &[(f32, Vec<u16>)],
    ) -> Result<(), String> {
        let _ = (draw_idx, verts, idxs, lod_alternates);
        Ok(())
    }

    // Replace the live IBL environment map with a freshly precomputed payload.
    // `payload` is the serialised byte format emitted by
    // [`crate::build::environment_map::compile_environment_map_payload`]
    // (header + irradiance cube + prefilter mip chain), so init and hot-reload
    // share a single byte format. Driven by asset hot-reload (`cn debug`
    // only). Default no-op: backends that have not implemented the swap leave
    // the IBL cubes bound at whatever payload was uploaded at init.
    fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()> {
        let _ = payload;
        Ok(())
    }

    // Replace the live volumetric-fog settings, or disable the fog pass when
    // `None`. Driven by world.jsonl hot-reload (`cn debug` only). Default
    // no-op: backends that have not implemented the swap leave the fog pass
    // at whatever settings were resolved at init.
    //
    // A backend that built its fog pipeline lazily based on the world's
    // init-time `VolumetricFog` cannot enable the pass via this call when
    // the world started with no fog declared; re-enabling fog on a world
    // that did not declare it at startup requires a relaunch.
    fn update_fog_settings(&mut self, settings: Option<FogSettings>) {
        let _ = settings;
    }

    // Capture the last presented frame to a PNG at `path` and return the saved
    // path. Driven by the `cn debug` WS `screenshot` command for headless
    // on-GPU render verification. Default `Err`: a backend without a capture
    // path reports it unsupported (all current backends override this).
    fn screenshot(&mut self, path: &str) -> Result<String, String> {
        let _ = path;
        Err("screenshot capture not supported on this backend".to_string())
    }

    // Instantiate a runtime copy of an existing draw object at a new transform:
    // re-use the source slot's geometry region (`vertex_offset` / `vertex_count`
    // / `index_offset` / `index_count` / `base_vertex` / `lod_alternates`) and
    // copy its texture slots, material, and cull distance, swapping only the
    // model matrix. The new slot reuses one freed by `retire_draw_object` before
    // growing the draw-object vec. Returned index is the new `DrawObject` slot.
    // Driven by runtime entity spawn (`SpawnRequest`). The copy is non-cullable
    // (sentinel AABB) and drawn every frame, since the init-time BVH cannot
    // refit to admit a slot added at runtime; moving copies (the common case)
    // opt out of the static BVH exactly like streamed chunks and held items.
    // Default no-op (returns `Err`): backends without an implementation leave
    // the spawn path logged + skipped at the caller.
    fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        model: [[f32; 4]; 4],
    ) -> Result<usize, String> {
        let _ = (src_draw_idx, model);
        Err("clone_static_draw_object: not implemented on this backend".to_string())
    }

    // Rewrite a draw slot's material parameters + texture/normal-map pool
    // indices in place. Driven by `world.jsonl` hot-reload (`cn debug` only)
    // when a Prop edits its `material` / `texture` arg. Default no-op: the
    // caller logs the change as skipped on backends without an implementation.
    fn set_draw_material(
        &mut self,
        draw_idx: usize,
        material: MaterialUniforms,
        texture_slot: usize,
        normal_map_slot: usize,
    ) {
        let _ = (draw_idx, material, texture_slot, normal_map_slot);
    }

    // Rewrite a draw slot's `cull_distance` in place. Driven by `world.jsonl`
    // hot-reload (`cn debug` only) when a Prop edits its `cull_distance` arg.
    // Default no-op.
    fn set_draw_cull_distance(&mut self, draw_idx: usize, cull_distance: f32) {
        let _ = (draw_idx, cull_distance);
    }

    // Append a projected-decal record at runtime, returning a stable slot
    // index the caller hands to [`Self::remove_decal`] later. Lets a
    // gameplay system stamp bullet holes, footprints, or other ad-hoc
    // decals after the world has built. Backends that have not implemented
    // the runtime path return `Err`; the caller logs and drops the request.
    fn add_decal(&mut self, record: crate::decal::DecalRecord) -> Result<usize, String> {
        let _ = record;
        Err("add_decal: not implemented on this backend".to_string())
    }

    // Tombstone a runtime decal slot. The id returned by
    // [`Self::add_decal`] becomes invalid; the next add may reuse it.
    // Default no-op-with-Err: backends without a runtime path leave the
    // remove logged + skipped at the caller.
    fn remove_decal(&mut self, decal_id: usize) -> Result<(), String> {
        let _ = decal_id;
        Err("remove_decal: not implemented on this backend".to_string())
    }

    // Append a particle-emitter record at runtime, returning a stable slot
    // index. The backend allocates the per-emitter GPU pool + atomic
    // spawn counter (matching the init-time path) so the compute kernel
    // can begin ticking on the next frame. Default no-op-with-Err.
    fn add_emitter(
        &mut self,
        record: crate::particles::ParticleEmitterRecord,
    ) -> Result<usize, String> {
        let _ = record;
        Err("add_emitter: not implemented on this backend".to_string())
    }

    // Tombstone a runtime emitter slot and release its GPU pool +
    // counter buffers (the GPU keeps them alive via its own refcount
    // until any in-flight command buffer that referenced them completes).
    // Default no-op-with-Err.
    fn remove_emitter(&mut self, emitter_id: usize) -> Result<(), String> {
        let _ = emitter_id;
        Err("remove_emitter: not implemented on this backend".to_string())
    }

    // Rebuild the live main / instanced / shadow render pipelines from
    // freshly compiled world-loaded shader stage bytes. Driven by asset
    // hot-reload (`cn debug` only) when one of the captured `Shader`
    // source files is saved or a debug-WS `reload-assets` command fires.
    // Each `Some(bytes)` replaces the matching live pipeline (and any
    // dependent state: bindless-texture argument encoder, cull pipeline,
    // instanced variant, shadow variant); `None` leaves the pipeline
    // untouched (e.g. a world without an instanced shader passes `None`
    // for the instanced slot). The backend should build every replacement
    // into a temporary first and only swap when every build succeeds;
    // mirrors the safety pattern in the Metal backend's `hot_reload` so a
    // compile error never overwrites a live pipeline with a half-built
    // replacement. Default no-op (returns `Err`): backends without an
    // implementation leave the world-loaded shader reload logged + skipped
    // at the caller.
    //
    // Skinned-mesh variants are out of scope here: their pipelines depend
    // on the world's `SkinnedMesh`-injected library bytes that
    // [`Self::upload_skinned`] consumes and drops.
    fn update_world_shader_pipelines(
        &mut self,
        vert_bytes: Option<&[u8]>,
        frag_bytes: Option<&[u8]>,
        shadow_bytes: Option<&[u8]>,
        vert_instanced_bytes: Option<&[u8]>,
    ) -> Result<(), String> {
        let _ = (vert_bytes, frag_bytes, shadow_bytes, vert_instanced_bytes);
        Err("update_world_shader_pipelines: not implemented on this backend".to_string())
    }

    // Build the render pipeline for one shader bucket from its compiled stage
    // bytes, making draws that carry that bucket renderable. Called by the
    // streaming pump when a scene that exclusively owns the bucket's `Shader`
    // pins: init skipped the build, so this is where the cost lands (behind
    // the loading screen, since the bucket counts as scene-resident content).
    // Bucket 0 is the world default program and is never installed this way.
    //
    // Default no-op-with-Ok: a backend that renders every draw with the world
    // default program has no per-bucket pipeline to build, and the bucket is
    // resident as far as scene loading is concerned.
    fn install_world_shader(&mut self, bucket: u32, shader: ShaderBytes<'_>) -> RenderResult<()> {
        let _ = (bucket, shader);
        Ok(())
    }

    // Release one shader bucket's render pipeline, undoing
    // [`Self::install_world_shader`]. Called when the owning scene unpins;
    // draws carrying the bucket stop rendering until it is installed again.
    // Default no-op, for the same reason as above.
    fn evict_world_shader(&mut self, bucket: u32) {
        let _ = bucket;
    }

    // The swapchain-level configuration this live backend can hot-swap a world
    // onto, or `None` when the backend cannot reload a world in place (it must
    // be fully rebuilt instead). Read by GraphicsSystem when a transplanted
    // backend is handed a new world (the `cn editor` live SAVE): the swap reuses
    // the backend via [`Self::reload_world`] only when this equals the new
    // world's `BackendInit::swapchain_config`; a `None` or a mismatch routes to a
    // full rebuild (recreating the window). Default `None`: DirectX / Vulkan
    // (and any backend without a real `reload_world`) always rebuild.
    fn hot_swap_config(&self) -> Option<SwapchainConfig> {
        None
    }

    // Re-upload a new world's GPU content onto this already-constructed backend,
    // reusing the live device + window + swapchain instead of building a new one.
    // Driven by the `cn editor` live SAVE: after a structural edit recompiles the
    // blobs, GraphicsSystem transplants the running backend into the rebuilt
    // world and calls this so the edit applies without recreating the OS window
    // or re-initialising the GPU device. The backend waits for the GPU to idle,
    // drops the old world's content resources, and rebuilds them from `init` on
    // the retained hardware. Only ever called when [`Self::hot_swap_config`]
    // reported a config matching `init.swapchain_config()`, so the swapchain
    // (pixel format / frames-in-flight / EDR) is guaranteed unchanged. Default
    // `Err`/unsupported: DirectX / Vulkan fall back to a full rebuild (no
    // regression; a real implementation is Windows-pending like the rest).
    fn reload_world(&mut self, init: BackendInit<'_>) -> RenderResult<()> {
        let _ = init;
        Err(RenderError::Other(
            "reload_world: not supported on this backend".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn input(
        vendor: GpuVendor,
        memory_budget_bytes: u64,
        discrete: bool,
        apple_family: u8,
    ) -> GpuClassInput {
        GpuClassInput {
            vendor,
            memory_budget_bytes,
            discrete,
            apple_family,
        }
    }

    #[test]
    fn unknown_profile_is_the_conservative_default() {
        // The opposite default from capabilities: quality auto-config fails safe.
        let p = GpuProfile::default();
        assert_eq!(p.tier, GpuTier::Unknown);
        assert_eq!(p.vendor, GpuVendor::Other);
        assert_eq!(p.memory_budget_bytes, 0);
        // Unknown sorts below every real tier, so a `>=` resolver treats it as
        // the floor.
        assert!(GpuTier::Unknown < GpuTier::Integrated);
        assert!(GpuTier::Integrated < GpuTier::EntryDiscrete);
        assert!(GpuTier::EntryDiscrete < GpuTier::MidDiscrete);
        assert!(GpuTier::MidDiscrete < GpuTier::HighDiscrete);
    }

    #[test]
    fn apple_family_reads_the_generation_out_of_the_device_name() {
        // The names MoltenVK reports, mapped onto Metal's family ranks.
        assert_eq!(apple_family_from_device_name("Apple M1"), 7);
        assert_eq!(apple_family_from_device_name("Apple M2 Max"), 8);
        assert_eq!(apple_family_from_device_name("Apple M3 Pro"), 9);
        assert_eq!(apple_family_from_device_name("Apple M4 Ultra"), 10);
        // A generation past what Metal's SDK names yet still ranks above M3, so
        // a newer Mac is not demoted.
        assert!(apple_family_from_device_name("Apple M9") > 9);
    }

    #[test]
    fn non_apple_device_names_report_no_family() {
        for name in [
            "NVIDIA GeForce RTX 4090",
            "AMD Radeon RX 7900 XTX",
            "Intel(R) Arc(tm) A770",
            // Apple's own non-M naming, and a truncated / malformed report.
            "Apple A17 Pro",
            "Apple M",
            "Apple MX",
            "",
        ] {
            assert_eq!(apple_family_from_device_name(name), 0, "{name}");
        }
    }

    #[test]
    fn apple_family_from_a_name_reaches_the_same_tier_metal_does() {
        // The whole point of the name probe: a MoltenVK build must land on the
        // tier the Metal backend reports for the same silicon, not on the
        // integrated floor a zero family falls through to.
        let family = apple_family_from_device_name("Apple M2 Max");
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 32 * GB, false, family)),
            GpuTier::MidDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 32 * GB, false, 0)),
            GpuTier::Integrated
        );
    }

    #[test]
    fn apple_silicon_classifies_by_generation() {
        // Unified memory is large on Apple silicon, but the family generation
        // (not the working-set) decides the tier, so the huge shared budget does
        // not read as a high-VRAM discrete card.
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 16 * GB, false, 7)),
            GpuTier::EntryDiscrete // M1
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 24 * GB, false, 8)),
            GpuTier::MidDiscrete // M2
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 48 * GB, false, 9)),
            GpuTier::HighDiscrete // M3
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 64 * GB, false, 10)),
            GpuTier::HighDiscrete // M4 and newer cap at high
        );
    }

    #[test]
    fn discrete_gpu_classifies_by_vram() {
        // An Intel-Mac AMD dGPU or a PC discrete card: vendor is not Apple and
        // there is no Apple family, so VRAM buckets the tier.
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 24 * GB, true, 0)),
            GpuTier::HighDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Amd, 8 * GB, true, 0)),
            GpuTier::MidDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 4 * GB, true, 0)),
            GpuTier::EntryDiscrete
        );
        // A discrete card that reports no memory budget is left Unknown rather
        // than guessed high.
        assert_eq!(
            classify_tier(&input(GpuVendor::Amd, 0, true, 0)),
            GpuTier::Unknown
        );
    }

    #[test]
    fn integrated_gpu_is_the_lowest_tier() {
        // Non-Apple integrated part: no dedicated memory, not unified, no Apple
        // family.
        assert_eq!(
            classify_tier(&input(GpuVendor::Intel, 0, false, 0)),
            GpuTier::Integrated
        );
    }

    #[test]
    fn vram_bucket_boundaries() {
        // Boundaries are inclusive lower bounds (>= 12 GB high, >= 6 GB mid).
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 12 * GB, true, 0)),
            GpuTier::HighDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 12 * GB - 1, true, 0)),
            GpuTier::MidDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 6 * GB, true, 0)),
            GpuTier::MidDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 6 * GB - 1, true, 0)),
            GpuTier::EntryDiscrete
        );
    }

    // A do-nothing backend used to exercise the trait's provided (default)
    // method bodies without a GPU. It supplies the smallest valid bodies for
    // the required methods and overrides none of the defaults, so calling a
    // defaulted method runs the trait's own body.
    struct StubBackend;

    impl SceneControl for StubBackend {
        fn update_visibility(&mut self, _draw_idx: usize, _visible: bool) {}
        fn set_fade(&mut self, _fade: f32) {}
    }

    impl RenderBackend for StubBackend {
        fn window_closed(&mut self) -> bool {
            false
        }
        fn capture_cursor(&mut self) {}
        fn take_input(&mut self) -> RenderInput {
            RenderInput::default()
        }
        fn wait_idle(&self) {}
        fn draw_frame(&mut self, _params: FrameParams<'_>) -> RenderResult<()> {
            Ok(())
        }
        fn update_view(&mut self, _matrix: [[f32; 4]; 4]) {}
        fn update_model(&mut self, _index: usize, _model: [[f32; 4]; 4]) {}
        fn retire_draw_object(&mut self, _draw_idx: usize) {}
        fn upload_skinned(
            &mut self,
            _vertices: &[SkinnedVertex],
            _indices: &[u16],
            _draw_objects: Vec<SkinnedDrawObject>,
            _vert_bytes: &[u8],
            _frag_bytes: &[u8],
            _shadow_bytes: &[u8],
        ) -> RenderResult<()> {
            Ok(())
        }
        fn update_skinned_pose(&mut self, _skinned_index: usize, _matrices: &[[[f32; 4]; 4]]) {}
        fn evict_texture_slot(&mut self, _slot: usize) -> Result<(), String> {
            Ok(())
        }
        fn update_texture_slot(
            &mut self,
            _slot: usize,
            _image: &crate::build::texture::TextureImage,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn evict_mesh(&mut self, _draw_idx: usize, _retire_frame: u64) -> Result<(), String> {
            Ok(())
        }
        fn upload_mesh(
            &mut self,
            _draw_idx: usize,
            _verts: &[Vertex],
            _idxs: &[u16],
            _frame: u64,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn setup_chunk_streaming(
            &mut self,
            _chunk_vtx_bytes: usize,
            _chunk_idx_bytes: usize,
            _texture_slot: usize,
            _normal_map_slot: usize,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn add_chunk_mesh(&mut self, _mesh: ChunkMesh<'_>) -> RenderResult<usize> {
            Ok(0)
        }
        fn remove_chunk_mesh(
            &mut self,
            _draw_idx: usize,
            _retire_frame: u64,
        ) -> Result<(), String> {
            Ok(())
        }
        fn set_chunk_model(
            &mut self,
            _draw_idx: usize,
            _model: [[f32; 4]; 4],
        ) -> Result<(), String> {
            Ok(())
        }
    }

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Minimal QualitySettings with every feature off, so no *Settings sub-type
    // needs constructing.
    fn stub_quality() -> QualitySettings {
        QualitySettings {
            taa: false,
            ssao: None,
            ssr: None,
            rt_reflections: None,
            ssgi: None,
            reflection_blur_scale: 1,
            auto_exposure: None,
            auto_exposure_bias_ev: 0.0,
        }
    }

    #[test]
    fn default_query_methods_report_conservative_values() {
        let backend = StubBackend;
        // Capabilities fail open: a backend that does not report keeps every
        // toggle live.
        assert!(backend.capabilities().ray_tracing);
        // Quality auto-config fails safe: the unknown/conservative profile.
        assert_eq!(backend.gpu_profile().tier, GpuTier::Unknown);
        assert_eq!(backend.gpu_profile().vendor, GpuVendor::Other);
        assert_eq!(backend.gpu_profile().memory_budget_bytes, 0);
        // Metal-only diagnostics: zeroed no-op defaults elsewhere.
        assert_eq!(backend.logical_size(), (0.0, 0.0));
        assert_eq!(backend.render_stats(), RenderStats::default());
        // No window-bounds tracking: the in-engine cursor always draws.
        assert!(!backend.cursor_outside_window());
        // No display enumeration and no hot-reload flag wired.
        assert!(backend.display_modes().is_empty());
        assert!(backend.current_display_mode().is_none());
        assert!(backend.shader_reload_flag().is_none());
        // Not hot-swap-capable: a live world reload routes to a full rebuild.
        assert!(backend.hot_swap_config().is_none());
        // No geometry-size introspection for the reload size check.
        assert!(backend.draw_geometry_size(0).is_none());
        assert!(backend.draw_lod_index_counts(0).is_none());
    }

    #[test]
    fn default_mutators_are_noops_and_fallible_hooks_report_defaults() {
        let mut backend = StubBackend;

        // Runtime skinned-spawn fallbacks: no pool, nothing to claim.
        backend.seed_skinned_instance_pool(vec![]);
        assert!(backend.spawn_skinned_instance(0, IDENTITY).is_none());
        backend.retire_skinned_draw_object(0);
        backend.update_skinned_model(0, IDENTITY);

        // Streaming + cursor + capture no-ops.
        backend.seed_mesh_streaming(0, 0, 0, 0);
        backend.set_ui_cursor_hidden(true);
        backend.set_menu_mode(true);
        backend.set_camera_capture(true);
        backend.set_reflection_probes(&[]);

        // Presentation + window no-ops.
        backend.set_vsync(true);
        backend.set_window_mode(crate::assets::WindowMode::Fullscreen);
        backend.set_window_size(1280, 720);
        backend.set_display_mode(crate::display_mode::DisplayMode {
            width: 1920,
            height: 1080,
            refresh_hz: 60,
        });

        // Live look + input tunable no-ops.
        backend.update_post_process(PostProcessParams::DEFAULT);
        backend.set_ambient_intensity(1.0);
        backend.set_keymap(&KeyMap::default());
        backend.apply_quality_settings(stub_quality());
        backend.update_quality_params(stub_quality());
        backend.set_shadow_update(crate::assets::ShadowUpdate::EveryFrame);
        backend.set_shadow_distance(200);
        backend.set_shadow_cascades(3);
        backend.update_fog_settings(None);
        backend.set_draw_material(0, MaterialUniforms::DEFAULT, 0, 0);
        backend.set_draw_cull_distance(0, 50.0);

        // Fallible hot-reload hooks that succeed by default (no-op Ok).
        assert!(backend.update_color_lut(2, &[0u8; 32]).is_ok());
        assert!(backend.rebuild_static_geometry(vec![]).is_ok());
        assert!(backend.update_skinned_mesh_geometry(0, 0, &[], &[]).is_ok());
        assert!(backend.rebuild_skinned_geometry(vec![]).unwrap().is_empty());
        assert!(backend.update_skinned_skeleton(0, 0).is_ok());
        assert!(backend.update_mesh_geometry(0, &[], &[], &[]).is_ok());
        assert!(backend.update_environment_map(&[]).is_ok());

        // Fallible hooks a bare backend does not implement: they report Err.
        assert!(backend.screenshot("unused.png").is_err());
        assert!(backend.clone_static_draw_object(0, IDENTITY).is_err());
        assert!(backend.add_decal(stub_decal()).is_err());
        assert!(backend.remove_decal(0).is_err());
        assert!(backend.add_emitter(stub_emitter()).is_err());
        assert!(backend.remove_emitter(0).is_err());
        assert!(
            backend
                .update_world_shader_pipelines(None, None, None, None)
                .is_err()
        );
    }

    // A minimal empty-world BackendInit borrowing `window`, for exercising the
    // default `reload_world`. Empty slices are `'static`; the only real borrow
    // is the window args.
    fn empty_backend_init(window: &crate::assets::WindowArgs) -> BackendInit<'_> {
        use crate::backend_init::{
            MediaPayloads, PostSettings, SceneData, ShaderBytes, ShadowParams, WorldFx,
        };
        BackendInit {
            window,
            validation: false,
            frames_in_flight: 2,
            vsync: false,
            clear_color: [0.0; 4],
            hot_reload: false,
            scene: SceneData {
                vertices: &[],
                indices: &[],
                draw_objects: Vec::new(),
                instanced_clusters: Vec::new(),
                n_skinned: 0,
                n_chunk_max: 0,
            },
            shaders: vec![ShaderBytes {
                vert: &[],
                frag: &[],
                shadow: &[],
                vert_instanced: &[],
                deferred: false,
            }],
            media: MediaPayloads {
                textures: &[],
                text_atlases: Vec::new(),
                env_map_bytes: None,
                color_lut_bytes: None,
            },
            light_uniforms: crate::render_types::LightUniforms::DEFAULT,
            local_lights: Vec::new(),
            spot_shadows: Vec::new(),
            area_lights: Vec::new(),
            shadows: ShadowParams {
                map_size: 0,
                update: crate::assets::ShadowUpdate::default(),
                distance: 0,
                cascades: 1,
            },
            anisotropy: 1,
            planar_planes: 0,
            post: PostSettings {
                post_process: PostProcessParams::DEFAULT,
                taa_enabled: false,
                ssao: None,
                ssr: None,
                ssgi: None,
                rt_reflections: None,
                reflection_blur_scale: 1,
                auto_exposure: None,
                auto_exposure_bias_ev: 0.0,
                hdr_display: false,
                hdr_pq: false,
                temporal_upscaling: false,
                upscale_scale: 1.0,
                upscale_backend: crate::assets::UpscalerBackend::Auto,
                occlusion_two_pass: false,
            },
            fx: WorldFx {
                decals: Vec::new(),
                particles: Vec::new(),
                fog: None,
                water_surfaces: Vec::new(),
                glass_panels: Vec::new(),
                sdf_volumes: Vec::new(),
            },
            requirements: Default::default(),
        }
    }

    #[test]
    fn default_reload_world_is_unsupported() {
        // A backend without a real reload path reports the swap unsupported, so
        // the caller falls back to a full rebuild.
        let mut backend = StubBackend;
        let window = crate::assets::WindowArgs::default();
        assert!(backend.reload_world(empty_backend_init(&window)).is_err());
    }

    fn stub_decal() -> crate::decal::DecalRecord {
        crate::decal::DecalRecord {
            model: IDENTITY,
            inv_model: IDENTITY,
            texture_slot: 0,
            tint: [1.0; 4],
        }
    }

    fn stub_emitter() -> crate::particles::ParticleEmitterRecord {
        crate::particles::ParticleEmitterRecord {
            texture_slot: 0,
            position: [0.0; 3],
            direction: [0.0, 1.0, 0.0],
            spread_cos: 1.0,
            speed_min: 0.0,
            speed_max: 1.0,
            lifetime_min: 0.0,
            lifetime_max: 1.0,
            gravity: [0.0, -9.8, 0.0],
            spawn_rate: 1.0,
            max_particles: 1,
            size_start: 1.0,
            size_end: 1.0,
            color_start: [1.0; 4],
            color_end: [1.0; 4],
        }
    }
}
