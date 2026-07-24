// src/gfx/graphics_system/mod.rs
//
// GraphicsSystem: the 3D renderer driver. An internal system (not a declarable
// asset); `World::start` constructs one when the world declares a
// `GraphicsConfig`. Deliberately a directory rather than a single file; the
// system is large enough that splitting it by responsibility is worth it:
//   mod.rs       struct + System/Debug trait impls (init/step delegate out)
//   init.rs      run_init: one-time backend + draw-list setup
//   frame.rs     run_step: the per-frame transform upload + draw submit
//   streaming.rs texture / normal-map / mesh / voxel-world streaming setup
//                (the per-frame drive lives in gfx::streaming_system)
//   scene.rs     scene-flow wiring + scene visibility
//   helpers.rs   shared free functions

use crate::assets::{PostProcessResolve, WindowArgs};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::backend::RenderBackend;
use crate::gfx::{scene_flow, text};
use std::time::Instant;

const IDENTITY4: [[f32; 4]; 4] = crate::gfx::draw_list::IDENTITY4;

// Initialises the GPU backend and draws frame data.
//
// Components drained during init():
//   Window          -- window title, size, and mode
//   GraphicsConfig  -- frames-in-flight, clear color, max frames
//   Mesh            -- raw inline geometry payloads (keyed by asset name)
//   ProceduralMesh  -- generator-built geometry payloads (keyed by asset name)
//   Model           -- multi-mesh model definitions (keyed by asset name)
//   Prop            -- scene objects referencing a Mesh/ProceduralMesh or Model
//   ShaderStage     -- compiled shader payloads (vertex, fragment, shadow)
//   Texture         -- one or more compiled RGBA texture payloads (keyed by asset name)
//
// Components queried (not drained) each step():
//   Camera3D       -- current view matrix and projection parameters
//
// Build process:
//   Each Mesh is deserialized and kept in a name-keyed map. For each Prop the
//   corresponding mesh is looked up and appended to the shared vertex/index
//   buffers; a DrawObject records its slice offsets, model matrix, and texture
//   slot. One implicit DrawObject is also created for any Mesh that has no Prop
//   referencing it (e.g. the room itself), placed at the world origin.
//
// Input polling + the FrameInput deposit live in InputSystem, scheduled
// immediately after this system (the OS event pump runs inside draw_frame on
// Metal, so sampling right after the draw is freshest). Camera3DSystem queries
// the deposit to update Camera3D, then writes the new view matrix back in time
// for the next frame, so it runs after both.

// One viewport-pick candidate captured at init: the prop's asset id, its
// entity (the live GlobalTransform source), and its local-space bounds.
struct PickCandidate {
    asset_id: AssetId,
    entity: crate::ecs::Entity,
    local_min: [f32; 3],
    local_max: [f32; 3],
}

pub struct GraphicsSystem {
    window_args: WindowArgs,
    clear_color: [f32; 4],
    frames_in_flight: usize,
    vsync: bool,
    // Frame-rate cap in FPS (GraphicsConfig.fps_cap; 0 = unlimited). Applied by
    // the App-level frame pacer, which reads it through the `FrameRateCap`
    // resource this system publishes (at init and on the settings row's live
    // change). Held here so the settings row can cycle from the current value.
    // Independent of the quality preset (a user/hardware preference, like vsync).
    fps_cap: u32,
    // The display modes the Resolution row offers, shaped at init from the
    // backend's enumeration (or the static fallback when it cannot enumerate)
    // and published once as the `DisplayModes` resource for the dropdown list.
    display_modes: Vec<crate::gfx::display_mode::DisplayMode>,
    // The user's chosen fullscreen display mode, persisted as `resolution`.
    // `None` = never chosen: the display keeps its own mode and the row shows
    // `current_mode`. Fullscreen-only: windowed sizes come from the window
    // (authored / dragged) and borderless covers the display, so the row is
    // grayed + inert outside Fullscreen and never resizes the window.
    resolution: Option<crate::gfx::display_mode::DisplayMode>,
    // The mode the display was running at init (the row's display value until
    // the user chooses one). `None` when the backend cannot read it.
    current_mode: Option<crate::gfx::display_mode::DisplayMode>,
    // The Resolution row's labels with their authored colors, captured at init
    // so window-mode changes can gray the row out and restore it (mirrors
    // `perf_sub_row_labels`).
    resolution_row_labels: Vec<(AssetId, [f32; 3])>,
    // Stats-HUD display state (GraphicsSettings perf_stats / show_fps / show_vram;
    // default shown). `perf_stats` is the master "Display performance stats"
    // toggle; the per-readout flags gate the FPS / VRAM chips under it. Published
    // each frame as the `HudPrefs` resource for StatHudSystem, and persisted +
    // applied live by the settings-menu rows. When the master is off the two
    // sub-rows are grayed (their captured labels in `perf_sub_row_labels`) and
    // made inert (the `DisabledSettingRows` resource read by UiInputSystem).
    perf_stats: bool,
    show_fps: bool,
    show_vram: bool,
    // The TextLabel ids of the show_fps / show_vram rows with their authored
    // colors, captured at init so the master toggle can gray them and restore
    // them (the menu's HitRegions are drained after init, so the row -> label map
    // is captured once rather than re-queried).
    perf_sub_row_labels: Vec<(AssetId, [f32; 3])>,
    max_frames: Option<u64>,
    shadow_map_size: u32,
    shadow_update: crate::assets::ShadowUpdate,
    // Shadow distance in world units (GraphicsConfig.shadow_distance). Applied
    // live via set_shadow_distance (the per-frame cascade-split math reads it);
    // preset-governed (a manual change flips the master preset to Custom).
    shadow_distance: u32,
    // Active shadow cascade count, 1..=4 (GraphicsConfig.shadow_cascades). Applied
    // live via set_shadow_cascades (the per-frame split + schedule read it);
    // preset-governed (a manual change flips the master preset to Custom).
    shadow_cascades: u32,
    // Scene-sampler max anisotropy. Restart-required (the sampler is built once at
    // backend init from this), so this is display/persist state; the value reaches
    // the backend through the ctor. Preset-governed (a manual change flips the
    // master preset to Custom).
    anisotropy: u32,
    failed: bool,
    start_time: Option<Instant>,
    frame_count: u64,
    // A togglable menu (a Screen) coexists with a controlled Camera3D. When set,
    // cursor capture is driven each frame by whether a menu screen is active
    // (release while open, capture otherwise) rather than fixed at startup.
    menu_mode: bool,
    // Current render-scale (upscaling) quality, seeded at init from the world's
    // PostProcessConfig overridden by any persisted choice. The settings row
    // cycles + persists it; it is restart-required, so this is display/persist
    // state only (the upscaler is sized once at init).
    render_scale: crate::assets::UpscaleQuality,
    // Current upscaler backend (Auto / FSR3 / DLSS / XeSS), seeded at init from
    // the world's PostProcessConfig overridden by any persisted choice. Like
    // render_scale this is restart-required display/persist state (the upscaler
    // is selected + built once at init); DirectX / Vulkan only.
    upscale_backend: crate::assets::UpscalerBackend,
    // The render backend while init constructs and wires it. Boxed
    // `dyn RenderBackend` so the setup logic in init.rs / streaming.rs /
    // scene.rs runs as one cfg-free path across Metal, DirectX, and Vulkan.
    // At the end of a successful init it is parked in the world's
    // `ActiveRenderBackend` resource, where every per-step user (this system's
    // frame encode, InputSystem's poll) takes and returns it; `None` from then
    // on.
    backend: Option<Box<dyn RenderBackend>>,
    // active scene-flow bookkeeping while init builds it; handed to the shared
    // `ActiveSceneFlow` resource at the end of init (SettingsSystem jumps it,
    // this system ticks it). None when no Scene assets were declared.
    scene_flow: Option<scene_flow::SceneFlow>,
    // Overlay build inputs assembled during init() and handed to OverlaySystem
    // (as the `OverlayAssets` resource) at its end; empty afterwards. Fonts is
    // the atlas data keyed by handle; sprite_texture_slots maps a Sprite's
    // texture into the text-atlas pool (appended after the font atlases); the
    // chip id lists and scroll clip bands drive the per-frame HUD layout.
    loaded_fonts: std::collections::HashMap<crate::ecs::FontHandle, text::LoadedFont>,
    sprite_texture_slots: std::collections::HashMap<crate::ecs::TextureHandle, usize>,
    debug_hud_chips: Vec<AssetId>,
    stat_hud_chips: Vec<AssetId>,
    // Viewport-pick candidates captured at init, one per prop entity, only
    // when a `PickIndex` resource was present (the editor's opt-in). The frame
    // step refreshes the published index from these + the live transforms;
    // empty in a shipped runtime, which skips the refresh entirely.
    pick_candidates: Vec<PickCandidate>,
    // Streaming pools built during init (shared albedo+normal texture pool,
    // mesh geometry, and voxel-world chunks), each `Some` only when a
    // `StreamingConfig` / `VoxelWorld` was declared and the backend supports it
    // (Metal). Init scratch: they are moved into the parked `StreamingState`
    // resource at the end of init, where StreamingSystem drives them each frame,
    // so they are `None` here from then on.
    texture_streamer: Option<crate::gfx::streaming::texture::TextureStreamer>,
    mesh_streamer: Option<crate::gfx::streaming::mesh::MeshStreamer>,
    // Maps a streamed mesh's id to its DrawObject index, so completed loads
    // and evictions are applied to the right draw. Empty when not streaming.
    mesh_stream_draw_indices: Vec<usize>,
    chunk_stream: Option<crate::gfx::streaming_system::ChunkStreamState>,
    // Source catalogues captured at init for asset hot-reload, handed off to
    // the `cn debug` binary's reload machinery (which owns the watcher + the
    // live `AssetHotReloadState`). `Some` only under `cn debug` with at least
    // one file-backed asset / world.jsonl; taken once by the debug drive via
    // `take_hot_reload_sources`. `cn run` never captures these; production
    // reads asset payloads from the compiled blob and never re-touches disk.
    pending_hot_reload_sources: Option<hot_reload_sources::HotReloadSources>,
    // Texture-name map captured at init for runtime decal / emitter spawn to
    // resolve an authored Texture name to its live pool slot. `Some` only under
    // `cn debug`; read-only after init.
    world_reload: Option<WorldReloadState>,
    // Last `VolumetricFog` settings pushed to the backend, used by the
    // world.jsonl reload pass to dedupe: if the resolved value matches what's
    // already live, the reload skips the trait call and the log entry. Tracks
    // both `None` (no fog / disabled) and `Some(settings)`. Initialised by
    // `run_init` to whatever was passed into the backend constructor.
    last_fog_settings: Option<crate::gfx::volumetric_fog::FogSettings>,
    // Live post-process parameters (bloom / exposure / vignette / LUT blend),
    // the source of truth for slider settings. Seeded at init from the world's
    // resolved PostProcessConfig (with any persisted overrides applied); a
    // slider drag mutates a field here and pushes the whole struct to the
    // backend via `update_post_process`.
    post_process: crate::gfx::render_types::PostProcessParams,
    // Live ambient (IBL) light scale, the source of truth for the Ambient
    // slider. Lives in the backend's `LightUniforms` (not `PostProcessParams`),
    // so it is held + pushed separately via `set_ambient_intensity`. Seeded at
    // init from the world's `PostProcessConfig.ambient_intensity` (with any
    // persisted override applied) and pushed to the backend once after it is
    // built.
    ambient_intensity: f32,
    // The world's resolved PostProcessConfig with the user's persisted
    // quality-toggle overrides applied (defaulted when the world declares none).
    // The source of truth for the Quality-group toggles: a toggle flips the
    // matching field here, re-derives the per-feature settings, and pushes them
    // to the backend's live rebuild. The non-toggle fields (exposure, bloom,
    // ambient) keep their authored values here; the sliders own those via
    // `post_process` / `ambient_intensity` instead.
    post_config: crate::assets::PostProcessConfig,
    // Slider rows in the world, captured at init from their drag HitRegions +
    // handle Sprites. Drives the handle position + value-label update when a
    // slider changes, and the one-time sync of both to the live value at init.
    sliders: Vec<SliderViz>,
    // Cycle rows' setting key -> value-label id, captured at init from their
    // `setting:<key>:next` HitRegions (drained by UiInputSystem afterwards). Lets
    // a runtime change relabel a row other than the one clicked: the master
    // "Graphics Quality" preset relabels the quality toggles + render scale it
    // re-derives, and an individual quality-row change relabels the master row.
    cycle_value_labels: std::collections::HashMap<String, AssetId>,
    // Per-element clip bands (reference space) captured at init from the world's
    // ScrollPanels: each scroll-content element id maps to its panel's content
    // band, so the draw path scissors it and off-band rows do not bleed over the
    // panel chrome. Empty when no ScrollPanel was declared; handed to
    // OverlaySystem (inside `OverlayAssets`) at the end of init.
    clip_rects: std::collections::HashMap<AssetId, [f32; 4]>,
    // Live gameplay movement key map (the source of truth for the Controls-tab
    // rebind rows). Seeded at init from the persisted `ControlsSettings.keymap`
    // or the engine default, pushed to the backend once after it is built, and
    // updated (with a swap) + re-pushed + persisted on each rebind.
    keymap: crate::gfx::keymap::KeyMap,
    // Rebind rows in the world, captured at init from their `setting:key_*:rebind`
    // HitRegions. Maps each rebindable action to its value `TextLabel`, so a
    // rebind (and the swap it may trigger) can refresh both affected row labels.
    rebind_rows: Vec<RebindViz>,
    // Live gamepad action -> button map. Seeded at init from the persisted
    // `ControlsSettings.gamepad_map` or the engine default; InputSystem applies
    // it (the gamepad is polled engine-side, so no backend push).
    gamepad_map: crate::assets::GamepadMap,
    // Gamepad rebind rows in the world, captured at init from their
    // `setting:pad_*:rebind` HitRegions, like `rebind_rows`.
    pad_rebind_rows: Vec<PadRebindViz>,
    // Device capability flags, queried from the backend once it is built. Drives
    // the capability gating at init: a settings row whose feature the device
    // cannot provide (e.g. ray-traced reflections without hardware ray tracing)
    // is grayed out and made inert. Held in memory only, never persisted.
    caps: crate::gfx::backend::DeviceCapabilities,
    // Coarse GPU performance profile, probed before the backend is built so the
    // auto-config quality ceiling can influence the render targets / effect
    // pipelines sized at backend init. Held in memory only, never persisted.
    gpu_profile: crate::gfx::backend::GpuProfile,
    // The live master "Graphics Quality" preset the settings-menu row cycles.
    // Seeded at init from the persisted choice (or `Auto` on first launch);
    // changing a preset re-derives the quality toggles + render scale under its
    // ceiling, and changing any individual quality row flips this to `Custom`.
    quality_preset: crate::gfx::quality_preset::QualityPreset,
    // The world's authored PostProcessConfig before the user overrides + preset
    // ceiling are applied (defaulted when the world declares none). The pristine
    // baseline a live preset change re-clamps from, so up-shifting a preset
    // restores the world's features and down-shifting clamps them off.
    authored_post_config: crate::assets::PostProcessConfig,
    // Display-output / upscaling preferences (the Display settings rows). Resolved
    // at init from the world's `PostProcessConfig` overridden by any persisted
    // choice, passed to the backend ctor, and held here so the rows display +
    // cycle them. Restart-required (swapchain format / render targets are sized
    // once at init), so a runtime change only persists + relabels; independent of
    // the quality preset.
    temporal_upscaling: bool,
    hdr_display: bool,
    hdr_pq: bool,
    // The world's authored shadow knobs before the user overrides + preset ceiling
    // (defaulted when the world declares no GraphicsConfig). The pristine baseline
    // a live preset change re-clamps from, like `authored_post_config`. The live
    // values are `shadow_map_size` / `shadow_update` above.
    authored_shadow_map_size: u32,
    authored_shadow_update: crate::assets::ShadowUpdate,
    // The world's authored shadow distance, the baseline a live preset change
    // re-clamps from. The live value is `shadow_distance` above.
    authored_shadow_distance: u32,
    // The world's authored shadow cascade count, the baseline a live preset
    // change re-clamps from. The live value is `shadow_cascades` above.
    authored_shadow_cascades: u32,
    // The world's authored anisotropy degree before the user override + preset
    // ceiling, the baseline a live preset change re-clamps from (like
    // `authored_shadow_map_size`). The live value is `anisotropy` above.
    authored_anisotropy: u32,
    // System / streaming restart preferences (the Advanced "Frame Buffering",
    // "Occlusion Culling", and "Texture Quality" rows). Resolved at init from the
    // world's config overridden by any persisted choice, passed to the backend
    // ctor / streamer, and held here so the rows display + cycle them. Restart-
    // required, independent of the quality preset. `frames_in_flight` lives above.
    occlusion_two_pass: bool,
    texture_cap: u32,
    texture_budget: u32,
    // Reused scratch + change-tracking for the per-frame transform propagation
    // (`draw_list::propagate_transforms_cached`): buffers are refilled in place
    // and the pass is skipped on frames where no Transform / Parent changed.
    transform_cache: crate::gfx::draw_list::TransformCache,
    // Test-only injection seam: pre-resolved settings, a fabricated GPU
    // profile, and a mock backend factory, so unit tests can drive
    // run_init / run_step without a GPU device or the on-disk settings store.
    #[cfg(test)]
    pub(crate) test_hooks: Option<crate::gfx::mock_backend::TestHooks>,
}

// One key-rebind row's runtime bookkeeping: the action it rebinds and the value
// `TextLabel` showing its bound key. Built at init (`init_rebind_rows`) from the
// row's `setting:key_*:rebind` HitRegion (`action` -> `Bindable`, `label`) and
// handed to SettingsState, which drives the live rebind drain.
pub(crate) struct RebindViz {
    pub(crate) action: crate::gfx::keymap::Bindable,
    pub(crate) value_id: AssetId,
}

// One gamepad-rebind row's runtime bookkeeping, mirroring `RebindViz`: built at
// init (`init_pad_rebind_rows`) from the row's `setting:pad_*:rebind` HitRegion
// and handed to SettingsState for the button-rebind drain.
pub(crate) struct PadRebindViz {
    pub(crate) action: crate::assets::GamepadAction,
    pub(crate) value_id: AssetId,
}

// One slider row's runtime bookkeeping: the engine setting it controls, the
// track geometry it maps a fraction onto, and the handle Sprite + value
// TextLabel it drives. Built at init (`init_sliders`) from the row's
// `setting:<key>:drag` HitRegion (track `x`/`width`, `label`, `drag_handle`) and
// the handle Sprite's width, then handed to SettingsState for the slider drain.
pub(crate) struct SliderViz {
    pub(crate) key: String,
    pub(crate) track_x: f32,
    pub(crate) track_w: f32,
    pub(crate) handle_w: f32,
    pub(crate) handle_id: AssetId,
    pub(crate) value_id: AssetId,
}

// Init-time asset-resolution tables consulted by the world.jsonl hot-reload
// pass when applying adds and non-transform edits. Captured at init and
// never mutated afterwards: the reload path cannot introduce new
// Materials / Textures / Meshes / Models on the fly (those need a process
// restart), but every authored Prop that points at an asset already in the
// init world resolves through these maps without re-running build.
// Built by init, read only by the `cn debug` binary's world.jsonl reload pass,
// so its fields read as dead under `cargo check --lib`.
#[allow(dead_code)]
pub struct WorldReloadState {
    // Texture asset name -> live pool slot, so runtime decal / emitter spawn
    // (`cn debug`) can resolve an authored Texture name to its slot.
    pub texture_name_to_slot: std::collections::HashMap<AssetId, usize>,
}

// Disjoint mutable screen of the `GraphicsSystem` fields the hot-reload passes
// edit in one tick: the active backend, the texture-name map for runtime
// decal / emitter spawn, and the fog bookkeeping the world.jsonl reload pass
// dedupes against. Returned by [`GraphicsSystem::hot_reload_apply_parts`] so the
// binary-only `DebugHook::tick` drive can apply the reload passes from outside
// the per-system step without the library depending on it. The reload catalogue
// + in-flight state live on the debug side (`crate::debug::hot_reload`), built
// from [`HotReloadSources`]. The library never constructs this; hence the
// `dead_code` allowance (the fields are read only from the `cn debug` binary's
// drive, never under `cargo check --lib`).
#[allow(dead_code)]
pub struct HotReloadApplyParts<'a> {
    pub backend: &'a mut dyn RenderBackend,
    pub world_reload: &'a Option<WorldReloadState>,
    pub last_fog_settings: &'a mut Option<crate::gfx::volumetric_fog::FogSettings>,
}

impl std::fmt::Debug for GraphicsSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphicsSystem")
            .field("frame_count", &self.frame_count)
            .field("failed", &self.failed)
            .finish()
    }
}

impl GraphicsSystem {
    // Fresh renderer driver with no backend yet. Config (frames-in-flight,
    // clear color, `max_frames`, shadow-map size) is read from the world's
    // `GraphicsConfig` in [`System::init`]; the validation request comes from
    // the CLI via `dev_flags`.
    pub fn new() -> Self {
        Self {
            window_args: Default::default(),
            clear_color: [0.01, 0.01, 0.02, 1.0],
            frames_in_flight: 2,
            vsync: false,
            fps_cap: 0,
            display_modes: Vec::new(),
            resolution: None,
            current_mode: None,
            resolution_row_labels: Vec::new(),
            perf_stats: true,
            show_fps: true,
            show_vram: true,
            perf_sub_row_labels: Vec::new(),
            max_frames: None,
            shadow_map_size: 2048,
            shadow_update: crate::assets::ShadowUpdate::default(),
            shadow_distance: 80,
            shadow_cascades: 4,
            anisotropy: 8,
            failed: false,
            start_time: None,
            frame_count: 0,
            menu_mode: false,
            render_scale: crate::assets::UpscaleQuality::default(),
            upscale_backend: crate::assets::UpscalerBackend::default(),
            backend: None,
            scene_flow: None,
            loaded_fonts: std::collections::HashMap::new(),
            sprite_texture_slots: std::collections::HashMap::new(),
            debug_hud_chips: Vec::new(),
            stat_hud_chips: Vec::new(),
            pick_candidates: Vec::new(),
            texture_streamer: None,
            mesh_streamer: None,
            mesh_stream_draw_indices: Vec::new(),
            chunk_stream: None,
            pending_hot_reload_sources: None,
            world_reload: None,
            last_fog_settings: None,
            post_process: crate::gfx::render_types::PostProcessParams::DEFAULT,
            // Matches PostProcessConfig's ambient_intensity default; overwritten
            // at init from the world / persisted store.
            ambient_intensity: 1.0,
            // Default until init resolves the world's config + persisted toggles.
            post_config: crate::assets::PostProcessConfig::default(),
            sliders: Vec::new(),
            cycle_value_labels: std::collections::HashMap::new(),
            clip_rects: std::collections::HashMap::new(),
            keymap: crate::gfx::keymap::KeyMap::default(),
            rebind_rows: Vec::new(),
            gamepad_map: crate::assets::GamepadMap::default(),
            pad_rebind_rows: Vec::new(),
            // All-capable until the backend reports otherwise at init.
            caps: crate::gfx::backend::DeviceCapabilities::ALL,
            // Conservative until probed at init.
            gpu_profile: crate::gfx::backend::GpuProfile::UNKNOWN,
            // Seeded at init from the persisted preset (Auto on first launch).
            quality_preset: crate::gfx::quality_preset::QualityPreset::Auto,
            // Defaulted until init captures the world's authored config.
            authored_post_config: crate::assets::PostProcessConfig::default(),
            // Resolved at init from the world's config + persisted overrides.
            temporal_upscaling: false,
            hdr_display: false,
            hdr_pq: false,
            authored_shadow_map_size: 2048,
            authored_shadow_update: crate::assets::ShadowUpdate::default(),
            authored_shadow_distance: 80,
            authored_shadow_cascades: 4,
            authored_anisotropy: 8,
            occlusion_two_pass: false,
            texture_cap: 96,
            texture_budget: 4,
            transform_cache: crate::gfx::draw_list::TransformCache::default(),
            #[cfg(test)]
            test_hooks: None,
        }
    }

    // The persisted settings store consulted at init. Reads the on-disk file
    // in production; a test-injected copy takes its place so unit tests never
    // read (or depend on) the developer's real settings.
    fn persisted_settings(&self) -> crate::config::Settings {
        #[cfg(test)]
        if let Some(hooks) = &self.test_hooks {
            return hooks.settings.clone();
        }
        crate::config::Settings::load()
    }

    // Detect the GPU performance profile for quality auto-config. Probes the
    // real device in production; a test-injected profile takes its place so
    // unit tests never create a GPU handle.
    fn detect_gpu_profile(&self) -> crate::gfx::backend::GpuProfile {
        #[cfg(test)]
        if let Some(hooks) = &self.test_hooks {
            return hooks.gpu_profile;
        }
        concinnity_device::probe_gpu_profile()
    }

    // Seed and persist the first-launch `Auto` quality preset. Skipped under
    // the test injection seam so tests never write the settings file.
    fn seed_first_launch_preset(&self) {
        #[cfg(test)]
        if self.test_hooks.is_some() {
            return;
        }
        let mut s = crate::config::Settings::load();
        s.graphics.quality_preset = Some(crate::gfx::quality_preset::QualityPreset::Auto);
        if let Err(e) = s.save() {
            tracing::warn!("first-launch quality preset save failed: {e}");
        }
    }

    // The mode the Resolution row displays and cycles from: the user's choice,
    // else the display's own mode, else the authored window size (a backend
    // that cannot read the display; snaps to the nearest listed mode).
    fn effective_resolution(&self) -> crate::gfx::display_mode::DisplayMode {
        self.resolution
            .or(self.current_mode)
            .unwrap_or(crate::gfx::display_mode::DisplayMode {
                width: self.window_args.width,
                height: self.window_args.height,
                refresh_hz: 0,
            })
    }
}

impl Default for GraphicsSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for GraphicsSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        self.run_init(ctx);
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        self.run_step(ctx)
    }
}

impl GraphicsSystem {
    // Disjoint mutable screen of the backend + hot-reload bookkeeping the
    // binary-only `DebugHook::tick` reload drive applies changes through. The
    // caller supplies the backend (borrowed from the world's parked slot via
    // `World::systems_and_render_backend`) since this system yields it after
    // init. The library never calls this (the asset hot-reload drive lives in
    // the `cn debug` binary), so it reads as dead code under
    // `cargo check --lib`.
    #[allow(dead_code)]
    pub fn hot_reload_apply_parts<'a>(
        &'a mut self,
        backend: &'a mut dyn RenderBackend,
    ) -> HotReloadApplyParts<'a> {
        HotReloadApplyParts {
            backend,
            world_reload: &self.world_reload,
            last_fog_settings: &mut self.last_fog_settings,
        }
    }

    // Take the init-captured hot-reload source catalogues, leaving `None`
    // behind. The `cn debug` drive calls this once on its first tick to build
    // the filesystem watcher + `AssetHotReloadState`. `None` under `cn run`,
    // or when no file-backed asset / world.jsonl was declared. Library-dead
    // (only the binary calls it).
    #[allow(dead_code)]
    pub fn take_hot_reload_sources(&mut self) -> Option<hot_reload_sources::HotReloadSources> {
        self.pending_hot_reload_sources.take()
    }

    // Stand up the albedo-texture streaming subsystem when a StreamingConfig
    // was declared. Every streamable slot is evicted to a placeholder now; the
    // streamer brings them back resident over the next frames, nearest first.
    //
    // The payload source depends on where the world came from: a disk-backed
    // `cn run` world re-reads each payload from its blob file (no RAM copy), an
    // in-memory `cn debug` world keeps the payloads RAM-resident.
}

// Quality-toggle plumbing shared by init (value-label sync + initial overlay)
// and the per-frame drain. Centralising the key -> `PostProcessConfig` field
// mapping here keeps the three call sites (read state, flip state, derive the
// backend settings) from drifting apart.

// The current on/off state of quality toggle `key` in `cfg`, or `None` for a
// key that is not a quality toggle.
pub(crate) fn quality_toggle_on(cfg: &crate::assets::PostProcessConfig, key: &str) -> Option<bool> {
    match key {
        "ssao" => Some(cfg.ssao),
        "ssr" => Some(cfg.ssr),
        "ray_traced_reflections" => Some(cfg.ray_traced_reflections),
        "ssgi" => Some(cfg.indirect_lighting == crate::assets::IndirectLighting::Ssgi),
        "auto_exposure" => Some(cfg.auto_exposure),
        _ => None,
    }
}

// Flip quality toggle `key` to `on` in `cfg`. Unknown keys are ignored.
pub(crate) fn set_quality_toggle(cfg: &mut crate::assets::PostProcessConfig, key: &str, on: bool) {
    match key {
        "ssao" => cfg.ssao = on,
        "ssr" => cfg.ssr = on,
        "ray_traced_reflections" => cfg.ray_traced_reflections = on,
        "ssgi" => {
            cfg.indirect_lighting = if on {
                crate::assets::IndirectLighting::Ssgi
            } else {
                crate::assets::IndirectLighting::Ibl
            }
        }
        "auto_exposure" => cfg.auto_exposure = on,
        _ => {}
    }
}

// Whether `key` is one of the cycle (dropdown) quality knobs governed by the
// preset ceiling like the boolean toggles (a manual change flips the preset to
// Custom). The set lives in `settings::QUALITY_CYCLE_KEYS`.
pub(crate) fn is_quality_cycle(key: &str) -> bool {
    crate::gfx::settings::QUALITY_CYCLE_KEYS.contains(&key)
}

// The current menu option index of cycle quality knob `key` in `cfg`, or `None`
// for a key that is not a cycle quality knob.
pub(crate) fn quality_cycle_index(
    cfg: &crate::assets::PostProcessConfig,
    key: &str,
) -> Option<usize> {
    use crate::gfx::settings;
    match key {
        "aa_mode" => Some(settings::aa_mode_index(cfg.aa_mode)),
        "ssgi_resolution" => Some(settings::ssgi_resolution_index(cfg.ssgi_resolution)),
        "ssgi_rays" => Some(settings::ssgi_rays_index(cfg.ssgi_rays)),
        "ssgi_steps" => Some(settings::ssgi_steps_index(cfg.ssgi_steps)),
        "reflection_blur_resolution" => Some(settings::reflection_blur_index(
            cfg.reflection_blur_resolution,
        )),
        _ => None,
    }
}

// Set cycle quality knob `key` in `cfg` from a menu option index. Unknown keys
// are ignored.
pub(crate) fn set_quality_cycle(
    cfg: &mut crate::assets::PostProcessConfig,
    key: &str,
    index: usize,
) {
    use crate::gfx::settings;
    match key {
        "aa_mode" => cfg.aa_mode = settings::aa_mode_at(index),
        "ssgi_resolution" => cfg.ssgi_resolution = settings::ssgi_resolution_at(index),
        "ssgi_rays" => cfg.ssgi_rays = settings::ssgi_rays_at(index),
        "ssgi_steps" => cfg.ssgi_steps = settings::ssgi_steps_at(index),
        "reflection_blur_resolution" => {
            cfg.reflection_blur_resolution = settings::reflection_blur_at(index)
        }
        _ => {}
    }
}

// Clamp cycle quality knob `key` in `cfg` DOWN under the ceiling (coarser
// resolution / smaller count; never raises), a no-op when the user explicitly
// overrode it. Shared by the init clamp and the live preset re-derive so both
// produce the same result.
pub(crate) fn clamp_quality_cycle(
    cfg: &mut crate::assets::PostProcessConfig,
    key: &str,
    ceiling: &crate::gfx::quality_preset::QualityCeiling,
    overridden: bool,
) {
    if overridden {
        return;
    }
    use crate::gfx::quality_preset::{
        clamp_aa_mode, coarser_reflection_blur, coarser_ssgi_resolution,
    };
    match key {
        "aa_mode" => cfg.aa_mode = clamp_aa_mode(cfg.aa_mode, ceiling.aa_mode),
        "ssgi_resolution" => {
            cfg.ssgi_resolution =
                coarser_ssgi_resolution(cfg.ssgi_resolution, ceiling.ssgi_resolution)
        }
        "ssgi_rays" => cfg.ssgi_rays = cfg.ssgi_rays.min(ceiling.ssgi_rays),
        "ssgi_steps" => cfg.ssgi_steps = cfg.ssgi_steps.min(ceiling.ssgi_steps),
        "reflection_blur_resolution" => {
            cfg.reflection_blur_resolution = coarser_reflection_blur(
                cfg.reflection_blur_resolution,
                ceiling.reflection_blur_resolution,
            )
        }
        _ => {}
    }
}

// Derive the backend's per-feature `QualitySettings` from a resolved config.
// Mirrors the init-time derivation (the same `*_settings()` methods), so a
// live rebuild reproduces exactly what a launch with this config would build.
pub(crate) fn derive_quality_settings(
    cfg: &crate::assets::PostProcessConfig,
) -> crate::gfx::backend::QualitySettings {
    crate::gfx::backend::QualitySettings {
        taa: cfg.aa_mode.taa_enabled(),
        ssao: cfg.ssao_settings(),
        ssr: cfg.ssr_settings(),
        rt_reflections: cfg.rt_reflection_settings(),
        ssgi: cfg.ssgi_settings(),
        reflection_blur_scale: cfg.reflection_blur_divisor(),
        auto_exposure: cfg.auto_exposure_settings(),
        auto_exposure_bias_ev: cfg.exposure_ev,
    }
}

mod frame;
mod helpers;
pub mod hot_reload_sources;
mod init;
pub(crate) mod scene;
mod streaming;
#[cfg(test)]
mod tests;
