//! GraphicsSystem: the 3D renderer driver. An internal system (not a declarable
//! asset); `World::start` constructs one when the run resolved to a windowed
//! one. Deliberately a directory rather than a single file; the
//! system is large enough that splitting it by responsibility is worth it:
//!   mod.rs       struct + System/Debug trait impls (init/step delegate out)
//!   init.rs      run_init: one-time backend + draw-list setup
//!   lines.rs     published world-space lines -> ribbon geometry
//!   frame.rs     run_step: extraction of the frame's draw inputs into the
//!                owned RenderSnapshot (the only per-frame world reads)
//!   submit.rs    replay of one RenderSnapshot onto the backend (no world
//!                access by construction)
//!   streaming.rs texture / normal-map / mesh / voxel-world streaming setup
//!                (the per-frame drive lives in gfx::streaming::system)
//!   scene.rs     scene-flow wiring + scene visibility
//!   stream_sources.rs  streamed texture payload sources + voxel palette entries
//!   draw_geometry.rs   draw-object positions + auto-seed triangle gathering

use concinnity_core::components::{GamepadAction, GraphicsConfig, PostProcessConfig};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{Entity, PipelineContext, StepResult, System};
use concinnity_core::input::keymap;
use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::post::rt_reflections::RtReflectionSettings;
use concinnity_core::render::post::ssao::SsaoSettings;
use concinnity_core::render::post::ssgi::settings::SsgiSettings;
use concinnity_core::render::post::ssr::settings::SsrSettings;
use concinnity_core::render::{backend, overlay_maps, scene_flow, snapshot, text};
use concinnity_core::settings::SettingKey;
use concinnity_core::transform::propagation;
use concinnity_host::store::paths::StateTree;

const IDENTITY4: [[f32; 4]; 4] = crate::gfx::draw_list::IDENTITY4;

// Initializes the GPU backend and draws frame data.
//
// Components drained during init():
//   Window          -- window title, size, and mode
//   GraphicsConfig  -- frames-in-flight, clear color, max frames
//   Mesh            -- raw inline geometry payloads (keyed by asset name)
//   ProceduralMesh  -- generator-built geometry payloads (keyed by asset name)
//   Model           -- multi-mesh model definitions (keyed by asset name)
//   Prop            -- scene objects referencing a Mesh/ProceduralMesh or Model
//   Shader          -- compiled shader payloads (vertex, fragment, instanced)
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
    entity: Entity,
    local_min: [f32; 3],
    local_max: [f32; 3],
}

/// Drives the render backend: builds it at init, submits a frame per step.
pub struct GraphicsSystem {
    // Where this world reads its source assets and writes its settings, or
    // `None` for a world with no state tree.
    state: Option<StateTree>,
    clear_color: [f32; 4],
    max_frames: Option<u64>,
    failed: bool,
    // Seconds of frame time this system has rendered, the shaders' clock.
    render_secs: f32,
    frame_count: u64,
    // Per-class recovery for failed frames; see `frame_policy`.
    frame_policy: frame_policy::FramePolicy,
    // A togglable menu (a Screen) coexists with a controlled Camera3D. When set,
    // cursor capture is driven each frame by whether a menu screen is active
    // (release while open, capture otherwise) rather than fixed at startup.
    menu_mode: bool,
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
    // Per-entity scene-visibility snapshot, refreshed (buffers reused) every
    // frame a fade runs and on scene-visibility applies.
    scene_visibility: scene::SceneVisibilityScratch,
    // Overlay build inputs assembled during init() and handed to OverlaySystem
    // (as the `OverlayAssets` resource) at its end; empty afterwards. Fonts is
    // the atlas data keyed by handle; sprite_texture_slots maps a Sprite's
    // texture into the text-atlas pool (appended after the font atlases); the
    // chip id lists and scroll clip bands drive the per-frame HUD layout.
    loaded_fonts: text::FontSet,
    sprite_texture_slots: overlay_maps::TextureSlots,
    debug_hud_chips: Vec<AssetId>,
    stat_hud_chips: Vec<AssetId>,
    // Viewport-pick candidates captured at init, one per prop entity, only
    // when a `PickIndex` resource was present (the editor's opt-in). The frame
    // step refreshes the published index from these + the live transforms;
    // empty in a shipped runtime, which skips the refresh entirely.
    pick_candidates: Vec<PickCandidate>,
    // Prop entities drawing a skybox-generated mesh, captured at init. The
    // frame step moves them onto the camera so the sky encloses it wherever it
    // goes; empty in a world with no sky.
    sky_props: Vec<Entity>,
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
    chunk_stream: Option<crate::gfx::streaming::system::ChunkStreamState>,
    // Shader buckets whose pipeline init deferred, with the payload source the
    // pump reads when their scene pins. Init scratch like the pools above.
    shader_warmup: Option<crate::gfx::streaming::shader::ShaderWarmup>,
    // Which scene exclusively owns each deferred bucket, so scene residency
    // can claim it as a member.
    deferred_shader_scenes: Vec<(u32, AssetId)>,
    // Per-element clip bands (reference space) captured at init from the world's
    // ScrollPanels: each scroll-content element id maps to its panel's content
    // band, so the draw path scissors it and off-band rows do not bleed over the
    // panel chrome. Empty when no ScrollPanel was declared; handed to
    // OverlaySystem (inside `OverlayAssets`) at the end of init.
    clip_rects: overlay_maps::ClipRects,
    // Device capability flags, queried from the backend once it is built. Drives
    // the capability gating at init: a settings row whose feature the device
    // cannot provide (e.g. ray-traced reflections without hardware ray tracing)
    // is grayed out and made inert. Held in memory only, never persisted.
    caps: backend::DeviceCapabilities,
    // Reused scratch + change-tracking for the per-frame transform propagation
    // (`propagation::propagate_transforms_cached`): buffers are refilled in place
    // and the pass is skipped on frames where no Transform / Parent changed.
    transform_cache: propagation::TransformCache,
    // The sky angle the directional-light set was last carried at. `None` until
    // the first frame, so a world whose sky never turns carries it exactly once.
    pushed_sky_angle: Option<f32>,
    // Last-pushed model matrix per draw slot / skinned instance: a static
    // slot costs a compare instead of a snapshot entry, and each family
    // crosses the backend trait once per frame.
    model_push: model_push::ModelPushCache,
    skinned_model_push: model_push::ModelPushCache,
    // The owned per-frame draw inputs `extract` fills from world state and
    // `submit` replays onto the backend. Held here so its buffers keep their
    // capacity across frames; taken out of `self` for the duration of one
    // step.
    snapshot: snapshot::RenderSnapshot,
    // Logical viewport size the line builder maps ribbon widths with. Seeded
    // from the backend at init, refreshed each frame from `FrameInput`.
    viewport: (f32, f32),
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
    pub(crate) action: keymap::Bindable,
    pub(crate) value_id: AssetId,
}

// One gamepad-rebind row's runtime bookkeeping, mirroring `RebindViz`: built at
// init (`init_rebind_rows`) from the row's `setting:pad_*:rebind` HitRegion
// and handed to SettingsState for the button-rebind drain.
pub(crate) struct PadRebindViz {
    pub(crate) action: GamepadAction,
    pub(crate) value_id: AssetId,
}

// One slider row's runtime bookkeeping: the engine setting it controls, the
// track geometry it maps a fraction onto, and the handle Sprite + value
// TextLabel it drives. Built at init (`init_sliders`) from the row's
// `setting:<key>:drag` HitRegion (track `x`/`width`, `label`, `drag_handle`) and
// the handle Sprite's width, then handed to SettingsState for the slider drain.
pub(crate) struct SliderViz {
    pub(crate) key: SettingKey,
    pub(crate) track_x: f32,
    pub(crate) track_w: f32,
    pub(crate) handle_w: f32,
    pub(crate) handle_id: AssetId,
    pub(crate) value_id: AssetId,
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
    /// Fresh renderer driver with no backend yet, reading and writing under
    /// `tree`. Config (frames-in-flight, clear color, `max_frames`, shadow-map
    /// size) is read from the world's `GraphicsConfig` in [`System::init`].
    pub fn new(tree: Option<&StateTree>) -> Self {
        // The schema's own defaults, so a world with no GraphicsConfig sees the
        // same values as one that declares an all-default component.
        let gfx = GraphicsConfig::default();
        Self {
            state: tree.cloned(),
            clear_color: gfx.clear_color,
            max_frames: gfx.max_frames,
            failed: false,
            render_secs: 0.0,
            frame_count: 0,
            frame_policy: frame_policy::FramePolicy::default(),
            menu_mode: false,
            backend: None,
            scene_flow: None,
            scene_visibility: Default::default(),
            loaded_fonts: text::FontSet::default(),
            sprite_texture_slots: overlay_maps::TextureSlots::new(),
            debug_hud_chips: Vec::new(),
            stat_hud_chips: Vec::new(),
            pick_candidates: Vec::new(),
            sky_props: Vec::new(),
            texture_streamer: None,
            mesh_streamer: None,
            mesh_stream_draw_indices: Vec::new(),
            chunk_stream: None,
            shader_warmup: None,
            deferred_shader_scenes: Vec::new(),
            clip_rects: overlay_maps::ClipRects::new(),
            // All-capable until the backend reports otherwise at init.
            caps: backend::DeviceCapabilities::ALL,
            transform_cache: propagation::TransformCache::default(),
            pushed_sky_angle: None,
            model_push: model_push::ModelPushCache::default(),
            skinned_model_push: model_push::ModelPushCache::default(),
            snapshot: snapshot::RenderSnapshot::default(),
            viewport: (0.0, 0.0),
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
        crate::config::Settings::load(self.state.as_ref())
    }

    // The `assets/` a bare source filename is searched under: the running
    // world's own, or nothing for a world with no state tree (which leaves a
    // bare filename unresolved rather than searched from the cwd).
    pub(crate) fn assets_dir(&self) -> Option<std::path::PathBuf> {
        self.state.as_ref().map(StateTree::assets_dir)
    }

    // Detect the GPU performance profile for quality auto-config. Probes the
    // real device in production; a test-injected profile takes its place so
    // unit tests never create a GPU handle.
    fn detect_gpu_profile(&self) -> backend::GpuProfile {
        #[cfg(test)]
        if let Some(hooks) = &self.test_hooks {
            return hooks.gpu_profile;
        }
        // A GPU that classifies as nothing clamps nothing. A machine with no
        // GPU at all never reaches here: the run resolved to headless and this
        // system was left out of the schedule.
        crate::device::probe_gpu_profile().unwrap_or(backend::GpuProfile::UNKNOWN)
    }

    // Seed and persist the first-launch `Auto` quality preset.
    fn seed_first_launch_preset(&self) {
        let mut s = crate::config::Settings::load(self.state.as_ref());
        s.graphics.quality_preset = Some(crate::gfx::quality_preset::QualityPreset::Auto);
        if let Err(e) = s.save(self.state.as_ref()) {
            tracing::warn!("first-launch quality preset save failed: {e}");
        }
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

// Derive the backend's per-feature `QualitySettings` from a resolved config.
// Mirrors the init-time derivation (the same resolves), so a
// live rebuild reproduces exactly what a launch with this config would build.
pub(crate) fn derive_quality_settings(cfg: &PostProcessConfig) -> backend::QualitySettings {
    backend::QualitySettings {
        taa: cfg.aa_mode.taa_enabled(),
        ssao: SsaoSettings::from_config(cfg),
        ssr: SsrSettings::from_config(cfg),
        rt_reflections: RtReflectionSettings::from_config(cfg),
        ssgi: SsgiSettings::from_config(cfg),
        reflection_blur_scale: cfg.reflection_blur_divisor(),
        auto_exposure: cfg.auto_exposure_settings(),
        auto_exposure_bias_ev: cfg.exposure_ev,
    }
}

mod backend_handoff;
mod blob_release;
pub(crate) mod character_shape;
mod draw_geometry;
mod frame;
pub(crate) mod frame_policy;
pub mod hot_reload_sources;
mod init;
mod lines;
mod mesh_seed_compaction;
mod mesh_stream_inputs;
mod model_push;
pub mod parked;
mod prop_draws;
pub(crate) mod scene;
mod scene_lights;
mod skinned_templates;
mod sky_follow;
mod stream_plan;
mod stream_sources;
mod streaming;
pub(crate) mod submit;
#[cfg(test)]
mod tests;
mod texture_payloads;
mod world_fx;
