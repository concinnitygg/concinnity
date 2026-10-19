//! GraphicsSystem: the 3D renderer driver. An internal system (not a declarable
//! asset); `World::start` constructs one when the run resolved to a windowed
//! one.

use concinnity_core::components::Background;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{PipelineContext, StepResult, System, World};
use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::error::RenderError;
use concinnity_core::render::extract::FrameExtractor;
use concinnity_core::render::frame_outcome::FrameCounter;
use concinnity_core::render::frame_submit::FramePolicy;
use concinnity_core::render::render_mode::RenderMode;
use concinnity_core::render::stream_pools::StreamingPools;
use concinnity_core::render::world_load::scene_deferral::ShaderDeferral;
use concinnity_core::render::{backend, overlay_maps, scene_flow, snapshot, text};
use concinnity_host::store::paths::StateTree;

/// Why the renderer could not be built, left by GraphicsSystem's init for
/// `Runtime::start` to pick up and report.
///
/// Only a world that already resolved to a windowed run can leave one, so this
/// always means a GPU that refused. A machine with no GPU never builds the
/// system that publishes it.
pub(crate) struct RenderInitFailure(pub RenderError);

/// Drives the render backend: builds it at init, submits a frame per step.
pub(crate) struct GraphicsSystem {
    // Where this world reads its source assets and writes its settings, or
    // `None` for a world with no state tree.
    state: Option<StateTree>,
    clear_color: [f32; 4],
    failed: bool,
    // Frames submitted, against the launch's cap.
    frames: FrameCounter,
    // Per-class recovery for failed frames.
    frame_policy: FramePolicy,
    // The cross-frame state the frame's extraction keeps.
    extractor: FrameExtractor,
    // The render backend while init constructs and wires it. Boxed
    // `dyn RenderBackend` so the setup logic in init.rs and streaming.rs runs
    // as one cfg-free path across Metal, DirectX, and Vulkan.
    // At the end of a successful init it is parked in the world's
    // `ActiveRenderBackend` resource, where this system's frame step takes and
    // returns it; `None` from then on.
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
    loaded_fonts: text::FontSet,
    sprite_texture_slots: overlay_maps::TextureSlots,
    debug_hud_chips: Vec<AssetId>,
    stat_hud_chips: Vec<AssetId>,
    // Streaming pools built during init (shared albedo+normal texture pool,
    // mesh geometry, voxel-world chunks, planet tiles) and their load threads,
    // with the shader buckets whose pipeline init deferred, each present only
    // when the world declares it and the backend supports it. Init scratch:
    // moved into the published `StreamingRuntime` resource at the end of init.
    stream_pools: StreamingPools,
    stream_workers: crate::render::streaming::loaders::StreamWorkers,
    // Which scene exclusively owns each deferred bucket, so scene residency
    // can claim it as a member.
    shader_deferral: ShaderDeferral,
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
    // The owned per-frame draw inputs `extract` fills from world state and
    // `submit` replays onto the backend. Held here so its buffers keep their
    // capacity across frames; taken out of `self` for the duration of one
    // step.
    snapshot: snapshot::RenderSnapshot,
    // Test-only injection seam: a fabricated GPU profile and a recording
    // backend factory, so unit tests can drive init and run_step without a
    // GPU device.
    #[cfg(test)]
    pub(crate) test_hooks: Option<tests::hooks::TestHooks>,
}

impl std::fmt::Debug for GraphicsSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphicsSystem")
            .field("frame_count", &self.frames.frames())
            .field("failed", &self.failed)
            .finish()
    }
}

impl GraphicsSystem {
    // The renderer, for a world that runs with a window.
    pub(crate) fn for_world(world: &World) -> Option<Self> {
        RenderMode::renders_in(world).then(|| Self::new(world.resource::<StateTree>()))
    }

    /// Fresh renderer driver with no backend yet, reading and writing under
    /// `tree`. The settings assets (frames in flight, clear color, shadow-map
    /// size, ...) and the launch's frame cap are read in [`System::init`].
    pub(crate) fn new(tree: Option<&StateTree>) -> Self {
        Self {
            state: tree.cloned(),
            // The schema's own default, so a world with no Background sees the
            // same color as one that declares an all-default component.
            clear_color: Background::default().color,
            failed: false,
            frames: FrameCounter::default(),
            frame_policy: FramePolicy::default(),
            extractor: FrameExtractor::default(),
            backend: None,
            scene_flow: None,
            loaded_fonts: text::FontSet::default(),
            sprite_texture_slots: overlay_maps::TextureSlots::new(),
            debug_hud_chips: Vec::new(),
            stat_hud_chips: Vec::new(),
            stream_pools: Default::default(),
            stream_workers: Default::default(),
            shader_deferral: ShaderDeferral::default(),
            clip_rects: overlay_maps::ClipRects::new(),
            // All-capable until the backend reports otherwise at init.
            caps: backend::DeviceCapabilities::ALL,
            snapshot: snapshot::RenderSnapshot::default(),
            #[cfg(test)]
            test_hooks: None,
        }
    }

    // The `assets/` a bare source filename is searched under: the running
    // world's own, or nothing for a world with no state tree (which leaves a
    // bare filename unresolved rather than searched from the cwd).
    pub(crate) fn assets_dir(&self) -> Option<std::path::PathBuf> {
        self.state.as_ref().map(StateTree::assets_dir)
    }

    // Detect the GPU performance profile for quality auto-config and what the
    // device can run. Probes the real device in production; a test-injected
    // profile takes its place so unit tests never create a GPU handle.
    fn detect_device(&self) -> backend::DeviceProbe {
        #[cfg(test)]
        if let Some(hooks) = &self.test_hooks {
            return backend::DeviceProbe {
                profile: hooks.gpu_profile,
                ..backend::DeviceProbe::UNKNOWN
            };
        }
        // A machine with no GPU at all never reaches here: the run resolved to
        // headless and this system was left out of the schedule.
        crate::platform::device::probe_device().unwrap_or(backend::DeviceProbe::UNKNOWN)
    }
}

impl System for GraphicsSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        if self.try_init(ctx).is_none() {
            self.failed = true;
        }
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        self.run_step(ctx)
    }
}

mod backend_handoff;
mod frame;
mod init;
mod screenshot;
mod skinned_templates;
mod streaming;
pub(crate) mod submit;
#[cfg(test)]
mod tests;
