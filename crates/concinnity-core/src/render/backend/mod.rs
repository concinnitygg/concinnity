//! The seam `GraphicsSystem` drives a graphics backend through, so the
//! per-frame step and the setup logic live in one cfg-free copy instead of
//! three.
//!
//! [`RenderBackend`] is the mandatory half: the window and input lifecycle,
//! the frame, and the pushes a frame is built from. It is required in full,
//! and it is the whole of what a backend must answer for a world to reach the
//! screen. Everything a backend can additionally do is grouped by domain into
//! one supertrait per file below, where the methods are defaulted: a default
//! is how a backend says it does not have that path, and the grouping is what
//! tells a new backend author which subset is which.
//!
//! A fallible default returns [`RenderError::Unsupported`](crate::render::error::RenderError::Unsupported),
//! never a silent `Ok`, unless doing nothing is the complete answer: a backend
//! without a morph path has no morph data to attach. An infallible setter
//! defaults to a no-op; a caller that must know whether one lands gates on
//! [`DeviceCapabilities`] first.
//!
//! Operations that only edit the CPU-side scene are defaulted over
//! [`SceneHost`], which the families that edit the scene require: the backend
//! lends its [`SceneState`](crate::render::scene_state::SceneState) and a
//! geometry writer, and the bookkeeping is written once. Operations that touch the
//! graphics API are thin forwarders to the inherent methods on `MtlContext` /
//! `DxContext` / `VkContext`; concinnity-device generates the 1:1 ones from a
//! shared `forward!` macro, one invocation per family.

// Decals and particle emitters, added and removed while the world runs.
mod effects;
// Rewriting a built world's GPU content: the hot-reload rebuilders and the
// editor's live previews.
mod live_edit;
#[cfg(any(test, feature = "test-support"))]
mod null;
// Building replacement pipelines on a thread other than the frame thread.
mod pipeline_builder;
// What the backend reports about itself: capabilities, GPU class, counters,
// and the CPU-side readbacks.
mod probe;
// The scene a backend lends to the defaults that only edit it.
mod scene_host;
// The skinned draw path, from one upload to each frame's joint palette.
mod skinned;
// Slot residency: the mesh, texture and chunk uploads a streaming world
// drives.
mod streaming;
// The live render settings: quality, shadows, post-process, lighting.
mod tuning;
// The presentation surface and its input, as the settings menu sees it.
mod window;

pub use crate::gfx::render_types::{DrawIndex, SkinnedIndex};
pub use effects::SceneEffects;
pub use live_edit::{
    DrawGeometryUpdate, LiveEdit, PipelineSwap, SkinnedDrawGeometryUpdate, SkinnedSlotLayout,
};
#[cfg(any(test, feature = "test-support"))]
pub use null::NullBackend;
pub use pipeline_builder::{PipelineBuilder, PreparedPipelines};
pub use probe::{
    BackendProbe, DeviceCapabilities, DeviceProbe, GpuClassInput, GpuProfile, GpuTier, GpuVendor,
    apple_family_from_device_name, classify_tier,
};
pub use scene_host::{GeometryEdit, SceneHost};
pub use skinned::SkinnedDraws;
pub use streaming::{ChunkMesh, DrawStreaming, PlanetChunk};
pub use tuning::RenderTuning;
pub use window::WindowControl;

use crate::gfx::render_types::{LineVertex, TextDrawCall};
use crate::input::snapshot::InputSnapshot;
use crate::render::error::RenderResult;
use crate::render::scene_flow::SceneControl;

/// Per-frame inputs for [`RenderBackend::draw_frame`]. `world_hidden` is set when
/// an opaque menu backdrop covers the scene: the backend skips every world pass
/// and presents only the overlay (`text_calls`) over a cleared target.
#[derive(Clone, Copy)]
pub struct FrameParams<'a> {
    /// Seconds since the world started, for time-driven effects. Kept in f64 so
    /// a backend can derive phases that stay exact over hours of uptime; the
    /// shaders take it rounded to f32.
    pub elapsed: f64,
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    /// Near clip distance in world units.
    pub near: f32,
    /// How far the camera sees in world units, or `None` for no limit: culls
    /// whole objects beyond it and caps the shadow distance.
    pub view_distance: Option<f32>,
    /// World-space camera position.
    pub cam_pos: [f32; 3],
    /// Overlay draw calls for this frame.
    pub text_calls: &'a [TextDrawCall],
    /// Expanded line ribbons (`lines::build_vertices`) for this frame's camera,
    /// drawn depth-tested into the scene after the world passes. Empty on any
    /// frame that submits no lines, which also drops the pass from the graph.
    pub lines: &'a [LineVertex],
    /// `true` when an opaque menu backdrop covers the scene.
    pub world_hidden: bool,
    /// Viewport view mode + show flags for the frame (`ViewOverrides` when the
    /// editor publishes one, defaults otherwise). Backends run their seeded
    /// graph inputs through `render_graph::apply_view` and steer the composite
    /// by the mode.
    pub view_mode: crate::gfx::view_modes::ViewMode,
    /// Feature passes to run this frame.
    pub show: crate::gfx::view_modes::ShowFlags,
    /// Rows of the rotation taking a world-space direction into the environment
    /// cubemaps' baked frame (`SkyOrientation::sample_rows`). The backend holds
    /// it for the frame and uploads it into every uniform block whose pass
    /// samples the sky, so the sky, the IBL and the reflections turn together.
    /// Identity in a world with no `SkyRotation`.
    pub sky_rot: [[f32; 4]; 3],
    /// The temporal history the previous frames accumulated no longer matches
    /// this one (see [`crate::render::history_reset`]): every temporal pass and
    /// upscaler starts over instead of reprojecting it.
    pub history_reset: bool,
    /// The world moved to another simulated frame since the last frame (a
    /// planet world following its camera): everything this frame draws is in
    /// the new frame, so every temporal pass carries what it remembers of the
    /// previous frame into it instead of reading the move as motion.
    pub rebase: Option<crate::planet::Rebase>,
    /// The characters and bodies that may be trampling grass this frame,
    /// stamped into the grass's bend field near the camera.
    pub grass_benders: &'a [crate::render::grass::GrassBender],
    /// Where the camera and the sun are about the planet, in a world with an
    /// `Atmosphere`; the backend draws the sky and the haze from it.
    pub atmosphere: Option<crate::render::atmosphere::AtmosphereFrame>,
    /// The authored intensity of the world's sun (its first directional
    /// light), which stands for full sunlight when the scene's luminance is
    /// read in absolute units.
    pub sunlight: f32,
    /// When the input this frame shows was sampled, in microseconds on the
    /// host's process-wide monotonic clock, or `None` when unknown. A backend
    /// that can tell when the frame's GPU work finished on that clock reports
    /// both as `RenderStats::gpu_done`.
    pub input_sampled_us: Option<u64>,
    /// Save the image this frame presents as a PNG at this path, or `None`.
    /// The copy rides the frame's own submission and the file is written off
    /// the frame path once the frame retires, so a capture costs the frame
    /// nothing it shows; [`RenderBackend::finish_captures`] waits the writes
    /// out. The outcome is logged.
    pub capture: Option<&'a str>,
}

/// The per-frame drive: what every graphics backend must implement for a world
/// to reach the screen.
///
/// The window, input and frame are required, so a backend missing one fails
/// to compile rather than silently drawing nothing. The scene pushes are
/// defaulted over [`SceneHost`]. The optional operation families are the
/// supertraits, each grouped by domain in its own module.
pub trait RenderBackend:
    SceneControl
    + SceneHost
    + BackendProbe
    + DrawStreaming
    + LiveEdit
    + RenderTuning
    + SceneEffects
    + SkinnedDraws
    + WindowControl
    + Send
{
    /// Whether the window has been asked to close, polled once a frame.
    fn window_closed(&mut self) -> bool;
    /// Ask the window to capture the cursor; a backend may defer until the first content click.
    fn request_cursor_capture(&mut self);
    /// Take the input sampled since the last call.
    fn take_input(&mut self) -> InputSnapshot;
    /// Block until the GPU has drained every submitted frame.
    fn wait_idle(&self);

    /// Block until the GPU has retired enough frames for the next one to be
    /// recorded, and reserve that frame's slot. A caller that samples input
    /// and simulates after this, right before `draw_frame`, keeps both out of
    /// the frames-in-flight wait. `draw_frame` takes the reserved slot instead
    /// of waiting itself; without a call it waits as before.
    fn prepare_frame(&mut self) {}
    /// Pump the window's pending events, so input taken next includes them.
    /// `draw_frame` pumps on its own; this is for a caller that samples input
    /// away from the draw.
    fn pump_events(&mut self) {}
    /// Per-frame drive. See [`FrameParams`] for the inputs.
    fn draw_frame(&mut self, params: FrameParams<'_>) -> RenderResult<()>;
    /// Block until every frame capture asked of [`Self::draw_frame`] so far is
    /// written to disk or has failed. Waits for the GPU first.
    fn finish_captures(&mut self) {}
    /// Push the camera's view matrix, column-major.
    fn update_view(&mut self, matrix: [[f32; 4]; 4]) {
        if let Some(scene) = self.scene_mut() {
            scene.update_view(matrix);
        }
    }

    /// Push this frame's changed model matrices, one `(draw slot, matrix)`
    /// entry per moved draw object, applied in order. Batched so the trait is
    /// crossed once per frame rather than once per entity; the caller sends
    /// only slots whose matrix actually changed. An out-of-range slot is
    /// ignored.
    fn update_models(&mut self, updates: &[(DrawIndex, [[f32; 4]; 4])]) {
        if let Some(scene) = self.scene_mut() {
            scene.update_models(updates);
        }
    }

    /// Set the planet ground draws are hidden behind once wholly past its
    /// horizon, or `None` for a world without one. Re-sent whenever the
    /// simulated frame moves, since the planet's center moves with it.
    fn set_horizon_occluder(&mut self, occluder: Option<crate::planet::HorizonOccluder>) {
        if let Some(scene) = self.scene_mut() {
            scene.horizon_occluder = occluder;
        }
    }

    /// Set the camera a planet's tiles morph by, which their vertices
    /// morph by, or `None` for a world without one. Re-sent whenever it or
    /// the simulated frame moves.
    fn set_tile_morph(&mut self, morph: Option<crate::planet::TileMorphFrame>) {
        if let Some(scene) = self.scene_mut() {
            scene.tile_morph = morph;
        }
    }

    /// Set where planet tile `draw` meets tiles of other levels.
    fn set_planet_tile_seams(&mut self, draw: DrawIndex, seams: crate::planet::TileSeams) {
        if let Some(scene) = self.scene_mut() {
            scene.set_planet_tile_seams(draw, seams);
        }
    }

    /// Retire a draw object: hide it from every pass (main, shadow, velocity)
    /// and exclude it from the ray-tracing acceleration structure, so a
    /// despawned entity's slot leaves no ghost. The slot's geometry buffers are
    /// untouched; the engine's draw-slot allocator returns the index to its
    /// free list so a later `clone_static_draw_object` can recycle it. A no-op
    /// if the index is out of range.
    fn retire_draw_object(&mut self, draw_idx: DrawIndex) {
        if let Some(scene) = self.scene_mut() {
            scene.retire_draw_object(draw_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NullBackend;
    use super::*;

    use crate::components::ShaderPrograms;
    use crate::gfx::render_types::MaterialUniforms;
    use crate::input::keymap::KeyMap;
    use crate::profile::RenderStats;
    use crate::render::backend_init::BackendInit;
    use crate::render::error::RenderError;
    use alloc::vec;

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    #[test]
    fn default_query_methods_report_conservative_values() {
        let backend = NullBackend;
        // Quality auto-config fails safe: the unknown/conservative profile.
        assert_eq!(backend.gpu_profile().tier, GpuTier::Unknown);
        assert_eq!(backend.gpu_profile().vendor, GpuVendor::Other);
        assert_eq!(backend.gpu_profile().memory_budget_bytes, 0);
        // Diagnostics a backend may leave to the default: zeroed here.
        assert_eq!(backend.logical_size(), (0.0, 0.0));
        // No chrome over the frame: UI anchored to the top starts at the top.
        assert_eq!(backend.top_content_inset(), 0.0);
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
        assert!(backend.draw_geometry_size(DrawIndex(0)).is_none());
        assert!(backend.draw_lod_index_counts(DrawIndex(0)).is_none());
        // No off-thread pipeline builds: a reload's swap builds for itself.
        assert!(backend.pipeline_builder().is_none());
    }

    #[test]
    fn default_mutators_are_noops_and_fallible_hooks_report_defaults() {
        let mut backend = NullBackend;

        // Without a scene the frame and skinned pushes have nowhere to land.
        backend.update_view(IDENTITY);
        backend.update_models(&[(DrawIndex(0), IDENTITY)]);
        backend.retire_draw_object(DrawIndex(0));
        backend.update_skinned_pose(SkinnedIndex(0), &[IDENTITY]);
        backend.update_morph_weights(SkinnedIndex(0), &[1.0]);
        backend.reveal_skinned_instance(SkinnedIndex(0), IDENTITY);
        backend.retire_skinned_draw_object(SkinnedIndex(0));
        backend.update_skinned_models(&[(SkinnedIndex(0), IDENTITY)]);
        // No morph deformation path: nothing to attach.
        assert!(backend.upload_skinned_morphs(vec![]).is_ok());

        // Streaming + cursor + capture no-ops.
        backend.seed_mesh_streaming(0, 0, 0, 0);
        backend.set_ui_cursor_hidden(true);
        backend.set_menu_mode(true);
        backend.set_camera_capture(true);
        backend.set_reflection_probes(&[]);

        // Presentation + window no-ops.
        backend.set_vsync(true);
        backend.set_window_mode(crate::components::WindowMode::Fullscreen);
        backend.set_window_size(1280, 720);
        backend.set_display_mode(crate::window::display_mode::DisplayMode {
            width: 1920,
            height: 1080,
            refresh_hz: 60,
        });

        // Live look + input tunable no-ops.
        backend.set_keymap(&KeyMap::default());
        let graphics = crate::settings::ResolvedGraphics::default();
        let mut plan =
            crate::settings::apply::ApplyPlan::live(crate::settings::apply::LiveChanges {
                post_process: true,
                ambient: true,
                shadow_cadence: true,
                feature_params: true,
                vsync: true,
            });
        assert!(backend.apply_settings(&graphics, &plan).is_ok());
        plan.rebuild
            .insert(crate::settings::apply::RebuildFeature::Ssao);
        assert!(matches!(
            backend.apply_settings(&graphics, &plan),
            Err(RenderError::Unsupported { .. })
        ));
        backend.update_fog_settings(None);
        backend.update_directional_lights(&[]);
        backend.set_draw_material(DrawIndex(0), MaterialUniforms::DEFAULT, 0, 0);
        backend.set_draw_cull_distance(DrawIndex(0), 50.0);

        // A bucket with no per-bucket pipeline to build is resident as-is.
        assert!(
            backend
                .install_world_shader(1, &crate::components::ShaderPrograms::default(), None)
                .is_ok()
        );
    }

    // Every fallible operation a bare backend does not implement reports
    // `Unsupported` naming itself, never a silent `Ok`.
    #[test]
    fn default_fallible_operations_report_unsupported() {
        use crate::render::error::RenderError;

        fn op<T>(result: RenderResult<T>) -> &'static str {
            match result {
                Err(RenderError::Unsupported { op }) => op,
                Err(e) => panic!("expected Unsupported, got {e}"),
                Ok(_) => panic!("expected Unsupported, got Ok"),
            }
        }

        let mut backend = NullBackend;
        let window = crate::components::Window::default();
        let cases = [
            (
                "update_color_lut",
                op(backend.update_color_lut(2, &[0u8; 32])),
            ),
            (
                "rebuild_static_geometry",
                op(backend.rebuild_static_geometry(vec![])),
            ),
            (
                "update_skinned_mesh_geometry",
                op(backend.update_skinned_mesh_geometry(SkinnedIndex(0), 0, &[], &[])),
            ),
            (
                "rebuild_skinned_geometry",
                op(backend.rebuild_skinned_geometry(vec![])),
            ),
            (
                "update_skinned_skeleton",
                op(backend.update_skinned_skeleton(SkinnedIndex(0), 0)),
            ),
            (
                "update_mesh_geometry",
                op(backend.update_mesh_geometry(DrawIndex(0), &[], &[], &[])),
            ),
            (
                "update_environment_map",
                op(backend.update_environment_map(&[])),
            ),
            (
                "update_world_shader",
                op(backend.update_world_shader(0, &ShaderPrograms::default(), None)),
            ),
            (
                "reload_world",
                op(backend.reload_world(empty_backend_init(&window))),
            ),
            ("screenshot", op(backend.screenshot("unused.png"))),
            ("read_cull_status", op(backend.read_cull_status())),
            (
                "clone_static_draw_object",
                op(backend.clone_static_draw_object(
                    DrawIndex(0),
                    IDENTITY,
                    crate::render::draw_slot::SlotAlloc::Append(DrawIndex(0)),
                )),
            ),
            (
                "upload_mesh",
                op(backend.upload_mesh(DrawIndex(0), &[], &[], 0)),
            ),
            ("evict_mesh", op(backend.evict_mesh(DrawIndex(0), 0))),
            (
                "add_chunk_mesh",
                op(backend.add_chunk_mesh(
                    ChunkMesh {
                        verts: &[],
                        idxs: &[],
                        model: IDENTITY,
                        texture_slot: 0,
                        normal_map_slot: 0,
                        material: MaterialUniforms::DEFAULT,
                        frame: 0,
                        planet_tile: None,
                        bounds: None,
                    },
                    crate::render::draw_slot::SlotAlloc::Append(DrawIndex(0)),
                )),
            ),
            (
                "remove_chunk_mesh",
                op(backend.remove_chunk_mesh(DrawIndex(0), 0)),
            ),
            (
                "set_chunk_model",
                op(backend.set_chunk_model(DrawIndex(0), IDENTITY)),
            ),
            ("add_decal", op(backend.add_decal(stub_decal()))),
            ("remove_decal", op(backend.remove_decal(0))),
            ("add_emitter", op(backend.add_emitter(stub_emitter()))),
            ("remove_emitter", op(backend.remove_emitter(0))),
        ];
        for (expected, reported) in cases {
            assert_eq!(reported, expected);
        }
    }

    // A minimal empty-world BackendInit borrowing `window`, for exercising the
    // default `reload_world`. Empty slices are `'static`; the only real borrow
    // is the window args.
    fn empty_backend_init(window: &crate::components::Window) -> BackendInit<'_> {
        BackendInit::minimal(window, alloc::vec::Vec::new())
    }

    fn stub_decal() -> crate::render::decal::DecalRecord {
        crate::render::decal::DecalRecord {
            model: IDENTITY,
            inv_model: IDENTITY,
            texture_slot: 0,
            tint: [1.0; 4],
        }
    }

    fn stub_emitter() -> crate::render::particles::ParticleEmitterRecord {
        crate::render::particles::ParticleEmitterRecord {
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
