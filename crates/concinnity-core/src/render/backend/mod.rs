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
//! Implementations are thin forwarders to the inherent methods on `MtlContext`
//! / `DxContext` / `VkContext`; concinnity-device generates the 1:1 ones from
//! a shared `forward!` macro, one invocation per family.

// Decals and particle emitters, added and removed while the world runs.
mod effects;
// Rewriting a built world's GPU content: the hot-reload rebuilders and the
// editor's live previews.
mod live_edit;
// What the backend reports about itself: capabilities, GPU class, counters,
// and the CPU-side readbacks.
mod probe;
// The skinned draw path, from one upload to each frame's joint palette.
mod skinned;
// Slot residency: the mesh, texture and chunk uploads a streaming world
// drives.
mod streaming;
// The live render settings: quality, shadows, post-process, lighting.
mod tuning;
// The presentation surface and its input, as the settings menu sees it.
mod window;

pub use effects::SceneEffects;
pub use live_edit::{DrawGeometryUpdate, LiveEdit, SkinnedDrawGeometryUpdate, SkinnedSlotLayout};
pub use probe::{
    BackendProbe, DeviceCapabilities, GpuClassInput, GpuProfile, GpuTier, GpuVendor,
    apple_family_from_device_name, classify_tier,
};
pub use skinned::SkinnedDraws;
pub use streaming::{ChunkMesh, DrawStreaming};
pub use tuning::{QualitySettings, RenderTuning};
pub use window::WindowControl;

use crate::gfx::render_types::{LineVertex, TextDrawCall};
use crate::render::error::RenderResult;
use crate::render::input::RenderInput;
use crate::render::scene_flow::SceneControl;

/// Per-frame inputs for [`RenderBackend::draw_frame`]. `world_hidden` is set when
/// an opaque menu backdrop covers the scene: the backend skips every world pass
/// and presents only the overlay (`text_calls`) over a cleared target.
#[derive(Clone, Copy)]
pub struct FrameParams<'a> {
    /// Seconds since the world started, for time-driven effects.
    pub elapsed: f32,
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    /// Near clip distance in world units.
    pub near: f32,
    /// Far clip distance in world units.
    pub far: f32,
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
}

/// The per-frame drive: what every graphics backend must implement for a world
/// to reach the screen.
///
/// Nothing here is defaulted, so a backend missing one of these fails to
/// compile rather than silently drawing nothing. The optional operation
/// families are the supertraits, each grouped by domain in its own module.
pub trait RenderBackend:
    SceneControl
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
    /// Confine the cursor to the window.
    fn capture_cursor(&mut self);
    /// Take the input sampled since the last call.
    fn take_input(&mut self) -> RenderInput;
    /// Block until the GPU has drained every submitted frame.
    fn wait_idle(&self);

    /// Per-frame drive. See [`FrameParams`] for the inputs.
    fn draw_frame(&mut self, params: FrameParams<'_>) -> RenderResult<()>;
    /// Push the camera's view matrix, column-major.
    fn update_view(&mut self, matrix: [[f32; 4]; 4]);

    /// Push this frame's changed model matrices, one `(draw slot, matrix)`
    /// entry per moved draw object, applied in order. Batched so the trait is
    /// crossed once per frame rather than once per entity; the caller sends
    /// only slots whose matrix actually changed. An out-of-range slot is
    /// ignored.
    fn update_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]);

    /// Retire a draw object: hide it from every pass (main, shadow, velocity)
    /// and exclude it from the ray-tracing acceleration structure, so a
    /// despawned entity's slot leaves no ghost. The slot's geometry buffers are
    /// untouched; the engine's draw-slot allocator returns the index to its
    /// free list so a later `clone_static_draw_object` can recycle it. A no-op
    /// if the index is out of range.
    fn retire_draw_object(&mut self, draw_idx: usize);
}

// A do-nothing backend used to exercise the provided (default) method bodies
// of every family without a GPU: the smallest valid bodies for the required
// methods, no defaults overridden. The empty impl blocks are the point of the
// split -- they are how a backend says it has none of that family. Shared by
// this module's tests and the ops tests.
#[cfg(test)]
pub(crate) mod test_stub {
    use super::*;

    pub(crate) struct StubBackend;

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
        fn update_models(&mut self, _updates: &[(u32, [[f32; 4]; 4])]) {}
        fn retire_draw_object(&mut self, _draw_idx: usize) {}
    }

    impl SkinnedDraws for StubBackend {
        fn upload_skinned(
            &mut self,
            _vertices: &[crate::gfx::mesh_payload::SkinnedVertex],
            _indices: &[u32],
            _draw_objects: alloc::vec::Vec<crate::gfx::render_types::SkinnedDrawObject>,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn update_skinned_pose(&mut self, _skinned_index: usize, _matrices: &[[[f32; 4]; 4]]) {}
    }

    impl DrawStreaming for StubBackend {
        fn evict_texture_slot(&mut self, _slot: usize) -> Result<(), alloc::string::String> {
            Ok(())
        }
        fn update_texture_slot(
            &mut self,
            _slot: usize,
            _image: &crate::bake::texture::TextureImage,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn evict_mesh(
            &mut self,
            _draw_idx: usize,
            _retire_frame: u64,
        ) -> Result<(), alloc::string::String> {
            Ok(())
        }
        fn upload_mesh(
            &mut self,
            _draw_idx: usize,
            _verts: &[crate::gfx::mesh_payload::Vertex],
            _idxs: &[u16],
            _frame: u64,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn setup_chunk_streaming(
            &mut self,
            _chunk_vtx_bytes: usize,
            _chunk_idx_bytes: usize,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn add_chunk_mesh(
            &mut self,
            _mesh: ChunkMesh<'_>,
            _dst: crate::render::draw_slot::SlotAlloc,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn remove_chunk_mesh(
            &mut self,
            _draw_idx: usize,
            _retire_frame: u64,
        ) -> Result<(), alloc::string::String> {
            Ok(())
        }
        fn set_chunk_model(
            &mut self,
            _draw_idx: usize,
            _model: [[f32; 4]; 4],
        ) -> Result<(), alloc::string::String> {
            Ok(())
        }
    }

    impl BackendProbe for StubBackend {}
    impl LiveEdit for StubBackend {}
    impl RenderTuning for StubBackend {}
    impl SceneEffects for StubBackend {}
    impl WindowControl for StubBackend {}
}

#[cfg(test)]
mod tests {
    use super::test_stub::StubBackend;
    use super::*;

    use crate::components::ShaderPrograms;
    use crate::gfx::profile::RenderStats;
    use crate::gfx::render_types::{MaterialUniforms, PostProcessTunables};
    use crate::render::backend_init::BackendInit;
    use crate::render::keymap::KeyMap;
    use alloc::vec;

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
        assert!(backend.draw_geometry_size(0).is_none());
        assert!(backend.draw_lod_index_counts(0).is_none());
    }

    #[test]
    fn default_mutators_are_noops_and_fallible_hooks_report_defaults() {
        let mut backend = StubBackend;

        // Runtime skinned-spawn fallbacks: nothing to reveal or hide.
        backend.reveal_skinned_instance(0, IDENTITY);
        backend.retire_skinned_draw_object(0);
        backend.update_skinned_models(&[(0, IDENTITY)]);

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
        backend.set_display_mode(crate::render::display_mode::DisplayMode {
            width: 1920,
            height: 1080,
            refresh_hz: 60,
        });

        // Live look + input tunable no-ops.
        backend.update_post_process(PostProcessTunables::DEFAULT);
        backend.set_ambient_intensity(1.0);
        backend.set_keymap(&KeyMap::default());
        backend.apply_quality_settings(stub_quality());
        backend.update_quality_params(stub_quality());
        backend.set_shadow_update(crate::components::ShadowUpdate::EveryFrame);
        backend.set_shadow_distance(200);
        backend.set_shadow_cascades(3);
        backend.update_fog_settings(None);
        backend.update_directional_lights(&[]);
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
        assert!(
            backend
                .clone_static_draw_object(
                    0,
                    IDENTITY,
                    crate::render::draw_slot::SlotAlloc::Append(0)
                )
                .is_err()
        );
        assert!(backend.add_decal(stub_decal()).is_err());
        assert!(backend.remove_decal(0).is_err());
        assert!(backend.add_emitter(stub_emitter()).is_err());
        assert!(backend.remove_emitter(0).is_err());
        assert!(
            backend
                .update_world_shader_pipelines(&ShaderPrograms::default())
                .is_err()
        );
    }

    // A minimal empty-world BackendInit borrowing `window`, for exercising the
    // default `reload_world`. Empty slices are `'static`; the only real borrow
    // is the window args.
    fn empty_backend_init(window: &crate::components::Window) -> BackendInit<'_> {
        BackendInit::minimal(window, alloc::vec::Vec::new())
    }

    #[test]
    fn default_reload_world_is_unsupported() {
        // A backend without a real reload path reports the swap unsupported, so
        // the caller falls back to a full rebuild.
        let mut backend = StubBackend;
        let window = crate::components::Window::default();
        assert!(backend.reload_world(empty_backend_init(&window)).is_err());
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
