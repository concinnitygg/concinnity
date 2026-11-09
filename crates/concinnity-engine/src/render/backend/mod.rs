//! The seam the renderer drives a graphics backend through, so the
//! per-frame step and the setup logic live in one cfg-free copy instead of
//! three.
//!
//! [`RenderBackend`] is the mandatory half: the frame, the pushes a frame is
//! built from, and the thread check. Everything else a backend does is split
//! by feature into one supertrait each, grouped into the files below, so a
//! backend implements each feature once, in the module that owns it. The
//! settings seam is [`SettingsTarget`], which lives in
//! `crate::settings::apply` with the settings it applies. Most methods are
//! defaulted: a default is how a backend says it does not have that path.
//!
//! A default describes a backend without the feature the operation serves,
//! by one rule:
//!
//! - A feed hands the backend state it draws from: lights, fog, probes,
//!   morph targets, a world shader's pipeline, the planet's shape. Without
//!   the feature there is nothing to draw it into, so the default does
//!   nothing and succeeds.
//! - A request asks for an outcome the caller reports or builds on: an id, a
//!   screenshot, readback, a geometry edit, a pipeline swap, a reload. Its
//!   default returns [`RenderError::Unsupported`], so "not done" is never
//!   mistaken for "done".
//! - A window hint (cursor, menu mode, vsync, window mode) returns nothing and
//!   defaults to doing nothing: the platform may refuse it anyway.
//!
//! A caller that must know whether a feed lands gates on
//! [`DeviceCapabilities`] first.
//!
//! Operations that only edit the CPU-side scene are defaulted over
//! [`SceneHost`], which the families that edit the scene require: the backend
//! lends its [`SceneState`](crate::render::scene_state::SceneState) and a
//! geometry writer, and the bookkeeping is written once.

// Decals and particle emitters, added and removed while the world runs.
mod effects;
// Rewriting a built world's geometry: the hot-reload rebuilders.
mod geometry_rebuild;
// The per-light tables a backend's GPU reads.
mod lights;
#[cfg(any(test, feature = "test-support"))]
mod null;
// Building replacement pipelines on a thread other than the frame thread.
mod pipeline_builder;
// What a backend keeps of the world's planet beyond its tiles.
mod planet;
// What the backend reports about itself: capabilities, GPU class, counters.
mod probe;
// The presented frame and the cull decisions, read back to the CPU.
mod readback;
#[cfg(any(test, feature = "test-support"))]
pub mod recording;
// Reloading shaders or a whole world on the live device.
mod reload;
// The scene a backend lends to the defaults that only edit it.
mod scene_host;
// The skinned draw path, from one upload to each frame's joint palette.
mod skinned;
// Slot residency: the mesh and chunk uploads a streaming world drives.
mod streaming;
// The texture pool, the color-grading LUT and the environment map.
mod textures;
// The presentation surface and its input, as the settings menu sees it.
mod window;
// The world Shader and SdfVolume pipelines.
pub mod error;
pub mod feature_plan;
pub mod init;
pub mod object_counts;
pub mod parallel_ctx;
pub mod pass_timing;
pub mod render_mode;
mod world_shaders;

pub use crate::gfx::render_types::{DrawIndex, SkinnedIndex};
pub use effects::{DecalPass, EmitterPass};
pub use geometry_rebuild::{
    DrawGeometryUpdate, GeometryBuffers, GeometryRebuild, SkinnedDrawGeometryUpdate,
    SkinnedSlotLayout,
};
pub use lights::LightTables;
#[cfg(any(test, feature = "test-support"))]
pub use null::NullBackend;
pub use pipeline_builder::{PipelineBuilder, PreparedPipelines};
pub use planet::PlanetReshape;
pub use probe::{
    AppleVariant, BackendProbe, DeviceCapabilities, DeviceProbe, GpuClassInput, GpuProfile,
    GpuTier, GpuVendor, apple_family_from_device_name, apple_variant_from_device_name,
    classify_tier,
};
pub use readback::{CullReadback, FrameCapture};
pub use reload::WorldReload;
pub use scene_host::{GeometryEdit, SceneHost};
pub use skinned::{SkinnedDraws, SkinnedFrame};
pub use streaming::{ChunkMesh, DrawStreaming, PlanetChunk};
pub use textures::TextureUploads;
pub use window::WindowControl;
pub use world_shaders::{PipelineSwap, SdfVolumeShaders, WorldShaders};

use crate::gfx::render_types::{LineVertex, MATERIAL_PARAM_COUNT, MaterialUniforms, TextDrawCall};
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::scene::flow::SceneControl;
use crate::settings::apply::SettingsTarget;

/// Per-frame inputs for [`RenderBackend::draw_frame`]. `world_hidden` is set when
/// an opaque menu backdrop covers the scene: the backend skips every world pass
/// and presents only the overlay (`text_calls`) over a cleared target.
#[derive(Clone, Copy)]
pub struct FrameParams<'a> {
    /// Seconds since the world started, for time-driven effects. Kept in f64 so
    /// a backend can derive phases that stay exact over hours of uptime; the
    /// shaders take it rounded to f32.
    pub elapsed: f64,
    /// The simulated instant the water is drawn at, in seconds: it stands
    /// still while the simulation does.
    pub water_time: f64,
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
    /// this one (see [`crate::render::temporal::history_reset`]): every temporal pass and
    /// upscaler starts over instead of reprojecting it.
    pub history_reset: bool,
    /// The world moved to another simulated frame since the last frame (a
    /// planet world following its camera): everything this frame draws is in
    /// the new frame, so every temporal pass carries what it remembers of the
    /// previous frame into it instead of reading the move as motion.
    pub rebase: Option<crate::environment::planet::Rebase>,
    /// The characters and bodies that may be trampling grass this frame,
    /// stamped into the grass's bend field near the camera.
    pub grass_benders: &'a [crate::render::environment::grass::GrassBender],
    /// What the floating bodies push aside this frame, which a planet sea's
    /// wake turns into waves near the camera.
    pub water_stirs: &'a [crate::environment::water::stir::WaterStir],
    /// The particle bursts systems asked for this frame, by emitter.
    pub particle_bursts: &'a [crate::protocol::ParticleBurst],
    /// Where the camera and the sun are about the planet, in a world with an
    /// `Atmosphere`; the backend draws the sky and the haze from it.
    pub atmosphere: Option<crate::render::environment::atmosphere::AtmosphereFrame>,
    /// A planet's per-point sky, which the views shading its ground and sea
    /// carry; off without one.
    pub daylight: &'a crate::render::uniforms::DaylightBlock,
    /// The authored intensity of the world's sun (its first directional
    /// light), which stands for full sunlight when the scene's luminance is
    /// read in absolute units.
    pub sunlight: f32,
    /// Where the world's prop reflection probe is this frame, if it has one:
    /// the backend renders one of its cube faces and keeps its record current.
    pub prop_probe: Option<crate::render::probe::prop::PropProbeFrame>,
    /// When the input this frame shows was sampled, in microseconds on the
    /// host's process-wide monotonic clock, or `None` when unknown. A backend
    /// that can tell when the frame's GPU work finished on that clock reports
    /// both as `RenderStats::gpu_done`.
    pub input_sampled_us: Option<u64>,
    /// Save the image this frame presents as a PNG at this path, or `None`.
    /// The copy rides the frame's own submission and the file is written off
    /// the frame path once the frame retires, so a capture costs the frame
    /// nothing it shows; [`FrameCapture::finish_captures`] waits the writes
    /// out. The outcome is logged.
    pub capture: Option<&'a str>,
}

/// The per-frame drive: what every graphics backend must implement for a world
/// to reach the screen.
///
/// The frame and the thread check are required, so a backend missing one
/// fails to compile rather than silently drawing nothing. The scene pushes are
/// defaulted over [`SceneHost`]. Everything else is a feature supertrait, each
/// in its own module, and the settings seam is [`SettingsTarget`] in
/// `crate::settings::apply`; the module docs give the rule their defaults
/// follow.
pub trait RenderBackend:
    SceneHost
    + BackendProbe
    + CullReadback
    + DecalPass
    + EmitterPass
    + DrawStreaming
    + FrameCapture
    + GeometryRebuild
    + PlanetReshape
    + SettingsTarget
    + LightTables
    + SdfVolumeShaders
    + SkinnedDraws
    + TextureUploads
    + WindowControl
    + WorldReload
    + WorldShaders
{
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
    ///
    /// # Errors
    ///
    /// Returns an error if recording, submitting, or presenting the frame
    /// fails.
    fn draw_frame(&mut self, params: FrameParams<'_>) -> RenderResult<()>;
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
    fn set_horizon_occluder(
        &mut self,
        occluder: Option<crate::environment::planet::HorizonOccluder>,
    ) {
        if let Some(scene) = self.scene_mut() {
            scene.horizon_occluder = occluder;
        }
    }

    /// Set the camera a planet's tiles morph by, which their vertices
    /// morph by, or `None` for a world without one. Re-sent whenever it or
    /// the simulated frame moves.
    fn set_tile_morph(&mut self, morph: Option<crate::environment::planet::TileMorphFrame>) {
        if let Some(scene) = self.scene_mut() {
            scene.view.tile_morph = morph;
        }
    }

    /// Set where planet tile `draw` meets tiles of other levels.
    fn set_planet_tile_seams(
        &mut self,
        draw: DrawIndex,
        seams: crate::gfx::render_types::TileSeams,
    ) {
        if let Some(scene) = self.scene_mut() {
            scene.set_planet_tile_seams(draw, seams);
        }
    }

    /// Show or hide one draw slot.
    fn update_visibility(&mut self, draw_idx: DrawIndex, visible: bool) {
        if let Some(scene) = self.scene_mut() {
            scene.update_visibility(draw_idx, visible);
        }
    }

    /// Fade the composited image to black by `fade` in `[0, 1]`.
    fn set_fade(&mut self, fade: f32) {
        if let Some(scene) = self.scene_mut() {
            scene.set_fade(fade);
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

    /// Rewrite a draw slot's material parameters + texture/normal-map pool
    /// indices in place. Driven by the editor's live draw seam when a Prop edits
    /// its `material` arg. The caller gates on
    /// [`DeviceCapabilities::rewrites_draws`], which a backend reports once its
    /// GPU records and ray-tracing geometry table follow the rewritten slot.
    fn set_draw_material(
        &mut self,
        draw_idx: DrawIndex,
        material: MaterialUniforms,
        texture_slot: usize,
        normal_map_slot: usize,
    ) {
        if let Some(scene) = self.scene_mut() {
            scene.set_draw_material(draw_idx, material, texture_slot, normal_map_slot);
        }
    }

    /// Replace one row of the material parameter table with a material's new
    /// `params`; `row` is [`crate::render::material_params::row_of`] of its
    /// handle. Every draw reading the row picks the values up on the next
    /// frame, as each of the backend's copies of the table comes round. Driven
    /// by the editor's live material seam. Out-of-range rows and the zero row
    /// are ignored.
    fn set_material_params(&mut self, row: u32, params: [f32; MATERIAL_PARAM_COUNT]) {
        if let Some(scene) = self.scene_mut() {
            scene.material_params.set(row, params);
        }
    }

    /// Supply the reflection-probe placements: declared `ReflectionProbe`
    /// assets, or a grid the caller auto-seeded over the scene. An empty list
    /// bakes none. The backend's bake places them on its next step, growing its
    /// cube array to a cube per placement (and one after theirs for the
    /// world's prop probe when `prop_probe`), and bakes them over the frames
    /// that follow; until a cube installs, its probe reflects the sky, as every
    /// probe does on a backend without a bake. Pushed once after construction.
    fn set_reflection_probes(
        &mut self,
        probes: &[crate::render::probe::capture::ProbePlacement],
        prop_probe: bool,
    ) {
        if let Some(scene) = self.scene_mut() {
            scene.probes.request(probes.to_vec(), prop_probe);
        }
    }

    /// Replace the live directional-light set (the sun), written into the
    /// scene's light block in place. Every reader, the cascade shadow
    /// direction and the fog sun included, takes the sun from it each frame.
    fn update_directional_lights(&mut self, lights: &[crate::components::DirectionalLight]) {
        if let Some(scene) = self.scene_mut() {
            scene.lights.set_directional(lights);
        }
    }

    /// Replace the spheres that keep part of the sky's light off their
    /// surroundings, written into the scene's light block in place.
    fn update_sky_occluders(
        &mut self,
        occluders: &crate::render::light::sky_occluders::SkyOccluders,
    ) {
        if let Some(scene) = self.scene_mut() {
            scene.lights.set_sky_occluders(occluders);
        }
    }

    /// Move the local lights (point, spot and rect area) in place: `lights` is
    /// `lights::build_light_data` over the same lights in the same order, so
    /// the count and every slot assignment are unchanged and only positions
    /// and directions differ. `uniforms` supplies the point array world shader
    /// hooks read; its other fields are ignored. This is how a world that
    /// moves to another simulated frame carries its lights. The GPU is drained
    /// before the backend's [`LightTables`] are rewritten, and the spot shadow
    /// slices re-render from their first refresh. Without a scene there are no
    /// lights to move, and this succeeds doing nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if the light count does not match the lights built at
    /// init, or if a table cannot be written.
    fn move_local_lights(
        &mut self,
        lights: &crate::render::light::lights::LightData,
        uniforms: &crate::gfx::render_types::LightUniforms,
    ) -> RenderResult<()> {
        let spot_slices = self.spot_slice_count();
        let Some(scene) = self.scene() else {
            return Ok(());
        };
        if !scene.lights.same_local_count(&lights.lights)
            || lights.spot_shadows.len() != spot_slices
        {
            return Err(RenderError::Other(
                "move_local_lights: the light count changed since init".into(),
            ));
        }
        self.wait_idle();
        self.rewrite_light_tables(lights)?;
        if let Some(scene) = self.scene_mut() {
            scene.planner.shadow.reset_spots();
            scene.lights.move_local(&lights.lights, uniforms);
        }
        Ok(())
    }

    /// Append a projected-decal record at runtime, returning a stable slot
    /// index the caller hands to [`Self::remove_decal`] later. Lets a
    /// gameplay system stamp bullet holes, footprints, or other ad-hoc
    /// decals after the world has built. The record lands in the scene's
    /// decal set, reusing a vacated slot before growing it, and the backend
    /// readies its [`DecalPass`] and binds the slot's albedo.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene or on a backend
    /// without the decal pass, or an error if the pass cannot be readied or
    /// the set is full.
    fn add_decal(&mut self, record: crate::render::decal::DecalRecord) -> RenderResult<usize> {
        // Asked first so a backend that builds its pass lazily builds nothing
        // for a scene-less add that is refused anyway.
        if self.scene().is_none() {
            return Err(RenderError::Unsupported { op: "add_decal" });
        }
        self.ensure_decal_pass()?;
        let texture_slot = record.texture_slot;
        let Some(scene) = self.scene_mut() else {
            return Err(RenderError::Unsupported { op: "add_decal" });
        };
        let capacity = scene.decals.capacity();
        let id = scene.decals.insert(record).map_err(|_| {
            RenderError::Other(alloc::format!(
                "add_decal: the decal set is full ({capacity} decals)"
            ))
        })?;
        self.bind_decal_texture(id, texture_slot);
        Ok(id)
    }

    /// Tombstone a runtime decal slot. The id returned by
    /// [`Self::add_decal`] becomes invalid; the next add may reuse it. What the
    /// pass draws with is kept, so a later add does not rebuild it.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if
    /// `decal_id` names no live decal.
    fn remove_decal(&mut self, decal_id: usize) -> RenderResult<()> {
        let Some(scene) = self.scene_mut() else {
            return Err(RenderError::Unsupported { op: "remove_decal" });
        };
        scene
            .decals
            .remove(decal_id)
            .map_err(|e| RenderError::Other(alloc::format!("remove_decal: id {decal_id} {e}")))?;
        self.release_decal_texture(decal_id);
        Ok(())
    }

    /// Append a particle-emitter record at runtime, returning a stable slot
    /// index the caller hands to [`Self::remove_emitter`] later. The record
    /// lands in the scene's particle system, reusing a vacated slot before
    /// growing it, and the backend readies its [`EmitterPass`] and allocates
    /// the emitter's pool, so the simulation begins ticking it on the next
    /// frame.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene or on a backend
    /// without the particle passes, or an error if the passes cannot be
    /// readied, the system is full, or the pool cannot be allocated.
    fn add_emitter(
        &mut self,
        record: crate::render::particles::ParticleEmitterRecord,
    ) -> RenderResult<usize> {
        // Asked first so a backend that builds its passes lazily builds
        // nothing for a scene-less add that is refused anyway.
        if self.scene().is_none() {
            return Err(RenderError::Unsupported { op: "add_emitter" });
        }
        self.ensure_emitter_pass()?;
        let Some(scene) = self.scene_mut() else {
            return Err(RenderError::Unsupported { op: "add_emitter" });
        };
        let id = scene.particles.insert(record).map_err(|_| {
            RenderError::Other(alloc::format!(
                "add_emitter: MAX_EMITTERS ({}) exceeded",
                crate::render::particles::MAX_EMITTERS
            ))
        })?;
        if let Err(e) = self.create_emitter(id, &record) {
            if let Some(scene) = self.scene_mut() {
                let removed = scene.particles.remove(id);
                debug_assert!(removed.is_ok(), "emitter {id} was just handed out");
            }
            return Err(e);
        }
        Ok(id)
    }

    /// Tombstone a runtime emitter slot and release its GPU pool. The id
    /// returned by [`Self::add_emitter`] becomes invalid; the next add may
    /// reuse it.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if
    /// `emitter_id` names no live emitter.
    fn remove_emitter(&mut self, emitter_id: usize) -> RenderResult<()> {
        let Some(scene) = self.scene_mut() else {
            return Err(RenderError::Unsupported {
                op: "remove_emitter",
            });
        };
        scene.particles.remove(emitter_id).map_err(|e| {
            RenderError::Other(alloc::format!("remove_emitter: id {emitter_id} {e}"))
        })?;
        self.destroy_emitter(emitter_id);
        Ok(())
    }

    /// Replace the live volumetric-fog settings, or turn the fog off with
    /// `None`. A world that started without fog has no fog passes to turn on,
    /// so a later `Some` is dropped and fog needs a relaunch (see
    /// [`SceneFog`](crate::render::environment::volumetric_fog::SceneFog)).
    fn update_fog_settings(
        &mut self,
        settings: Option<crate::render::environment::volumetric_fog::FogSettings>,
    ) {
        if let Some(scene) = self.scene_mut() {
            scene.fog.set(settings);
        }
    }

    /// Rewrite a draw slot's `cull_distance` in place. Driven by the editor's
    /// live draw seam when a Prop edits its `cull_distance` arg. Gated by the
    /// same [`DeviceCapabilities::rewrites_draws`] flag.
    fn set_draw_cull_distance(&mut self, draw_idx: DrawIndex, cull_distance: f32) {
        if let Some(scene) = self.scene_mut() {
            scene.set_draw_cull_distance(draw_idx, cull_distance);
        }
    }
}

// Scene flow drives a live backend through the trait object.
impl SceneControl for dyn RenderBackend + '_ {
    fn update_visibility(&mut self, draw_idx: DrawIndex, visible: bool) {
        RenderBackend::update_visibility(self, draw_idx, visible);
    }

    fn set_fade(&mut self, fade: f32) {
        RenderBackend::set_fade(self, fade);
    }
}

#[cfg(test)]
mod tests {
    use super::{NullBackend, *};

    use crate::components::ShaderPrograms;
    use crate::gfx::render_types::MaterialUniforms;
    use crate::profile::RenderStats;
    use crate::render::backend::init::BackendInit;
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
        let mut poses = crate::render::frame::snapshot::SpanBuffer::default();
        poses.push(SkinnedIndex(0), &[IDENTITY]);
        let mut morphs = crate::render::frame::snapshot::SpanBuffer::default();
        morphs.push(SkinnedIndex(0), &[1.0]);
        backend.update_skinned(SkinnedFrame {
            poses: &poses,
            morphs: &morphs,
            models: &[(SkinnedIndex(0), IDENTITY)],
        });
        backend.reveal_skinned_instance(SkinnedIndex(0), IDENTITY);
        backend.retire_skinned_draw_object(SkinnedIndex(0));
        // No morph deformation path: nothing to attach.
        assert!(backend.upload_skinned_morphs(&[]).is_ok());

        // Streaming + cursor + capture no-ops.
        assert!(backend.seed_mesh_streaming(0, 0, 0, 0).is_ok());
        backend.set_ui_cursor_hidden(true);
        backend.set_menu_mode(true);
        backend.set_camera_capture(true);
        backend.set_reflection_probes(&[], false);

        // Presentation + window no-ops.
        backend.set_vsync(true);
        backend.set_window_mode(crate::components::WindowMode::Fullscreen);
        backend.set_window_size(1280, 720);
        backend.set_display_mode(crate::presentation::window::display_mode::DisplayMode {
            width: 1920,
            height: 1080,
            refresh_hz: 60,
        });

        // Live look + input tunable no-ops.
        backend.update_fog_settings(None);
        backend.update_directional_lights(&[]);
        // Moving local lights is a feed like the directional push beside it.
        let lights = crate::render::light::lights::build_light_data(&[], &[], &[]);
        let uniforms = crate::render::light::lights::build_light_uniforms(
            &[],
            vec::Vec::new(),
            &lights.lights,
            1.0,
        );
        assert!(backend.move_local_lights(&lights, &uniforms).is_ok());
        backend.set_draw_material(DrawIndex(0), MaterialUniforms::DEFAULT, 0, 0);
        backend.set_draw_cull_distance(DrawIndex(0), 50.0);

        // A bucket with no per-bucket pipeline to build is resident as-is.
        assert!(
            backend
                .install_world_shader(1, &ShaderPrograms::default(), None)
                .is_ok()
        );
    }

    // Every fallible operation a bare backend does not implement reports
    // `Unsupported` naming itself, never a silent `Ok`.
    #[test]
    fn default_fallible_operations_report_unsupported() {
        use crate::render::backend::error::RenderError;

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
                    crate::render::slots::draw_slot::SlotAlloc::Append(DrawIndex(0)),
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
                    crate::render::slots::draw_slot::SlotAlloc::Append(DrawIndex(0)),
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
        BackendInit::minimal(
            concinnity_core::platform::Platform::Metal,
            window,
            vec::Vec::new(),
        )
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
            id: None,
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
            size_variance: 0.0,
            stretch: 0.0,
            color_start: [1.0; 4],
            color_end: [1.0; 4],
            lit: false,
            draw_after_transparent: false,
        }
    }
}
