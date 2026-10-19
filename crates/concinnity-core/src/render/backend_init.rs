//! Grouped construction inputs for the render backends, plus the requirements
//! derivation that trims scene-scoped features when a world has no 3D content.
//! Graphics init assembles a `BackendInit` from the drained world assets,
//! calls `resolve_requirements()`, and hands it to the constructor of the
//! backend `platform` names (Metal / DirectX / Vulkan). Every backend receives
//! the same struct; each reads the fields its feature set consumes.

use crate::components::{
    GlassPanel, PassResolution, SdfVolume, ShaderPrograms, ShadowUpdate, Window,
};
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{
    AreaLightData, DrawObject, GpuLight, GpuMaterialParams, InstancedCluster, LightUniforms,
    SpotShadowData,
};
use crate::platform::Platform;
use crate::render::backend::RenderBackend;
use crate::render::decal::DecalRecord;
use crate::render::particles::ParticleEmitterRecord;
use crate::render::rt_geom::RtDynamicMode;
use crate::render::volumetric_fog::FogSettings;
use crate::settings::ResolvedGraphics;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Static scene geometry and the draw lists built over it.
pub struct SceneData<'a> {
    /// The world's shared static vertex buffer.
    pub vertices: &'a [Vertex],
    /// The world's shared static index buffer.
    pub indices: &'a [u32],
    /// One draw record per static placement.
    pub draw_objects: Vec<DrawObject>,
    /// One record per instanced-prop cluster.
    pub instanced_clusters: Vec<InstancedCluster>,
    /// Skinned draw-object count (the world's `SkinnedMesh` count). Sizes each
    /// backend's shared GPU-cull buffers for the merged total (static +
    /// instances + skinned) at init; the skinned geometry itself is uploaded
    /// later via `upload_skinned`.
    pub n_skinned: usize,
    /// Worst-case resident chunk count for a streaming VoxelWorld (0
    /// otherwise). Reserves a chunk record region in the shared GPU-cull
    /// buffers at init; resident chunks fold into the indirect path each
    /// frame. Honored by DirectX + Vulkan; Metal's per-frame rebuild already
    /// covers chunks, so it needs no reserve.
    pub n_chunk_max: usize,
    /// The material parameter table's rows (see
    /// [`crate::render::material_params`]): the zero row, then one per
    /// material in handle order. Every draw's `params_index` addresses it.
    pub material_params: Vec<GpuMaterialParams>,
}

/// One world Shader as the backend receives it: the cook's compiled programs,
/// which the backend resolves per entry against the source it assembles (see
/// each backend's surface-source lookup), or nothing for the engine's own
/// program.
#[derive(Clone, Copy)]
pub struct WorldShader<'a> {
    /// The decoded payload; `None` for the engine's own main-pass program,
    /// which every backend compiles from its embedded source.
    pub programs: Option<&'a ShaderPrograms>,
    /// This entry's payload was not decoded because a scene other than the start
    /// scene owns it: the backend leaves the bucket's pipeline unbuilt and the
    /// streaming pump installs it when that scene pins.
    pub deferred: bool,
    /// A surface with a far band (`FarFieldTint::end > 0`) draws under this
    /// Shader, so its draws need the canopy (see
    /// [`crate::render::world_pipelines`]).
    pub canopy: bool,
}

/// Decoded image payloads: texture pools, glyph atlases, and the serialized
/// IBL / grading payloads (None = the backend binds identity fallbacks).
pub struct MediaPayloads<'a> {
    /// Decoded textures for the shared handle-indexed pool: one `TextureImage`
    /// per slot carrying its GPU format and mip chain. Every texture -- albedo,
    /// normal map, emissive/ORM, terrain secondary -- lives here once at its
    /// handle; the backend appends a flat-normal fallback past the last entry for
    /// normal-less draws. RGBA8 images regenerate mips on upload; block-
    /// compressed images upload their chain verbatim.
    pub textures: &'a [crate::bake::texture::TextureImage],
    /// Glyph atlas textures for text rendering; empty = no text support.
    pub text_atlases: Vec<(u32, u32, Vec<u8>)>,
    /// Serialized EnvironmentMap payload (irradiance + prefilter cubemaps).
    /// None disables IBL; the runtime binds 1x1 gray fallback cubes.
    pub env_map_bytes: Option<&'a [u8]>,
    /// Draw the environment map as the background behind all geometry (see
    /// [`crate::render::sky`]). Without a map the background is the clear
    /// color either way.
    pub env_map_background: bool,
    /// Serialized ColorLut payload (3D grading LUT). None = identity LUT.
    pub color_lut_bytes: Option<&'a [u8]>,
}

/// The cascade-shadow schedule the backend reads each frame. Unlike the shadow
/// map resolution, none of it sizes a GPU resource, so it can change live.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShadowCadence {
    /// Cascade re-render policy: hybrid amortizes far cascades across frames.
    pub update: ShadowUpdate,
    /// Shadow distance in world units, capped at the camera far plane by the
    /// per-frame cascade split.
    pub distance: u32,
    /// Cascade count (1..=4) the per-frame split + schedule render; the
    /// cascade array capacity stays 4.
    pub cascades: u32,
}

/// Shadow-mapping knobs from the `Shadows` asset. `map_size == 0` disables the
/// shadow pipeline and cascade array entirely.
#[derive(Copy, Clone, Debug)]
pub struct ShadowParams {
    /// Shadow map edge in texels; 0 disables shadows entirely.
    pub map_size: u32,
    /// The live cascade schedule, seeded here at init.
    pub cadence: ShadowCadence,
}

/// How the launch asks ray tracing to treat the scene. Not graphics settings:
/// the CLI picks them for one run.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RenderLaunch {
    /// How the ray-tracing acceleration structure tracks moving props. Inert
    /// without ray-traced reflections.
    pub rt_dynamic: RtDynamicMode,
    /// Whether skinned meshes join the ray-tracing acceleration structure.
    /// False leaves the BVH over static + instanced geometry only, so nothing
    /// animated appears in a ray-traced reflection.
    pub rt_skinned_geometry: bool,
}

impl Default for RenderLaunch {
    fn default() -> Self {
        Self {
            rt_dynamic: RtDynamicMode::Auto,
            rt_skinned_geometry: true,
        }
    }
}

/// World-authored effect content drained from components. Empty / None means
/// the backend builds no pipelines or pools for that feature.
pub struct WorldFx {
    /// Projected decals declared by the world.
    pub decals: Vec<DecalRecord>,
    /// Particle emitters declared by the world.
    pub particles: Vec<ParticleEmitterRecord>,
    /// Volumetric fog settings, or `None` when the world declares none.
    pub fog: Option<FogSettings>,
    /// Transparent water surfaces, each resolved against the world's wind.
    pub water_surfaces: Vec<crate::render::water::WaterBody>,
    /// The simulated oceans the large surfaces share.
    pub ocean: crate::render::water::ocean::OceanPlan,
    /// Refractive glass panels declared by the world.
    pub glass_panels: Vec<GlassPanel>,
    /// Raymarched SDF volumes, each paired with its compiled payload.
    pub sdf_volumes: Vec<SdfVolumeSource>,
    /// The grass field, or `None` when the world grows none.
    pub grass: Option<crate::render::grass::GrassField>,
    /// The atmosphere the sky is drawn from, or `None` when the world declares
    /// none.
    pub atmosphere: Option<crate::render::atmosphere::AtmosphereSettings>,
}

/// One raymarched SDF volume with the payload its pipelines build from.
pub struct SdfVolumeSource {
    /// The volume as authored.
    pub volume: SdfVolume,
    /// The compiled distance-field payload read from the blob.
    pub fragment_source: Vec<u8>,
    /// The volume's asset name, for error messages and pipeline labels.
    pub label: String,
}

/// A native view the host application owns (an `NSView` on macOS, a `UIView`
/// on iOS) that the backend renders into instead of opening a window. Inserted
/// as a world resource before start; graphics init carries it into
/// [`BackendInit::embedded_surface`].
#[derive(Clone, Copy, Debug)]
pub struct EmbeddedSurface {
    /// The host's view pointer, which must outlive the world.
    pub view: core::ptr::NonNull<core::ffi::c_void>,
    /// Drain the platform event queue during a step, for a host that runs no
    /// event loop of its own.
    pub pump_events: bool,
}

// SAFETY: the host keeps the view alive for the world's lifetime, and the
// engine only dereferences it on the main thread during backend construction.
// Moving the pointer between threads as a world resource never touches the view.
unsafe impl Send for EmbeddedSurface {}

/// Everything a backend constructor needs, assembled once by the renderer
/// init after the world's assets have been drained and settings resolved.
pub struct BackendInit<'a> {
    /// The graphics API the run renders with, one of the backends the build
    /// compiled in.
    pub platform: Platform,
    /// The window the backend opens.
    pub window: &'a Window,
    /// Debug-layer toggle for the DirectX / Vulkan validation layers.
    pub validation: bool,
    /// The resolved graphics settings the backend is built with.
    pub graphics: ResolvedGraphics,
    /// Sample count for the off-screen HDR color + depth attachments,
    /// [`ResolvedGraphics::hdr_samples`] before any trim. `1` means no
    /// multisampling: the color target is the scene spine and no resolve step
    /// runs. Each backend clamps it to what the device reports for the HDR
    /// format.
    pub hdr_samples: u32,
    /// The launch's ray-tracing requests.
    pub launch: RenderLaunch,
    /// True only under `cn debug`: disk-first shader resolution + watcher.
    pub hot_reload: bool,
    /// Keep the presented frame blit-readable so `screenshot` can capture it.
    /// On under the dev loop, and armed by `cn run --screenshot`; production
    /// otherwise pays nothing for it (Metal leaves the drawable
    /// framebuffer-only and retains nothing).
    pub capture: bool,
    /// A host-owned view to render into instead of opening a window, or `None`
    /// for a window of the backend's own.
    pub embedded_surface: Option<EmbeddedSurface>,
    /// The world's static geometry and draw lists.
    pub scene: SceneData<'a>,
    /// One entry per world Shader, indexed by the dense ShaderHandle value a
    /// DrawObject's `shader_bucket` carries; entry 0 is the world default
    /// program. Never empty for a rendering world.
    pub shaders: Vec<WorldShader<'a>>,
    /// Compiled media payloads (textures, fonts, environment maps).
    pub media: MediaPayloads<'a>,
    /// The fixed directional / point light arrays.
    pub light_uniforms: LightUniforms,
    /// Every local light (point + spot + area) for the clustered forward pass,
    /// uploaded to a per-scene GpuLight storage buffer. The first MAX_POINT_LIGHTS
    /// point lights are also mirrored into `light_uniforms.point` for the
    /// raymarch / fog / probe paths that still read the fixed array.
    pub local_lights: Vec<GpuLight>,
    /// One entry per spot shadow map slice, indexed by `GpuLight.shadow_index`.
    /// Empty when no spot light casts shadows, in which case the backend skips
    /// allocating the shadow array entirely.
    pub spot_shadows: Vec<SpotShadowData>,
    /// One entry per rectangular area light, indexed by `GpuLight.data_index`.
    /// Empty when the world declares none.
    pub area_lights: Vec<AreaLightData>,
    /// Planar-reflection limits from the quality preset / GPU tier ceiling.
    pub planar: PlanarBudget,
    /// World-authored effect content.
    pub fx: WorldFx,
    /// Derived by `resolve_requirements()`; the conservative default assumes a
    /// full scene so a caller that skips resolution never under-allocates.
    pub requirements: RenderRequirements,
}

/// How much planar reflection a backend allocates for: the number of distinct
/// mirror planes, and each mirror target's resolution relative to the render
/// resolution. Fixed at init, since the mirror targets are allocated there.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PlanarBudget {
    /// Distinct mirror planes; reflectors past it fall back to the probe cube.
    pub planes: usize,
    /// Mirror target resolution relative to the render resolution.
    pub resolution: PassResolution,
}

impl PlanarBudget {
    /// No mirror planes at all.
    pub const NONE: PlanarBudget = PlanarBudget {
        planes: 0,
        resolution: PassResolution::Full,
    };
}

/// The swapchain-level configuration a backend bakes into its window / surface
/// at construction: the ring depth and the HDR-output request that together fix
/// the drawable pixel format and frames-in-flight sizing. A live world swap
/// (`RenderBackend::reload_world`) can only reuse the existing window when these
/// are unchanged; a difference forces a full backend rebuild (a new window).
/// Kept small + `Eq` so the swap decision is one comparison.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct SwapchainConfig {
    /// Frames the backend keeps in flight.
    pub frames_in_flight: usize,
    /// Whether the swapchain requests an HDR pixel format.
    pub hdr_display: bool,
    /// Whether HDR output is PQ-encoded.
    pub hdr_pq: bool,
}

/// What the world's content requires of the renderer. Derived from the
/// assembled scene + fx data, backend-agnostic, so all three backends make
/// identical trimming decisions.
#[derive(Copy, Clone, Debug)]
pub struct RenderRequirements {
    /// True when any 3D scene content exists (meshes, instances, skinned
    /// meshes, streamed chunks, water, glass, SDF volumes, particles, decals,
    /// or an environment map drawn as the background). False = the world
    /// renders UI / text only: the backend skips the scene pipelines and the
    /// frame collapses to a clear + composite.
    pub scene: bool,
}

impl Default for RenderRequirements {
    fn default() -> Self {
        RenderRequirements { scene: true }
    }
}

impl RenderRequirements {
    /// Derive the requirements a scene plus its effect content imposes, where
    /// `sky` says the world draws its environment map as the background. An
    /// atmosphere is a sky too.
    pub fn derive(scene: &SceneData, fx: &WorldFx, sky: bool) -> Self {
        let scene_present = sky
            || fx.atmosphere.is_some()
            || !scene.vertices.is_empty()
            || !scene.draw_objects.is_empty()
            || !scene.instanced_clusters.is_empty()
            || scene.n_skinned > 0
            || scene.n_chunk_max > 0
            || !fx.water_surfaces.is_empty()
            || !fx.glass_panels.is_empty()
            || !fx.sdf_volumes.is_empty()
            || !fx.particles.is_empty()
            || !fx.decals.is_empty()
            || fx.grass.is_some();
        RenderRequirements {
            scene: scene_present,
        }
    }
}

impl<'a> BackendInit<'a> {
    /// A backend carrying nothing but a window and glyph atlases: no geometry,
    /// no textures, no lights, no effects. The single shader entry has empty
    /// stage bytes, which every backend resolves to its built-in default
    /// program. `resolve_requirements` then trims every scene-scoped feature.
    ///
    /// This is the startup error screen's path, which has to stand up a window
    /// with no compiled world data at all. Keeping it here means the field
    /// defaulting is maintained beside the struct it fills.
    pub fn minimal(
        platform: Platform,
        window: &'a Window,
        text_atlases: Vec<(u32, u32, Vec<u8>)>,
    ) -> Self {
        let mut init = Self {
            platform,
            window,
            validation: false,
            graphics: minimal_graphics(),
            hdr_samples: crate::components::HDR_MULTISAMPLE_COUNT,
            launch: RenderLaunch::default(),
            hot_reload: false,
            capture: false,
            embedded_surface: None,
            scene: SceneData {
                vertices: &[],
                indices: &[],
                draw_objects: Vec::new(),
                instanced_clusters: Vec::new(),
                n_skinned: 0,
                n_chunk_max: 0,
                material_params: Vec::new(),
            },
            shaders: vec![WorldShader {
                programs: None,
                deferred: false,
                canopy: false,
            }],
            media: MediaPayloads {
                textures: &[],
                text_atlases,
                env_map_bytes: None,
                env_map_background: true,
                color_lut_bytes: None,
            },
            light_uniforms: LightUniforms::DEFAULT,
            local_lights: Vec::new(),
            spot_shadows: Vec::new(),
            area_lights: Vec::new(),
            planar: PlanarBudget::NONE,
            fx: WorldFx {
                decals: Vec::new(),
                particles: Vec::new(),
                fog: None,
                water_surfaces: Vec::new(),
                ocean: crate::render::water::ocean::OceanPlan::new(Default::default()),
                glass_panels: Vec::new(),
                sdf_volumes: Vec::new(),
                grass: None,
                atmosphere: None,
            },
            requirements: Default::default(),
        };
        init.resolve_requirements();
        init
    }

    /// Whether a backend transplanted from another world can reload this one in
    /// place, keeping its window: only when it hot-swaps onto exactly this
    /// world's swapchain configuration. Otherwise it is rebuilt.
    pub fn reloads_into(&self, backend: &dyn RenderBackend) -> bool {
        backend
            .live_edit_ref()
            .and_then(|edit| edit.hot_swap_config())
            == Some(self.swapchain_config())
    }

    /// The swapchain-level configuration this world needs. Compared against a
    /// transplanted backend's `RenderBackend::hot_swap_config` to decide whether
    /// a live SAVE can reuse the existing window (`reload_world`) or must rebuild.
    pub fn swapchain_config(&self) -> SwapchainConfig {
        SwapchainConfig {
            // Normalize to at least 1 to match how the backends size their ring
            // buffers (e.g. Metal stores `frames_in_flight.max(1)`), so an
            // out-of-range authored 0 does not read as a swapchain change vs a
            // backend that already clamped it, spuriously forcing a full rebuild.
            frames_in_flight: self.graphics.frames_in_flight(),
            hdr_display: self.graphics.display.hdr,
            hdr_pq: self.graphics.display.pq,
        }
    }

    /// Derive the requirements from the assembled content and trim
    /// scene-scoped features accordingly. Runtime spawning can only clone
    /// assets already declared in the world, so the derivation here is
    /// complete: a world with no scene content at init can never grow one.
    pub fn resolve_requirements(&mut self) {
        let sky = self.media.env_map_bytes.is_some() && self.media.env_map_background;
        let req = RenderRequirements::derive(&self.scene, &self.fx, sky);
        if !req.scene {
            self.graphics.trim_scene_features();
            self.fx.fog = None;
            self.planar.planes = 0;
        }
        self.requirements = req;
    }
}

// The settings the startup error screen stands up with: two frames in flight,
// display sync, a black clear, no shadows and no anisotropy.
fn minimal_graphics() -> ResolvedGraphics {
    let mut graphics = ResolvedGraphics::default();
    graphics.frame_pacing.frames_in_flight = 2;
    graphics.frame_pacing.vsync = true;
    graphics.background.color = [0.0, 0.0, 0.0, 1.0];
    graphics.shadows.map_size = 0;
    graphics.texture_filtering.anisotropy = 1;
    graphics
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::WaterSurface;

    fn empty_scene() -> SceneData<'static> {
        SceneData {
            vertices: &[],
            indices: &[],
            draw_objects: Vec::new(),
            instanced_clusters: Vec::new(),
            n_skinned: 0,
            n_chunk_max: 0,
            material_params: Vec::new(),
        }
    }

    fn empty_fx() -> WorldFx {
        WorldFx {
            decals: Vec::new(),
            particles: Vec::new(),
            fog: None,
            water_surfaces: Vec::new(),
            ocean: crate::render::water::ocean::OceanPlan::new(Default::default()),
            glass_panels: Vec::new(),
            sdf_volumes: Vec::new(),
            grass: None,
            atmosphere: None,
        }
    }

    #[test]
    fn text_only_world_derives_no_scene() {
        let req = RenderRequirements::derive(&empty_scene(), &empty_fx(), false);
        assert!(!req.scene);
    }

    // A world that draws only its environment, behind a camera and nothing
    // else, still renders a 3D view of it.
    #[test]
    fn a_shown_environment_is_scene_content() {
        assert!(RenderRequirements::derive(&empty_scene(), &empty_fx(), true).scene);
    }

    // A map that lights the world but is not shown draws nothing on its own, so
    // with no props there is no scene; shown, it is the scene.
    #[test]
    fn a_backend_reloads_only_onto_its_own_swapchain_config() {
        use crate::render::backend::recording::{RecordingBackend, recording_backend};
        let window = Window::default();
        let init = BackendInit::minimal(Platform::Metal, &window, Vec::new());
        let (state, fresh) = recording_backend();
        assert!(!init.reloads_into(&fresh), "no hot-swap config");
        let same = RecordingBackend::transplant(state.clone(), Some(init.swapchain_config()));
        assert!(init.reloads_into(&same));
        let mut other = init.swapchain_config();
        other.hdr_display = !other.hdr_display;
        let other = RecordingBackend::transplant(state, Some(other));
        assert!(!init.reloads_into(&other));
    }

    #[test]
    fn only_a_shown_environment_resolves_to_a_scene() {
        let window = Window::default();
        let ibl = [0u8; 4];
        let mut init = BackendInit::minimal(Platform::Metal, &window, Vec::new());
        init.media.env_map_bytes = Some(&ibl);
        init.media.env_map_background = false;
        init.resolve_requirements();
        assert!(!init.requirements.scene);
        init.media.env_map_background = true;
        init.resolve_requirements();
        assert!(init.requirements.scene);
    }

    #[test]
    fn minimal_carries_only_a_window_and_its_atlases() {
        let window = Window::default();
        let atlas = vec![(2u32, 2u32, vec![255u8; 2 * 2 * 4])];
        let init = BackendInit::minimal(Platform::Metal, &window, atlas);

        // The text pipeline is the one thing it keeps: backends gate that pass
        // on a non-empty atlas list.
        assert_eq!(init.media.text_atlases.len(), 1);
        // One shader entry carrying no payload, so every backend resolves it
        // to its built-in default program rather than leaving bucket 0 unbuilt.
        assert_eq!(init.shaders.len(), 1);
        assert!(init.shaders[0].programs.is_none());
        assert!(!init.shaders[0].deferred);
        // No scene content, so `resolve_requirements` ran and trimmed the
        // scene-scoped features.
        assert!(!init.requirements.scene);
        assert_eq!(init.graphics.shadows.map_size, 0);
        assert!(!init.graphics.taa());
        assert!(init.graphics.ssao().is_none());
        assert_eq!(init.planar.planes, 0);
    }

    #[test]
    fn minimal_opens_its_own_window() {
        let window = Window::default();
        let init = BackendInit::minimal(Platform::Metal, &window, Vec::new());
        assert!(init.embedded_surface.is_none());
    }

    #[test]
    fn any_scene_content_derives_scene() {
        let mut scene = empty_scene();
        scene.n_skinned = 1;
        assert!(RenderRequirements::derive(&scene, &empty_fx(), false).scene);

        let mut scene = empty_scene();
        scene.n_chunk_max = 8;
        assert!(RenderRequirements::derive(&scene, &empty_fx(), false).scene);

        // FX content alone is scene content too (a water-only world still
        // renders into the HDR scene chain).
        let scene = empty_scene();
        let mut fx = empty_fx();
        fx.water_surfaces
            .push(crate::render::water::WaterBody::resolve(
                &WaterSurface::default(),
                &crate::render::wind::WindField::STILL,
            ));
        assert!(RenderRequirements::derive(&scene, &fx, false).scene);

        let mut fx = empty_fx();
        let terrain = crate::render::grass::GrassTerrain {
            center: [0.0; 3],
            grid: crate::terrain::TerrainGrid::new(4, [8.0, 8.0], alloc::vec![0.0; 25]).unwrap(),
            layers: alloc::vec![(crate::components::Grass::default(), None)],
        };
        fx.grass = crate::render::grass::GrassField::resolve(&[terrain], None);
        assert!(fx.grass.is_some());
        assert!(RenderRequirements::derive(&scene, &fx, false).scene);
    }

    // A world with every scene feature on loses each of them, keeps the
    // composite's FXAA, and keeps the HDR sample count it was resolved with.
    #[test]
    fn sceneless_world_trims_scene_features() {
        let window = Window::default();
        let mut init = BackendInit::minimal(Platform::Metal, &window, Vec::new());
        let mut graphics = ResolvedGraphics::default();
        graphics.anti_aliasing.upscaling = true;
        graphics.exposure.auto = true;
        init.hdr_samples = graphics.hdr_samples();
        init.graphics = graphics;
        init.planar = PlanarBudget {
            planes: 3,
            resolution: PassResolution::Half,
        };
        let fxaa = init.graphics.post_process().fxaa;
        init.resolve_requirements();
        let g = &init.graphics;
        assert_eq!(g.shadows.map_size, 0);
        assert!(!g.taa());
        assert!(g.ssao().is_none() && g.ssr().is_none() && g.ssgi().is_none());
        assert!(g.rt_reflections().is_none() && g.auto_exposure().is_none());
        assert!(!g.anti_aliasing.upscaling);
        assert!(!g.culling.two_pass_occlusion);
        assert_eq!(g.post_process().fxaa, fxaa);
        assert_eq!(init.hdr_samples, 1);
        assert_eq!(init.planar.planes, 0);
    }

    #[test]
    fn scene_world_keeps_settings() {
        // A world with content must pass its resolved settings through
        // untouched: derivation flags the scene, and nothing is trimmed.
        let mut scene = empty_scene();
        scene.n_skinned = 2;
        let fx = empty_fx();
        let req = RenderRequirements::derive(&scene, &fx, false);
        assert!(req.scene);
    }
}
