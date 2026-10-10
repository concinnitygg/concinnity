//! Grouped construction inputs for the render backends, plus the requirements
//! derivation that trims scene-scoped features when a world has no 3D content.
//! GraphicsSystem init assembles a `BackendInit` from the drained world assets,
//! calls `resolve_requirements()`, and hands it to the backend constructor
//! selected at compile time (Metal / DirectX / Vulkan). Every backend receives
//! the same struct; each reads the fields its feature set consumes.

use crate::components::{
    GlassPanel, PassResolution, SdfVolume, ShaderPrograms, ShadowUpdate, UpscalerBackend,
    WaterSurface, Window,
};
use crate::gfx::auto_exposure::AutoExposureSettings;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{
    AreaLightData, DrawObject, GpuLight, GpuMaterialParams, InstancedCluster, LightUniforms,
    PostProcessTunables, SpotShadowData,
};
use crate::render::decal::DecalRecord;
use crate::render::dlss::DlssPreset;
use crate::render::particles::ParticleEmitterRecord;
use crate::render::post::rt_reflections::RtReflectionSettings;
use crate::render::post::ssao::settings::SsaoSettings;
use crate::render::post::ssgi::settings::SsgiSettings;
use crate::render::post::ssr::settings::SsrSettings;
use crate::render::rt_geom::RtDynamicMode;
use crate::render::volumetric_fog::FogSettings;
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

/// Shadow-mapping knobs from GraphicsConfig. `map_size == 0` disables the
/// shadow pipeline and cascade array entirely.
#[derive(Copy, Clone, Debug)]
pub struct ShadowParams {
    /// Shadow map edge in texels; 0 disables shadows entirely.
    pub map_size: u32,
    /// The live cascade schedule, seeded here at init.
    pub cadence: ShadowCadence,
}

/// Post-process and display settings resolved from PostProcessConfig (plus
/// the user's persisted overrides, the quality-preset ceiling, and the
/// launch's render requests). Every Option here is an init-time gate: None
/// allocates nothing.
pub struct PostSettings {
    /// Composite tunables pushed to the post pass. The backend pairs them with
    /// the display-output flags it negotiates below (`hdr_display` / `hdr_pq`)
    /// to build the uniform the shaders read.
    pub post_process: PostProcessTunables,
    /// Whether the temporal anti-aliasing pass runs.
    pub taa_enabled: bool,
    /// Sample count for the off-screen HDR color + depth attachments, resolved
    /// by [`crate::components::hdr_sample_count`] from the anti-aliasing mode
    /// and the upscaling request. `1` means no multisampling: the color target
    /// is the scene spine and no resolve step runs. Each backend clamps it to
    /// what the device reports for the HDR format.
    pub hdr_samples: u32,
    /// Screen-space ambient occlusion, or `None` when off.
    pub ssao: Option<SsaoSettings>,
    /// Screen-space reflections, or `None` when off.
    pub ssr: Option<SsrSettings>,
    /// Screen-space global illumination, or `None` when off.
    pub ssgi: Option<SsgiSettings>,
    /// Requires an RT-capable GPU; backends fall back to SSR without one.
    pub rt_reflections: Option<RtReflectionSettings>,
    /// How the ray-tracing acceleration structure tracks moving props. Inert
    /// when `rt_reflections` is None.
    pub rt_dynamic: RtDynamicMode,
    /// Whether skinned meshes join the ray-tracing acceleration structure.
    /// False leaves the BVH over static + instanced geometry only, so nothing
    /// animated appears in a ray-traced reflection.
    pub rt_skinned_geometry: bool,
    /// Per-axis divisor for the roughness-aware reflection blur target.
    pub reflection_blur_scale: u32,
    /// Auto-exposure, or `None` when off.
    pub auto_exposure: Option<AutoExposureSettings>,
    /// Authored exposure_ev carried as a bias on the adapted EV when
    /// auto-exposure is on; otherwise baked into post_process.exposure.
    pub auto_exposure_bias_ev: f32,
    /// HDR display request; each backend gates it on its own EDR / color-
    /// space capability probe and falls back to SDR with a warning.
    pub hdr_display: bool,
    /// PQ-encoded HDR output; honored by Metal today, accepted elsewhere.
    pub hdr_pq: bool,
    /// Whether temporal upscaling runs.
    pub temporal_upscaling: bool,
    /// Per-axis input-to-output ratio; ignored when upscaling is off.
    pub upscale_scale: f32,
    /// Upscaler selector for DirectX / Vulkan (FSR3 / DLSS / XeSS); Metal
    /// always uses MetalFX and ignores it.
    pub upscale_backend: UpscalerBackend,
    /// The render preset DLSS runs when it is the upscaler.
    pub dlss_preset: DlssPreset,
    /// Two-pass Hi-Z occlusion request; gated on the bindless cull path.
    pub occlusion_two_pass: bool,
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
    /// Transparent water surfaces; rendered by Metal today, accepted by the
    /// other backends for parity until their water ports land.
    pub water_surfaces: Vec<WaterSurface>,
    /// Refractive glass panels declared by the world.
    pub glass_panels: Vec<GlassPanel>,
    /// Raymarched SDF volumes, each paired with its compiled payload.
    pub sdf_volumes: Vec<SdfVolumeSource>,
    /// The grass field, or `None` when the world grows none.
    pub grass: Option<crate::render::grass::GrassField>,
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
/// as a world resource before start; GraphicsSystem init carries it into
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

/// Everything a backend constructor needs, assembled once by GraphicsSystem
/// init after the world's assets have been drained and settings resolved.
pub struct BackendInit<'a> {
    /// The window the backend opens.
    pub window: &'a Window,
    /// Debug-layer toggle for the DirectX / Vulkan validation layers.
    pub validation: bool,
    /// Frames the backend keeps in flight.
    pub frames_in_flight: usize,
    /// Whether presentation waits for vertical blank.
    pub vsync: bool,
    /// Linear RGBA the target is cleared to.
    pub clear_color: [f32; 4],
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
    /// Shadow-mapping settings.
    pub shadows: ShadowParams,
    /// Scene-sampler max anisotropy, clamped to the GPU's range at init.
    pub anisotropy: u32,
    /// Planar-reflection limits from the quality preset / GPU tier ceiling.
    pub planar: PlanarBudget,
    /// Post-process and display settings.
    pub post: PostSettings,
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
    /// `sky` says the world draws its environment map as the background.
    pub fn derive(scene: &SceneData, fx: &WorldFx, sky: bool) -> Self {
        let scene_present = sky
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
    pub fn minimal(window: &'a Window, text_atlases: Vec<(u32, u32, Vec<u8>)>) -> Self {
        let mut init = Self {
            window,
            validation: false,
            frames_in_flight: 2,
            vsync: true,
            clear_color: [0.0, 0.0, 0.0, 1.0],
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
            shadows: ShadowParams {
                map_size: 0,
                cadence: ShadowCadence {
                    update: ShadowUpdate::default(),
                    distance: 0,
                    cascades: 1,
                },
            },
            anisotropy: 1,
            planar: PlanarBudget::NONE,
            post: PostSettings {
                post_process: PostProcessTunables::DEFAULT,
                taa_enabled: false,
                hdr_samples: crate::components::HDR_MULTISAMPLE_COUNT,
                ssao: None,
                ssr: None,
                ssgi: None,
                rt_reflections: None,
                rt_dynamic: RtDynamicMode::Auto,
                rt_skinned_geometry: true,
                reflection_blur_scale: 1,
                auto_exposure: None,
                auto_exposure_bias_ev: 0.0,
                hdr_display: false,
                hdr_pq: false,
                temporal_upscaling: false,
                upscale_scale: 1.0,
                upscale_backend: UpscalerBackend::Auto,
                dlss_preset: DlssPreset::Default,
                occlusion_two_pass: false,
            },
            fx: WorldFx {
                decals: Vec::new(),
                particles: Vec::new(),
                fog: None,
                water_surfaces: Vec::new(),
                glass_panels: Vec::new(),
                sdf_volumes: Vec::new(),
                grass: None,
            },
            requirements: Default::default(),
        };
        init.resolve_requirements();
        init
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
            frames_in_flight: self.frames_in_flight.max(1),
            hdr_display: self.post.hdr_display,
            hdr_pq: self.post.hdr_pq,
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
            trim_scene_features(
                &mut self.shadows,
                &mut self.post,
                &mut self.fx,
                &mut self.planar,
            );
        }
        self.requirements = req;
    }
}

// Force off every feature that only decorates a 3D scene. All of these are
// existing init-time gates in the backends, so zeroing them here means every
// backend skips the matching resources with no backend-side changes.
fn trim_scene_features(
    shadows: &mut ShadowParams,
    post: &mut PostSettings,
    fx: &mut WorldFx,
    planar: &mut PlanarBudget,
) {
    shadows.map_size = 0;
    post.taa_enabled = false;
    post.ssao = None;
    post.ssr = None;
    post.ssgi = None;
    post.rt_reflections = None;
    post.auto_exposure = None;
    post.temporal_upscaling = false;
    post.occlusion_two_pass = false;
    fx.fog = None;
    planar.planes = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

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
            glass_panels: Vec::new(),
            sdf_volumes: Vec::new(),
            grass: None,
        }
    }

    fn full_post() -> PostSettings {
        PostSettings {
            post_process: PostProcessTunables::DEFAULT,
            taa_enabled: true,
            // Temporal: `hdr_sample_count` resolves both of this world's
            // temporal flags to a single-sample target.
            hdr_samples: 1,
            ssao: Some(SsaoSettings::resolve(0.5, 1.0)),
            ssr: None,
            ssgi: None,
            rt_reflections: None,
            rt_dynamic: RtDynamicMode::Auto,
            rt_skinned_geometry: true,
            reflection_blur_scale: 2,
            auto_exposure: None,
            auto_exposure_bias_ev: 0.0,
            hdr_display: false,
            hdr_pq: false,
            temporal_upscaling: true,
            upscale_scale: 0.5,
            upscale_backend: UpscalerBackend::Auto,
            dlss_preset: DlssPreset::Default,
            occlusion_two_pass: true,
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
    fn only_a_shown_environment_resolves_to_a_scene() {
        let window = Window::default();
        let ibl = [0u8; 4];
        let mut init = BackendInit::minimal(&window, Vec::new());
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
        let init = BackendInit::minimal(&window, atlas);

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
        assert_eq!(init.shadows.map_size, 0);
        assert!(!init.post.taa_enabled);
        assert!(init.post.ssao.is_none());
        assert_eq!(init.planar.planes, 0);
    }

    #[test]
    fn minimal_opens_its_own_window() {
        let window = Window::default();
        let init = BackendInit::minimal(&window, Vec::new());
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
        fx.water_surfaces.push(WaterSurface::default());
        assert!(RenderRequirements::derive(&scene, &fx, false).scene);

        let mut fx = empty_fx();
        fx.grass =
            crate::render::grass::GrassField::resolve(&[crate::components::Grass::default()], None);
        assert!(fx.grass.is_some());
        assert!(RenderRequirements::derive(&scene, &fx, false).scene);
    }

    #[test]
    fn sceneless_world_trims_scene_features() {
        let mut shadows = ShadowParams {
            map_size: 2048,
            cadence: ShadowCadence {
                update: ShadowUpdate::default(),
                distance: 120,
                cascades: 4,
            },
        };
        let mut post = full_post();
        let mut fx = empty_fx();
        let mut planar = PlanarBudget {
            planes: 3,
            resolution: PassResolution::Half,
        };
        trim_scene_features(&mut shadows, &mut post, &mut fx, &mut planar);
        assert_eq!(shadows.map_size, 0);
        assert!(!post.taa_enabled);
        assert!(post.ssao.is_none());
        assert!(!post.temporal_upscaling);
        assert!(!post.occlusion_two_pass);
        assert!(fx.fog.is_none());
        assert_eq!(planar.planes, 0);
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
