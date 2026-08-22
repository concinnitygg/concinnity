//! Grouped construction inputs for the render backends, plus the requirements
//! derivation that trims scene-scoped features when a world has no 3D content.
//! GraphicsSystem init assembles a `BackendInit` from the drained world assets,
//! calls `resolve_requirements()`, and hands it to the backend constructor
//! selected at compile time (Metal / DirectX / Vulkan). Every backend receives
//! the same struct; each reads the fields its feature set consumes.

use crate::assets::{GlassPanel, SdfVolume, ShadowUpdate, UpscalerBackend, WaterSurface, Window};
use crate::auto_exposure::AutoExposureSettings;
use crate::decal::DecalRecord;
use crate::mesh_payload::Vertex;
use crate::particles::ParticleEmitterRecord;
use crate::render_types::{
    AreaLightData, DrawObject, GpuLight, InstancedCluster, LightUniforms, PostProcessParams,
    SpotShadowData,
};
use crate::rt_reflections::RtReflectionSettings;
use crate::ssao::SsaoSettings;
use crate::ssgi::SsgiSettings;
use crate::ssr::SsrSettings;
use crate::volumetric_fog::FogSettings;

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
    /// frame. Honoured by DirectX + Vulkan; Metal's per-frame rebuild already
    /// covers chunks, so it needs no reserve.
    pub n_chunk_max: usize,
}

/// Compiled shader payloads. Each backend loads the format its toolchain
/// produced (metallib / DXBC / SPIR-V).
#[derive(Clone, Copy)]
pub struct ShaderBytes<'a> {
    /// Compiled main-pass vertex shader.
    pub vert: &'a [u8],
    /// Compiled main-pass fragment shader.
    pub frag: &'a [u8],
    /// Compiled shadow-pass vertex shader; consumed by DirectX / Vulkan. Metal
    /// compiles its shadow shader internally (shadow.metal) and ignores it.
    pub shadow: &'a [u8],
    /// Compiled GPU-instanced vertex shader; empty slice = no instanced
    /// pipeline (any InstancedProp in the world will fail to render).
    pub vert_instanced: &'a [u8],
    /// This entry's payload was not decoded because a scene other than the start
    /// scene owns it: the backend leaves the bucket's pipeline unbuilt and the
    /// streaming pump installs it when that scene pins. Distinct from empty stage
    /// bytes, which a backend whose built-in default ships no compiled payload
    /// (Vulkan's inline GLSL) also sees for an engine-default program.
    pub deferred: bool,
}

/// Decoded image payloads: texture pools, glyph atlases, and the serialised
/// IBL / grading payloads (None = the backend binds identity fallbacks).
pub struct MediaPayloads<'a> {
    /// Decoded textures for the shared handle-indexed pool: one `TextureImage`
    /// per slot carrying its GPU format and mip chain. Every texture -- albedo,
    /// normal map, emissive/ORM, terrain secondary -- lives here once at its
    /// handle; the backend appends a flat-normal fallback past the last entry for
    /// normal-less draws. RGBA8 images regenerate mips on upload; block-
    /// compressed images upload their chain verbatim.
    pub textures: &'a [crate::build::texture::TextureImage],
    /// Glyph atlas textures for text rendering; empty = no text support.
    pub text_atlases: Vec<(u32, u32, Vec<u8>)>,
    /// Serialised EnvironmentMap payload (irradiance + prefilter cubemaps).
    /// None disables IBL; the runtime binds 1x1 grey fallback cubes.
    pub env_map_bytes: Option<&'a [u8]>,
    /// Serialised ColorLut payload (3D grading LUT). None = identity LUT.
    pub color_lut_bytes: Option<&'a [u8]>,
}

/// Shadow-mapping knobs from GraphicsConfig. `map_size == 0` disables the
/// shadow pipeline and cascade array entirely.
#[derive(Copy, Clone, Debug)]
pub struct ShadowParams {
    /// Shadow map edge in texels; 0 disables shadows entirely.
    pub map_size: u32,
    /// Cascade re-render policy: hybrid amortizes far cascades across frames.
    pub update: ShadowUpdate,
    /// Shadow distance in world units, capped at the camera far plane by the
    /// per-frame cascade split.
    pub distance: u32,
    /// Cascade count (1..=4) the per-frame split + schedule render.
    pub cascades: u32,
}

/// Post-process and display settings resolved from PostProcessConfig (plus
/// the user's persisted overrides and the quality-preset ceiling). Every
/// Option here is an init-time gate: None allocates nothing.
pub struct PostSettings {
    /// Composite tunables pushed to the post pass.
    pub post_process: PostProcessParams,
    /// Whether the temporal anti-aliasing pass runs.
    pub taa_enabled: bool,
    /// Screen-space ambient occlusion, or `None` when off.
    pub ssao: Option<SsaoSettings>,
    /// Screen-space reflections, or `None` when off.
    pub ssr: Option<SsrSettings>,
    /// Screen-space global illumination, or `None` when off.
    pub ssgi: Option<SsgiSettings>,
    /// Requires an RT-capable GPU; backends fall back to SSR without one.
    pub rt_reflections: Option<RtReflectionSettings>,
    /// Per-axis divisor for the roughness-aware reflection blur target.
    pub reflection_blur_scale: u32,
    /// Auto-exposure, or `None` when off.
    pub auto_exposure: Option<AutoExposureSettings>,
    /// Authored exposure_ev carried as a bias on the adapted EV when
    /// auto-exposure is on; otherwise baked into post_process.exposure.
    pub auto_exposure_bias_ev: f32,
    /// HDR display request; each backend gates it on its own EDR / colour-
    /// space capability probe and falls back to SDR with a warning.
    pub hdr_display: bool,
    /// PQ-encoded HDR output; honoured by Metal today, accepted elsewhere.
    pub hdr_pq: bool,
    /// Whether temporal upscaling runs.
    pub temporal_upscaling: bool,
    /// Per-axis input-to-output ratio; ignored when upscaling is off.
    pub upscale_scale: f32,
    /// Upscaler selector for DirectX / Vulkan (FSR3 / DLSS / XeSS); Metal
    /// always uses MetalFX and ignores it.
    pub upscale_backend: UpscalerBackend,
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
    /// Raymarched SDF volumes as (volume, compiled fragment source bytes,
    /// asset label for error messages).
    pub sdf_volumes: Vec<(SdfVolume, Vec<u8>, String)>,
}

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
    /// The world's static geometry and draw lists.
    pub scene: SceneData<'a>,
    /// One entry per world Shader, indexed by the dense ShaderHandle value a
    /// DrawObject's `shader_bucket` carries; entry 0 is the world default
    /// program. Never empty for a rendering world.
    pub shaders: Vec<ShaderBytes<'a>>,
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
    /// Distinct planar-reflection plane budget from the quality preset / GPU
    /// tier ceiling; reflectors past it fall back to the probe cube.
    pub planar_planes: usize,
    /// Post-process and display settings.
    pub post: PostSettings,
    /// World-authored effect content.
    pub fx: WorldFx,
    /// Derived by `resolve_requirements()`; the conservative default assumes a
    /// full scene so a caller that skips resolution never under-allocates.
    pub requirements: RenderRequirements,
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
    /// meshes, streamed chunks, water, glass, SDF volumes, particles, or
    /// decals). False = the world renders UI / text only: the backend skips
    /// the scene pipelines and the frame collapses to a clear + composite.
    pub scene: bool,
}

impl Default for RenderRequirements {
    fn default() -> Self {
        RenderRequirements { scene: true }
    }
}

impl RenderRequirements {
    /// Derive the requirements a scene plus its effect content imposes.
    pub fn derive(scene: &SceneData, fx: &WorldFx) -> Self {
        let scene_present = !scene.vertices.is_empty()
            || !scene.draw_objects.is_empty()
            || !scene.instanced_clusters.is_empty()
            || scene.n_skinned > 0
            || scene.n_chunk_max > 0
            || !fx.water_surfaces.is_empty()
            || !fx.glass_panels.is_empty()
            || !fx.sdf_volumes.is_empty()
            || !fx.particles.is_empty()
            || !fx.decals.is_empty();
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
                text_atlases,
                env_map_bytes: None,
                color_lut_bytes: None,
            },
            light_uniforms: LightUniforms::DEFAULT,
            local_lights: Vec::new(),
            spot_shadows: Vec::new(),
            area_lights: Vec::new(),
            shadows: ShadowParams {
                map_size: 0,
                update: ShadowUpdate::default(),
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
                upscale_backend: UpscalerBackend::Auto,
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
        };
        init.resolve_requirements();
        init
    }

    /// The swapchain-level configuration this world needs. Compared against a
    /// transplanted backend's `RenderBackend::hot_swap_config` to decide whether
    /// a live SAVE can reuse the existing window (`reload_world`) or must rebuild.
    pub fn swapchain_config(&self) -> SwapchainConfig {
        SwapchainConfig {
            // Normalise to at least 1 to match how the backends size their ring
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
        let req = RenderRequirements::derive(&self.scene, &self.fx);
        if !req.scene {
            trim_scene_features(
                &mut self.shadows,
                &mut self.post,
                &mut self.fx,
                &mut self.planar_planes,
            );
            tracing::info!(
                "render requirements: no 3D scene content; scene-scoped features disabled"
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
    planar_planes: &mut usize,
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
    *planar_planes = 0;
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
        }
    }

    fn full_post() -> PostSettings {
        PostSettings {
            post_process: PostProcessParams::DEFAULT,
            taa_enabled: true,
            ssao: Some(SsaoSettings::resolve(0.5, 1.0)),
            ssr: None,
            ssgi: None,
            rt_reflections: None,
            reflection_blur_scale: 2,
            auto_exposure: None,
            auto_exposure_bias_ev: 0.0,
            hdr_display: false,
            hdr_pq: false,
            temporal_upscaling: true,
            upscale_scale: 0.5,
            upscale_backend: UpscalerBackend::Auto,
            occlusion_two_pass: true,
        }
    }

    #[test]
    fn text_only_world_derives_no_scene() {
        let req = RenderRequirements::derive(&empty_scene(), &empty_fx());
        assert!(!req.scene);
    }

    #[test]
    fn minimal_carries_only_a_window_and_its_atlases() {
        let window = Window::default();
        let atlas = vec![(2u32, 2u32, vec![255u8; 2 * 2 * 4])];
        let init = BackendInit::minimal(&window, atlas);

        // The text pipeline is the one thing it keeps: backends gate that pass
        // on a non-empty atlas list.
        assert_eq!(init.media.text_atlases.len(), 1);
        // One shader entry with empty stage bytes, so every backend resolves it
        // to its built-in default program rather than leaving bucket 0 unbuilt.
        assert_eq!(init.shaders.len(), 1);
        assert!(init.shaders[0].vert.is_empty());
        assert!(!init.shaders[0].deferred);
        // No scene content, so `resolve_requirements` ran and trimmed the
        // scene-scoped features.
        assert!(!init.requirements.scene);
        assert_eq!(init.shadows.map_size, 0);
        assert!(!init.post.taa_enabled);
        assert!(init.post.ssao.is_none());
        assert_eq!(init.planar_planes, 0);
    }

    #[test]
    fn any_scene_content_derives_scene() {
        let mut scene = empty_scene();
        scene.n_skinned = 1;
        assert!(RenderRequirements::derive(&scene, &empty_fx()).scene);

        let mut scene = empty_scene();
        scene.n_chunk_max = 8;
        assert!(RenderRequirements::derive(&scene, &empty_fx()).scene);

        // FX content alone is scene content too (a water-only world still
        // renders into the HDR scene chain).
        let scene = empty_scene();
        let mut fx = empty_fx();
        fx.water_surfaces.push(WaterSurface::default());
        assert!(RenderRequirements::derive(&scene, &fx).scene);
    }

    #[test]
    fn sceneless_world_trims_scene_features() {
        let mut shadows = ShadowParams {
            map_size: 2048,
            update: ShadowUpdate::default(),
            distance: 120,
            cascades: 4,
        };
        let mut post = full_post();
        let mut fx = empty_fx();
        let mut planar = 3usize;
        trim_scene_features(&mut shadows, &mut post, &mut fx, &mut planar);
        assert_eq!(shadows.map_size, 0);
        assert!(!post.taa_enabled);
        assert!(post.ssao.is_none());
        assert!(!post.temporal_upscaling);
        assert!(!post.occlusion_two_pass);
        assert!(fx.fog.is_none());
        assert_eq!(planar, 0);
    }

    #[test]
    fn scene_world_keeps_settings() {
        // A world with content must pass its resolved settings through
        // untouched: derivation flags the scene, and nothing is trimmed.
        let mut scene = empty_scene();
        scene.n_skinned = 2;
        let fx = empty_fx();
        let req = RenderRequirements::derive(&scene, &fx);
        assert!(req.scene);
    }
}
