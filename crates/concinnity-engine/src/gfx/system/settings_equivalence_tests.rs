// Pins what the settings chain hands the backend for a matrix of inputs: every
// preset (Auto over each GPU tier), worlds authored above and below every
// ceiling, and the user's overrides on and off. Each case folds the resolved
// settings, the backend's `PostSettings` and `QualitySettings`, and a preset
// change's re-resolved quality group into one digest, so a change to any value
// the backend receives fails here by name.

use concinnity_core::components::{
    AaMode, AmbientOcclusion, AntiAliasing, Bloom, Culling, Display, Exposure, FramePacing,
    IndirectLighting, IndirectTechnique, PassResolution, Reflections, ShadowUpdate, Shadows,
    StreamingConfig, TextureFiltering, UpscaleQuality, UpscalerBackend, Vignette, WindowMode,
};
use concinnity_core::render::backend::{GpuProfile, GpuTier, QualitySettings};
use concinnity_core::render::backend_init::PostSettings;
use concinnity_core::render::dlss::DlssPreset;

use concinnity_core::settings::quality::resolve_ceiling;
use concinnity_core::settings::{
    QualityPreset, ResolvedSettings, UserGraphicsSettings, WorldAssets, WorldSettings,
};

use crate::app::run::LaunchRequest;
use crate::settings::system::SettingsState;

fn ceilings() -> Vec<(String, QualityPreset, GpuTier)> {
    let mut out = Vec::new();
    for tier in [
        GpuTier::Unknown,
        GpuTier::Integrated,
        GpuTier::EntryDiscrete,
        GpuTier::MidDiscrete,
        GpuTier::HighDiscrete,
    ] {
        out.push((format!("auto-{tier:?}"), QualityPreset::Auto, tier));
    }
    for preset in [
        QualityPreset::Low,
        QualityPreset::Medium,
        QualityPreset::High,
        QualityPreset::Ultra,
        QualityPreset::Custom,
    ] {
        out.push((format!("{preset:?}"), preset, GpuTier::Unknown));
    }
    out
}

// Every settings asset a world can declare, so a fixture names only what it
// changes from the defaults.
#[derive(Default)]
struct Declared {
    frame_pacing: FramePacing,
    shadows: Shadows,
    texture_filtering: TextureFiltering,
    ambient_occlusion: AmbientOcclusion,
    reflections: Reflections,
    indirect_lighting: IndirectLighting,
    anti_aliasing: AntiAliasing,
    exposure: Exposure,
    bloom: Bloom,
    vignette: Vignette,
    display: Display,
    culling: Culling,
}

impl Declared {
    // The world declaring every asset, and `streaming` where given.
    fn world(&self, streaming: Option<&StreamingConfig>) -> WorldSettings {
        WorldSettings::new(WorldAssets {
            frame_pacing: Some(&self.frame_pacing),
            shadows: Some(&self.shadows),
            texture_filtering: Some(&self.texture_filtering),
            ambient_occlusion: Some(&self.ambient_occlusion),
            reflections: Some(&self.reflections),
            indirect_lighting: Some(&self.indirect_lighting),
            anti_aliasing: Some(&self.anti_aliasing),
            exposure: Some(&self.exposure),
            bloom: Some(&self.bloom),
            vignette: Some(&self.vignette),
            display: Some(&self.display),
            culling: Some(&self.culling),
            lut_strength: None,
            streaming,
        })
    }
}

fn worlds() -> Vec<(&'static str, WorldSettings)> {
    let above = Declared {
        frame_pacing: FramePacing {
            frames_in_flight: 3,
            vsync: true,
            fps_cap: 0,
        },
        shadows: Shadows {
            map_size: 8192,
            update: ShadowUpdate::EveryFrame,
            distance: 500,
            cascades: 4,
        },
        texture_filtering: TextureFiltering { anisotropy: 16 },
        ambient_occlusion: AmbientOcclusion {
            enabled: true,
            ..Default::default()
        },
        reflections: Reflections {
            screen_space: true,
            ray_traced: true,
            ray_traced_shadows: true,
            ray_traced_resolution: PassResolution::Full,
            blur_resolution: PassResolution::Full,
            ..Default::default()
        },
        indirect_lighting: IndirectLighting {
            technique: IndirectTechnique::Ssgi,
            ssgi_resolution: PassResolution::Full,
            ssgi_rays: 4,
            ..Default::default()
        },
        anti_aliasing: AntiAliasing {
            mode: AaMode::Taa,
            upscaling: true,
            upscale_quality: UpscaleQuality::Quality,
            upscaler: UpscalerBackend::Dlss,
        },
        exposure: Exposure {
            auto: true,
            ..Default::default()
        },
        display: Display {
            hdr: true,
            pq: true,
        },
        culling: Culling {
            two_pass_occlusion: true,
        },
        ..Default::default()
    };
    let below = Declared {
        frame_pacing: FramePacing {
            frames_in_flight: 1,
            ..Default::default()
        },
        shadows: Shadows {
            map_size: 512,
            update: ShadowUpdate::Hybrid,
            distance: 10,
            cascades: 1,
        },
        texture_filtering: TextureFiltering { anisotropy: 1 },
        ambient_occlusion: AmbientOcclusion {
            enabled: false,
            ..Default::default()
        },
        reflections: Reflections {
            screen_space: false,
            ray_traced: false,
            ray_traced_shadows: false,
            ray_traced_resolution: PassResolution::Quarter,
            blur_resolution: PassResolution::Quarter,
            ..Default::default()
        },
        indirect_lighting: IndirectLighting {
            technique: IndirectTechnique::Ibl,
            ssgi_resolution: PassResolution::Quarter,
            ssgi_rays: 1,
            ..Default::default()
        },
        anti_aliasing: AntiAliasing {
            mode: AaMode::Off,
            upscale_quality: UpscaleQuality::UltraPerformance,
            ..Default::default()
        },
        culling: Culling {
            two_pass_occlusion: false,
        },
        ..Default::default()
    };
    let odd = Declared {
        frame_pacing: FramePacing {
            frames_in_flight: 7,
            ..Default::default()
        },
        shadows: Shadows {
            cascades: 9,
            ..Default::default()
        },
        texture_filtering: TextureFiltering { anisotropy: 64 },
        ambient_occlusion: AmbientOcclusion {
            intensity: 9.0,
            ..Default::default()
        },
        reflections: Reflections {
            max_distance: 5000.0,
            ..Default::default()
        },
        indirect_lighting: IndirectLighting {
            ambient_intensity: 40.0,
            ssgi_rays: 9,
            ..Default::default()
        },
        exposure: Exposure {
            ev: 30.0,
            auto: true,
            auto_min_ev: 30.0,
            auto_max_ev: -30.0,
            ..Default::default()
        },
        bloom: Bloom {
            intensity: -1.0,
            ..Default::default()
        },
        vignette: Vignette { strength: 3.0 },
        ..Default::default()
    };
    let streaming = StreamingConfig::default();
    vec![
        ("bare", WorldSettings::default()),
        ("defaults", Declared::default().world(Some(&streaming))),
        ("above", above.world(Some(&streaming))),
        ("below", below.world(None)),
        ("odd", odd.world(None)),
    ]
}

fn users() -> Vec<(&'static str, UserGraphicsSettings)> {
    let every = UserGraphicsSettings {
        quality_preset: Some(QualityPreset::Custom),
        vsync: Some(true),
        fps_cap: Some(60),
        perf_stats: Some(false),
        show_fps: Some(false),
        show_vram: Some(true),
        window_mode: Some(WindowMode::Borderless),
        resolution: Some([1920, 1080, 60]),
        render_scale: Some(UpscaleQuality::Balanced),
        upscale_backend: Some(UpscalerBackend::Fsr3),
        dlss_preset: Some(DlssPreset::K),
        exposure_ev: Some(1.5),
        bloom_intensity: Some(1.2),
        bloom_threshold: Some(2.0),
        bloom_knee: Some(0.25),
        vignette: Some(0.4),
        lut_strength: Some(0.5),
        ambient_intensity: Some(2.0),
        fov: Some(90.0),
        aa_mode: Some(AaMode::Taa),
        ssao: Some(true),
        ssr: Some(true),
        ray_traced_reflections: Some(true),
        rt_reflection_shadows: Some(false),
        ssgi: Some(true),
        auto_exposure: Some(false),
        ssgi_resolution: Some(PassResolution::Full),
        ssgi_rays: Some(4),
        reflection_blur_resolution: Some(PassResolution::Full),
        rt_reflection_resolution: Some(PassResolution::Half),
        ssao_radius: Some(1.5),
        ssao_intensity: Some(2.0),
        ssr_intensity: Some(0.4),
        ssr_max_distance: Some(120.0),
        ssgi_intensity: Some(1.5),
        ssgi_max_distance: Some(20.0),
        auto_exposure_min_ev: Some(-4.0),
        auto_exposure_max_ev: Some(12.0),
        auto_exposure_speed: Some(3.0),
        shadow_map_size: Some(2048),
        shadow_update: Some(ShadowUpdate::EveryFrame),
        shadow_distance: Some(320),
        shadow_cascades: Some(3),
        anisotropy: Some(8),
        temporal_upscaling: Some(true),
        hdr_display: Some(false),
        hdr_pq: Some(true),
        frames_in_flight: Some(9),
        occlusion_two_pass: Some(true),
        texture_cap: Some(512),
        texture_budget: Some(8),
    };
    let off = UserGraphicsSettings {
        aa_mode: Some(AaMode::Off),
        ssao: Some(false),
        ssr: Some(false),
        ray_traced_reflections: Some(false),
        ssgi: Some(false),
        auto_exposure: Some(true),
        shadow_map_size: Some(0),
        exposure_ev: Some(-40.0),
        ssao_radius: Some(0.0),
        frames_in_flight: Some(0),
        ..Default::default()
    };
    vec![
        ("none", UserGraphicsSettings::default()),
        ("every", every),
        ("off", off),
    ]
}

// The values the settings chain resolves, in a shape independent of the
// structs that carry them. The quality group prints as the single config the
// pinned digests were taken over: the resolved effects, and the world's values
// for the fields no layer above it moves.
fn resolved_text(g: &ResolvedSettings, world: &WorldSettings) -> String {
    format!(
        "{:?}",
        (
            (
                &PostProcessConfig::of(&g.quality.effects, world),
                g.quality.render_scale,
                g.quality.shadow_map_size,
                g.quality.shadow_cadence,
                g.quality.anisotropy,
            ),
            g.post_process,
            g.ambient_intensity,
            (g.vsync, g.fps_cap, g.perf_stats, g.show_fps, g.show_vram),
            g.frames_in_flight,
            (g.hdr_display, g.hdr_pq, g.temporal_upscaling),
            (g.upscale_backend, g.dlss_preset, g.occlusion_two_pass),
            (g.texture_cap, g.texture_budget),
        )
    )
}

// The quality group in the field order the digests were pinned in.
#[derive(Debug)]
#[expect(dead_code, reason = "the fields are read only through `Debug`")]
struct PostProcessConfig {
    bloom_intensity: f32,
    bloom_threshold: f32,
    bloom_knee: f32,
    exposure_ev: f32,
    vignette_strength: f32,
    lut_strength: f32,
    aa_mode: AaMode,
    ssao: bool,
    ssao_radius: f32,
    ssao_intensity: f32,
    ssr: bool,
    ssr_intensity: f32,
    ssr_max_distance: f32,
    ray_traced_reflections: bool,
    rt_reflection_resolution: PassResolution,
    rt_reflection_shadows: bool,
    reflection_blur_resolution: PassResolution,
    indirect_lighting: IndirectTechnique,
    ambient_intensity: f32,
    ssgi_intensity: f32,
    ssgi_max_distance: f32,
    ssgi_resolution: PassResolution,
    ssgi_rays: u32,
    auto_exposure: bool,
    auto_exposure_min_ev: f32,
    auto_exposure_max_ev: f32,
    auto_exposure_speed: f32,
    hdr_display: bool,
    hdr_pq: bool,
    temporal_upscaling: bool,
    upscale_quality: UpscaleQuality,
    upscale_backend: UpscalerBackend,
    occlusion_two_pass: bool,
}

impl PostProcessConfig {
    fn of(effects: &concinnity_core::settings::QualityEffects, world: &WorldSettings) -> Self {
        let ao = &effects.ambient_occlusion;
        let r = &effects.reflections;
        let gi = &effects.indirect_lighting;
        let aa = &effects.anti_aliasing;
        let ex = &effects.exposure;
        Self {
            bloom_intensity: world.bloom.intensity,
            bloom_threshold: world.bloom.threshold,
            bloom_knee: world.bloom.knee,
            exposure_ev: ex.ev,
            vignette_strength: world.vignette.strength,
            lut_strength: world.lut_strength,
            aa_mode: aa.mode,
            ssao: ao.enabled,
            ssao_radius: ao.radius,
            ssao_intensity: ao.intensity,
            ssr: r.screen_space,
            ssr_intensity: r.intensity,
            ssr_max_distance: r.max_distance,
            ray_traced_reflections: r.ray_traced,
            rt_reflection_resolution: r.ray_traced_resolution,
            rt_reflection_shadows: r.ray_traced_shadows,
            reflection_blur_resolution: r.blur_resolution,
            indirect_lighting: gi.technique,
            ambient_intensity: gi.ambient_intensity,
            ssgi_intensity: gi.ssgi_intensity,
            ssgi_max_distance: gi.ssgi_max_distance,
            ssgi_resolution: gi.ssgi_resolution,
            ssgi_rays: gi.ssgi_rays,
            auto_exposure: ex.auto,
            auto_exposure_min_ev: ex.auto_min_ev,
            auto_exposure_max_ev: ex.auto_max_ev,
            auto_exposure_speed: ex.auto_speed,
            hdr_display: world.display.hdr,
            hdr_pq: world.display.pq,
            temporal_upscaling: aa.upscaling,
            upscale_quality: aa.upscale_quality,
            upscale_backend: aa.upscaler,
            occlusion_two_pass: world.culling.two_pass_occlusion,
        }
    }
}

fn post_settings_text(p: &PostSettings) -> String {
    format!(
        "{:?}",
        (
            (p.post_process, p.taa_enabled, p.hdr_samples),
            (p.ssao, p.ssr, p.ssgi, p.rt_reflections),
            (p.rt_dynamic, p.rt_skinned_geometry, p.reflection_blur_scale),
            (p.auto_exposure, p.auto_exposure_bias_ev),
            (
                p.hdr_display,
                p.hdr_pq,
                p.temporal_upscaling,
                p.upscale_scale
            ),
            (p.upscale_backend, p.dlss_preset, p.occlusion_two_pass),
        )
    )
}

fn quality_text(q: &QualitySettings) -> String {
    format!(
        "{:?}",
        (
            q.taa,
            q.ssao,
            q.ssr,
            q.rt_reflections,
            q.ssgi,
            q.reflection_blur_scale,
            q.auto_exposure,
            q.auto_exposure_bias_ev,
        )
    )
}

// The user's choices after a preset change: every preset-governed override
// cleared, the way the settings menu clears them.
fn preset_cleared(user: &UserGraphicsSettings) -> UserGraphicsSettings {
    use concinnity_core::settings::rows::{QUALITY_CYCLES, QUALITY_TOGGLES};
    let mut user = user.clone();
    for row in &QUALITY_TOGGLES {
        *(row.persisted.get_mut)(&mut user) = None;
    }
    for row in &QUALITY_CYCLES {
        (row.clear)(&mut user);
    }
    user.shadow_map_size = None;
    user.shadow_update = None;
    user.shadow_distance = None;
    user.shadow_cascades = None;
    user.anisotropy = None;
    user.render_scale = None;
    user
}

fn case_text(
    world: &WorldSettings,
    user: &UserGraphicsSettings,
    preset: QualityPreset,
    tier: GpuTier,
) -> String {
    let profile = GpuProfile {
        tier,
        ..GpuProfile::UNKNOWN
    };
    let ceiling = resolve_ceiling(preset, &profile);
    let launch = LaunchRequest {
        dlss_preset: Some(DlssPreset::M),
        ..LaunchRequest::default()
    };
    // The settings state as init leaves it, then a preset change.
    let mut state = SettingsState::new();
    state.authored = world.clone();
    state.session = launch.session_overrides();
    let graphics = state.resolve(user, &ceiling);
    let post = super::init::post_settings(&graphics, world, &launch);
    let quality = graphics.quality_settings();

    let cleared = preset_cleared(user);
    let mut after_preset = graphics.clone();
    after_preset.set_quality(state.resolve_quality(&cleared, &ceiling));
    format!(
        "{}\n{}\n{}\n{}",
        resolved_text(&graphics, world),
        post_settings_text(&post),
        quality_text(&quality),
        resolved_text(&after_preset, world),
    )
}

fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn digests() -> Vec<(String, u64)> {
    let mut out = Vec::new();
    for (world_name, world) in worlds() {
        for (user_name, user) in users() {
            for (ceiling_name, preset, tier) in ceilings() {
                let text = case_text(&world, &user, preset, tier);
                out.push((
                    format!("{world_name}/{user_name}/{ceiling_name}"),
                    fnv1a(&text),
                ));
            }
        }
    }
    out
}

// Every case's digest as the settings chain produced it before resolution moved
// into `concinnity_core::settings`.
const EXPECTED: &[(&str, u64)] = &[
    ("bare/none/auto-Unknown", 0xeed2d2a0065b7e55),
    ("bare/none/auto-Integrated", 0xafab80de76ba38ae),
    ("bare/none/auto-EntryDiscrete", 0xeed2d2a0065b7e55),
    ("bare/none/auto-MidDiscrete", 0x7ec1dfc19d5657a5),
    ("bare/none/auto-HighDiscrete", 0x4431b6499071d6dd),
    ("bare/none/Low", 0xafab80de76ba38ae),
    ("bare/none/Medium", 0xeed2d2a0065b7e55),
    ("bare/none/High", 0x7ec1dfc19d5657a5),
    ("bare/none/Ultra", 0x4431b6499071d6dd),
    ("bare/none/Custom", 0x4431b6499071d6dd),
    ("bare/every/auto-Unknown", 0xd99fae68c068fd36),
    ("bare/every/auto-Integrated", 0x4862b65edb6faeb6),
    ("bare/every/auto-EntryDiscrete", 0xd99fae68c068fd36),
    ("bare/every/auto-MidDiscrete", 0xeaf6dab1088a8fcf),
    ("bare/every/auto-HighDiscrete", 0xeed7bfa8ae7274c2),
    ("bare/every/Low", 0x4862b65edb6faeb6),
    ("bare/every/Medium", 0xd99fae68c068fd36),
    ("bare/every/High", 0xeaf6dab1088a8fcf),
    ("bare/every/Ultra", 0xeed7bfa8ae7274c2),
    ("bare/every/Custom", 0xeed7bfa8ae7274c2),
    ("bare/off/auto-Unknown", 0x2646fa0a47182f3d),
    ("bare/off/auto-Integrated", 0x9737970484e06773),
    ("bare/off/auto-EntryDiscrete", 0x2646fa0a47182f3d),
    ("bare/off/auto-MidDiscrete", 0x3166d7332373aeec),
    ("bare/off/auto-HighDiscrete", 0xad560a00ae0b4581),
    ("bare/off/Low", 0x9737970484e06773),
    ("bare/off/Medium", 0x2646fa0a47182f3d),
    ("bare/off/High", 0x3166d7332373aeec),
    ("bare/off/Ultra", 0xad560a00ae0b4581),
    ("bare/off/Custom", 0xad560a00ae0b4581),
    ("defaults/none/auto-Unknown", 0x63890111c2c1be85),
    ("defaults/none/auto-Integrated", 0xd3406cc11cd5a35f),
    ("defaults/none/auto-EntryDiscrete", 0x63890111c2c1be85),
    ("defaults/none/auto-MidDiscrete", 0x0904c232147e916e),
    ("defaults/none/auto-HighDiscrete", 0x10f2abb9d5d42ba0),
    ("defaults/none/Low", 0xd3406cc11cd5a35f),
    ("defaults/none/Medium", 0x63890111c2c1be85),
    ("defaults/none/High", 0x0904c232147e916e),
    ("defaults/none/Ultra", 0x10f2abb9d5d42ba0),
    ("defaults/none/Custom", 0x10f2abb9d5d42ba0),
    ("defaults/every/auto-Unknown", 0x987d1ade87899a29),
    ("defaults/every/auto-Integrated", 0x4e0c958ff6642ea9),
    ("defaults/every/auto-EntryDiscrete", 0x987d1ade87899a29),
    ("defaults/every/auto-MidDiscrete", 0xa9f1172665c1b302),
    ("defaults/every/auto-HighDiscrete", 0xa4fa410c2030fe35),
    ("defaults/every/Low", 0x4e0c958ff6642ea9),
    ("defaults/every/Medium", 0x987d1ade87899a29),
    ("defaults/every/High", 0xa9f1172665c1b302),
    ("defaults/every/Ultra", 0xa4fa410c2030fe35),
    ("defaults/every/Custom", 0xa4fa410c2030fe35),
    ("defaults/off/auto-Unknown", 0xaca97199d1b95f3f),
    ("defaults/off/auto-Integrated", 0xa56330a304be88ac),
    ("defaults/off/auto-EntryDiscrete", 0xaca97199d1b95f3f),
    ("defaults/off/auto-MidDiscrete", 0x818073cda1ee64d3),
    ("defaults/off/auto-HighDiscrete", 0xd6341aaa43bf7130),
    ("defaults/off/Low", 0xa56330a304be88ac),
    ("defaults/off/Medium", 0xaca97199d1b95f3f),
    ("defaults/off/High", 0x818073cda1ee64d3),
    ("defaults/off/Ultra", 0xd6341aaa43bf7130),
    ("defaults/off/Custom", 0xd6341aaa43bf7130),
    ("above/none/auto-Unknown", 0x4f906cfb48363081),
    ("above/none/auto-Integrated", 0xac4494e819ea5910),
    ("above/none/auto-EntryDiscrete", 0x4f906cfb48363081),
    ("above/none/auto-MidDiscrete", 0x32cd1a4dfbe67978),
    ("above/none/auto-HighDiscrete", 0xe0ea0aaf21313d36),
    ("above/none/Low", 0xac4494e819ea5910),
    ("above/none/Medium", 0x4f906cfb48363081),
    ("above/none/High", 0x32cd1a4dfbe67978),
    ("above/none/Ultra", 0xe0ea0aaf21313d36),
    ("above/none/Custom", 0xfab08f9e23709592),
    ("above/every/auto-Unknown", 0x575426b99fc3545a),
    ("above/every/auto-Integrated", 0x4219b80feff8e8b0),
    ("above/every/auto-EntryDiscrete", 0x575426b99fc3545a),
    ("above/every/auto-MidDiscrete", 0x3d444a0f6549dadc),
    ("above/every/auto-HighDiscrete", 0x63100362cad2d9f5),
    ("above/every/Low", 0x4219b80feff8e8b0),
    ("above/every/Medium", 0x575426b99fc3545a),
    ("above/every/High", 0x3d444a0f6549dadc),
    ("above/every/Ultra", 0x63100362cad2d9f5),
    ("above/every/Custom", 0x911a4aa0e46c14c3),
    ("above/off/auto-Unknown", 0x35027f5ab8d948c1),
    ("above/off/auto-Integrated", 0x89d1d80960fe6308),
    ("above/off/auto-EntryDiscrete", 0x35027f5ab8d948c1),
    ("above/off/auto-MidDiscrete", 0x6ce423607933e095),
    ("above/off/auto-HighDiscrete", 0x49e7f79f1d870df9),
    ("above/off/Low", 0x89d1d80960fe6308),
    ("above/off/Medium", 0x35027f5ab8d948c1),
    ("above/off/High", 0x6ce423607933e095),
    ("above/off/Ultra", 0x49e7f79f1d870df9),
    ("above/off/Custom", 0xb26adbf9434643dd),
    ("below/none/auto-Unknown", 0xc28da7c65173b787),
    ("below/none/auto-Integrated", 0xc28da7c65173b787),
    ("below/none/auto-EntryDiscrete", 0xc28da7c65173b787),
    ("below/none/auto-MidDiscrete", 0xc28da7c65173b787),
    ("below/none/auto-HighDiscrete", 0xc28da7c65173b787),
    ("below/none/Low", 0xc28da7c65173b787),
    ("below/none/Medium", 0xc28da7c65173b787),
    ("below/none/High", 0xc28da7c65173b787),
    ("below/none/Ultra", 0xc28da7c65173b787),
    ("below/none/Custom", 0xc28da7c65173b787),
    ("below/every/auto-Unknown", 0x034f1b9e8edab563),
    ("below/every/auto-Integrated", 0x034f1b9e8edab563),
    ("below/every/auto-EntryDiscrete", 0x034f1b9e8edab563),
    ("below/every/auto-MidDiscrete", 0x034f1b9e8edab563),
    ("below/every/auto-HighDiscrete", 0x034f1b9e8edab563),
    ("below/every/Low", 0x034f1b9e8edab563),
    ("below/every/Medium", 0x034f1b9e8edab563),
    ("below/every/High", 0x034f1b9e8edab563),
    ("below/every/Ultra", 0x034f1b9e8edab563),
    ("below/every/Custom", 0x034f1b9e8edab563),
    ("below/off/auto-Unknown", 0xd76fe1f6b89709c4),
    ("below/off/auto-Integrated", 0xd76fe1f6b89709c4),
    ("below/off/auto-EntryDiscrete", 0xd76fe1f6b89709c4),
    ("below/off/auto-MidDiscrete", 0xd76fe1f6b89709c4),
    ("below/off/auto-HighDiscrete", 0xd76fe1f6b89709c4),
    ("below/off/Low", 0xd76fe1f6b89709c4),
    ("below/off/Medium", 0xd76fe1f6b89709c4),
    ("below/off/High", 0xd76fe1f6b89709c4),
    ("below/off/Ultra", 0xd76fe1f6b89709c4),
    ("below/off/Custom", 0xd76fe1f6b89709c4),
    ("odd/none/auto-Unknown", 0x07f944a7f36a5bfe),
    ("odd/none/auto-Integrated", 0xc9f2174afa8be2f2),
    ("odd/none/auto-EntryDiscrete", 0x07f944a7f36a5bfe),
    ("odd/none/auto-MidDiscrete", 0xb0eecd8578c20093),
    ("odd/none/auto-HighDiscrete", 0x70029145c8ace4e7),
    ("odd/none/Low", 0xc9f2174afa8be2f2),
    ("odd/none/Medium", 0x07f944a7f36a5bfe),
    ("odd/none/High", 0xb0eecd8578c20093),
    ("odd/none/Ultra", 0x70029145c8ace4e7),
    ("odd/none/Custom", 0x6567916232beda83),
    ("odd/every/auto-Unknown", 0xffa3224c77372dc8),
    ("odd/every/auto-Integrated", 0x25bf1c847aafcd9c),
    ("odd/every/auto-EntryDiscrete", 0xffa3224c77372dc8),
    ("odd/every/auto-MidDiscrete", 0x19a127f1ab793ec5),
    ("odd/every/auto-HighDiscrete", 0xf154161fb64cb6d1),
    ("odd/every/Low", 0x25bf1c847aafcd9c),
    ("odd/every/Medium", 0xffa3224c77372dc8),
    ("odd/every/High", 0x19a127f1ab793ec5),
    ("odd/every/Ultra", 0xf154161fb64cb6d1),
    ("odd/every/Custom", 0xdb773d61190bdff7),
    ("odd/off/auto-Unknown", 0xbaa10962819db2bb),
    ("odd/off/auto-Integrated", 0x8614cde3bcfa776c),
    ("odd/off/auto-EntryDiscrete", 0xbaa10962819db2bb),
    ("odd/off/auto-MidDiscrete", 0xe0c95574a0046581),
    ("odd/off/auto-HighDiscrete", 0x6364302f996c170e),
    ("odd/off/Low", 0x8614cde3bcfa776c),
    ("odd/off/Medium", 0xbaa10962819db2bb),
    ("odd/off/High", 0xe0c95574a0046581),
    ("odd/off/Ultra", 0x6364302f996c170e),
    ("odd/off/Custom", 0x523824b2528c259e),
];

#[test]
fn the_settings_chain_resolves_every_case_as_pinned() {
    let actual = digests();
    let listing: String = actual
        .iter()
        .map(|(name, digest)| format!("    (\"{name}\", 0x{digest:016x}),\n"))
        .collect();
    let expected: Vec<(String, u64)> = EXPECTED
        .iter()
        .map(|(name, digest)| (name.to_string(), *digest))
        .collect();
    assert_eq!(actual.len(), 150);
    assert!(
        actual == expected,
        "the resolved settings changed:\n{listing}"
    );
}
