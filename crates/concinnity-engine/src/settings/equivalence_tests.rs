// Pins what the settings chain hands the backend for a matrix of inputs: every
// preset (Auto over each GPU tier), worlds authored above and below every
// ceiling, and the user's overrides on and off. Each case folds the resolved
// settings, what the backend reads from them, and a preset change's
// re-resolved quality group into one digest, so a change to any value the
// backend receives fails here by name. The text keeps the shape of the
// settings structs the digests were pinned over.
//
// The exposure slider once moved only the composite's exposure, and
// auto-exposure metered for the world's authored display. The slider now sets
// the resolved EV, which also biases auto-exposure, and metering follows the
// resolved display; the text prints both as the pinned digests saw them, and
// `user_overrides_reach_the_exposure_bias_and_metering` checks the new values.
//
// The `bare` world (no `GraphicsConfig`) once resolved two values apart from
// the section defaults: two-pass occlusion off and an upscaler at native scale.
// Absence now means the defaults, so a `bare` case checks those two values
// against the resolver's rules and prints them as the pinned digests saw them.

use crate::components::BackgroundOverrides;
use crate::components::PlanetDetailOverrides;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write as _;

use crate::components::{
    AaMode, AmbientOcclusion, AntiAliasing, Bloom, Culling, Display, Exposure, FramePacing,
    GraphicsConfig, IndirectLighting, IndirectTechnique, PassResolution, Reflections, ShadowUpdate,
    Shadows, StreamingConfig, TextureFiltering, UpscaleQuality, UpscalerBackend, Vignette,
    WindowMode,
};
use crate::launch::LaunchRequest;
use crate::render::backend::{GpuProfile, GpuTier};
use crate::render::output::dlss::DlssPreset;
use crate::settings::quality::{more_aggressive_upscale, resolve_ceiling};
use crate::settings::{
    QualityPreset, ResolvedGraphics, ResolvedSettings, SettingsLayers, UserGraphicsSettings,
    WorldAssets, WorldSettings,
};
use concinnity_core::fnv::Fnv1a;

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
    // The world setting every field of every section, and `streaming` where
    // given.
    fn world(&self, streaming: Option<&StreamingConfig>) -> WorldSettings {
        let graphics = GraphicsConfig {
            frame_pacing: self.frame_pacing.clone().into(),
            shadows: self.shadows.clone().into(),
            texture_filtering: self.texture_filtering.clone().into(),
            background: BackgroundOverrides::default(),
            ambient_occlusion: self.ambient_occlusion.clone().into(),
            reflections: self.reflections.clone().into(),
            indirect_lighting: self.indirect_lighting.clone().into(),
            anti_aliasing: self.anti_aliasing.clone().into(),
            exposure: self.exposure.clone().into(),
            bloom: self.bloom.clone().into(),
            vignette: self.vignette.clone().into(),
            display: self.display.clone().into(),
            culling: self.culling.clone().into(),
            planet_detail: PlanetDetailOverrides::default(),
        };
        WorldSettings::new(WorldAssets {
            graphics: Some(&graphics),
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
        graphics_api: Some(crate::platform::Platform::Vulkan),
        graphics_api_trial: Some(crate::settings::GraphicsApiTrial::Untried),
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
// for the fields no layer above it moved then.
fn resolved_text(r: &ResolvedSettings, world: &WorldSettings, two_pass: bool) -> String {
    let g = &r.graphics;
    format!(
        "{:?}",
        (
            (
                &PostProcessConfig::of(g, world),
                g.anti_aliasing.upscale_quality,
                g.shadows.map_size,
                g.shadow_cadence(),
                g.texture_filtering.anisotropy,
            ),
            g.post_process(),
            g.ambient_intensity(),
            (
                g.frame_pacing.vsync,
                g.frame_pacing.fps_cap,
                r.perf_stats,
                r.show_fps,
                r.show_vram
            ),
            g.frame_pacing.frames_in_flight as usize,
            (g.display.hdr, g.display.pq, g.anti_aliasing.upscaling),
            (g.anti_aliasing.upscaler, g.dlss_preset, two_pass),
            (r.texture_cap, r.texture_budget),
        )
    )
}

// The quality group in the field order the digests were pinned in.
#[derive(Debug)]
#[expect(dead_code, reason = "the fields are read only through `Debug`")]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors independent post-process settings"
)]
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
    fn of(g: &ResolvedGraphics, world: &WorldSettings) -> Self {
        let w = &world.graphics;
        let ao = &g.ambient_occlusion;
        let r = &g.reflections;
        let gi = &g.indirect_lighting;
        let ex = &g.exposure;
        Self {
            bloom_intensity: w.bloom.intensity,
            bloom_threshold: w.bloom.threshold,
            bloom_knee: w.bloom.knee,
            exposure_ev: w.exposure.ev,
            vignette_strength: w.vignette.strength,
            lut_strength: w.lut_strength,
            aa_mode: g.anti_aliasing.mode,
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
            ambient_intensity: w.indirect_lighting.ambient_intensity,
            ssgi_intensity: gi.ssgi_intensity,
            ssgi_max_distance: gi.ssgi_max_distance,
            ssgi_resolution: gi.ssgi_resolution,
            ssgi_rays: gi.ssgi_rays,
            auto_exposure: ex.auto,
            auto_exposure_min_ev: ex.auto_min_ev,
            auto_exposure_max_ev: ex.auto_max_ev,
            auto_exposure_speed: ex.auto_speed,
            hdr_display: w.display.hdr,
            hdr_pq: w.display.pq,
            temporal_upscaling: w.anti_aliasing.upscaling,
            upscale_quality: w.anti_aliasing.upscale_quality,
            upscale_backend: w.anti_aliasing.upscaler,
            occlusion_two_pass: w.culling.two_pass_occlusion,
        }
    }
}

// What the backend reads at init, in the field order of the pinned digests.
fn backend_text(
    g: &ResolvedGraphics,
    world: &WorldSettings,
    launch: &LaunchRequest,
    bare: Option<(f32, bool)>,
) -> String {
    let (upscale_scale, two_pass) =
        bare.unwrap_or_else(|| (g.upscale_scale(), g.culling.two_pass_occlusion));
    format!(
        "{:?}",
        (
            (g.post_process(), g.taa(), g.hdr_samples()),
            (g.ssao(), g.ssr(), g.ssgi(), g.rt_reflections()),
            (
                launch.resolve_rt_dynamic(),
                launch.resolve_rt_skinned_geometry(),
                g.reflection_blur_scale()
            ),
            legacy_exposure(g, world),
            (
                g.display.hdr,
                g.display.pq,
                g.anti_aliasing.upscaling,
                upscale_scale
            ),
            (g.anti_aliasing.upscaler, g.dlss_preset, two_pass),
        )
    )
}

// What a live quality rebuild reads, in the field order of the pinned digests.
fn quality_text(g: &ResolvedGraphics, world: &WorldSettings) -> String {
    let (auto_exposure, bias) = legacy_exposure(g, world);
    format!(
        "{:?}",
        (
            g.taa(),
            g.ssao(),
            g.ssr(),
            g.rt_reflections(),
            g.ssgi(),
            g.reflection_blur_scale(),
            auto_exposure,
            bias,
        )
    )
}

// Auto-exposure as the pinned digests saw it: metered for the world's display,
// biased by the world's EV.
fn legacy_exposure(
    g: &ResolvedGraphics,
    world: &WorldSettings,
) -> (Option<AutoExposureSettings>, f32) {
    let w = &world.graphics;
    let settings = g.exposure.auto_settings(w.display.hdr).map(|s| {
        // No case authors a night bias or night vision; the digests predate
        // them.
        assert_eq!(s.night_bias_ev, 0.0);
        assert_eq!(s.night_vision, 0.0);
        AutoExposureSettings {
            min_ev: s.min_ev,
            max_ev: s.max_ev,
            speed: s.speed,
            target_log_lum: s.target_log_lum,
        }
    });
    (settings, w.exposure.ev)
}

// The auto-exposure settings in the shape the digests were pinned over.
#[derive(Debug)]
#[expect(dead_code, reason = "the fields are read only through `Debug`")]
struct AutoExposureSettings {
    min_ev: f32,
    max_ev: f32,
    speed: f32,
    target_log_lum: f32,
}

// The user's choices after a preset change: every preset-governed override
// cleared, the way the settings menu clears them.
fn preset_cleared(user: &UserGraphicsSettings) -> UserGraphicsSettings {
    use crate::settings::rows::{QUALITY_CYCLES, QUALITY_TOGGLES};
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
    world_name: &str,
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
    let session = launch.session_overrides();
    let layers = |user| SettingsLayers {
        session: Some(&session),
        ..SettingsLayers::new(world, &ceiling, user)
    };
    let resolved = layers(user).resolve();

    let cleared = preset_cleared(user);
    let mut after_preset = resolved.clone();
    after_preset.set_quality(&layers(&cleared).resolve_quality());
    let (bare, two_pass, after_two_pass) = if world_name == "bare" {
        let render_scale = user.render_scale.unwrap_or_else(|| {
            more_aggressive_upscale(UpscaleQuality::default(), ceiling.min_upscale)
        });
        let g = &resolved.graphics;
        assert_eq!(g.upscale_scale(), render_scale.scale());
        let two_pass = user.occlusion_two_pass.unwrap_or(true);
        assert_eq!(g.culling.two_pass_occlusion, two_pass);
        assert_eq!(after_preset.graphics.culling.two_pass_occlusion, two_pass);
        let legacy_two_pass = user.occlusion_two_pass.unwrap_or(false);
        (
            Some((1.0, legacy_two_pass)),
            legacy_two_pass,
            legacy_two_pass,
        )
    } else {
        (
            None,
            resolved.graphics.culling.two_pass_occlusion,
            after_preset.graphics.culling.two_pass_occlusion,
        )
    };
    format!(
        "{}\n{}\n{}\n{}",
        resolved_text(&resolved, world, two_pass),
        backend_text(&resolved.graphics, world, &launch, bare),
        quality_text(&resolved.graphics, world),
        resolved_text(&after_preset, world, after_two_pass),
    )
}

fn digests() -> Vec<(String, u64)> {
    let mut out = Vec::new();
    for (world_name, world) in worlds() {
        for (user_name, user) in users() {
            for (ceiling_name, preset, tier) in ceilings() {
                let text = case_text(world_name, &world, &user, preset, tier);
                out.push((
                    format!("{world_name}/{user_name}/{ceiling_name}"),
                    Fnv1a::of(text.as_bytes()),
                ));
            }
        }
    }
    out
}

// Every case's digest as the settings chain resolves it.
const EXPECTED: &[(&str, u64)] = &[
    ("bare/none/auto-Unknown", 0x4780_53a6_9c4e_9f40),
    ("bare/none/auto-Integrated", 0x934a_487a_1a6b_7c8b),
    ("bare/none/auto-EntryDiscrete", 0x4780_53a6_9c4e_9f40),
    ("bare/none/auto-MidDiscrete", 0x855f_e24c_4f67_b88e),
    ("bare/none/auto-HighDiscrete", 0x3d0a_8536_c6f3_f7ba),
    ("bare/none/Low", 0x934a_487a_1a6b_7c8b),
    ("bare/none/Medium", 0x4780_53a6_9c4e_9f40),
    ("bare/none/High", 0x855f_e24c_4f67_b88e),
    ("bare/none/Ultra", 0x3d0a_8536_c6f3_f7ba),
    ("bare/none/Custom", 0x3d0a_8536_c6f3_f7ba),
    ("bare/every/auto-Unknown", 0xa2da_51c8_7efc_9772),
    ("bare/every/auto-Integrated", 0xaf21_63bc_c5c4_4cfa),
    ("bare/every/auto-EntryDiscrete", 0xa2da_51c8_7efc_9772),
    ("bare/every/auto-MidDiscrete", 0x6b57_5323_becc_6e13),
    ("bare/every/auto-HighDiscrete", 0x6361_b104_1593_e3b8),
    ("bare/every/Low", 0xaf21_63bc_c5c4_4cfa),
    ("bare/every/Medium", 0xa2da_51c8_7efc_9772),
    ("bare/every/High", 0x6b57_5323_becc_6e13),
    ("bare/every/Ultra", 0x6361_b104_1593_e3b8),
    ("bare/every/Custom", 0x6361_b104_1593_e3b8),
    ("bare/off/auto-Unknown", 0x93ab_bbdf_fe96_aba0),
    ("bare/off/auto-Integrated", 0xb7cc_e711_e3a9_4a8e),
    ("bare/off/auto-EntryDiscrete", 0x93ab_bbdf_fe96_aba0),
    ("bare/off/auto-MidDiscrete", 0x104d_f9a1_5cf3_4227),
    ("bare/off/auto-HighDiscrete", 0x734a_0e53_6bf0_07c2),
    ("bare/off/Low", 0xb7cc_e711_e3a9_4a8e),
    ("bare/off/Medium", 0x93ab_bbdf_fe96_aba0),
    ("bare/off/High", 0x104d_f9a1_5cf3_4227),
    ("bare/off/Ultra", 0x734a_0e53_6bf0_07c2),
    ("bare/off/Custom", 0x734a_0e53_6bf0_07c2),
    ("defaults/none/auto-Unknown", 0x73b0_0efc_66ca_adf4),
    ("defaults/none/auto-Integrated", 0x8fa8_1548_48f5_8876),
    ("defaults/none/auto-EntryDiscrete", 0x73b0_0efc_66ca_adf4),
    ("defaults/none/auto-MidDiscrete", 0x4888_5822_f403_b029),
    ("defaults/none/auto-HighDiscrete", 0x5633_3ce9_2a7d_0353),
    ("defaults/none/Low", 0x8fa8_1548_48f5_8876),
    ("defaults/none/Medium", 0x73b0_0efc_66ca_adf4),
    ("defaults/none/High", 0x4888_5822_f403_b029),
    ("defaults/none/Ultra", 0x5633_3ce9_2a7d_0353),
    ("defaults/none/Custom", 0x5633_3ce9_2a7d_0353),
    ("defaults/every/auto-Unknown", 0x1d2b_20c1_bbb6_a04f),
    ("defaults/every/auto-Integrated", 0xe7e4_baa6_3b72_91f7),
    ("defaults/every/auto-EntryDiscrete", 0x1d2b_20c1_bbb6_a04f),
    ("defaults/every/auto-MidDiscrete", 0xf907_1f57_1b69_2ee0),
    ("defaults/every/auto-HighDiscrete", 0xdbb0_c012_a8fb_844b),
    ("defaults/every/Low", 0xe7e4_baa6_3b72_91f7),
    ("defaults/every/Medium", 0x1d2b_20c1_bbb6_a04f),
    ("defaults/every/High", 0xf907_1f57_1b69_2ee0),
    ("defaults/every/Ultra", 0xdbb0_c012_a8fb_844b),
    ("defaults/every/Custom", 0xdbb0_c012_a8fb_844b),
    ("defaults/off/auto-Unknown", 0x95e0_92eb_59b8_c3aa),
    ("defaults/off/auto-Integrated", 0x8359_d1bc_b813_6f8d),
    ("defaults/off/auto-EntryDiscrete", 0x95e0_92eb_59b8_c3aa),
    ("defaults/off/auto-MidDiscrete", 0x2c27_af3b_709b_82f4),
    ("defaults/off/auto-HighDiscrete", 0xda92_89d1_d6ae_3e0f),
    ("defaults/off/Low", 0x8359_d1bc_b813_6f8d),
    ("defaults/off/Medium", 0x95e0_92eb_59b8_c3aa),
    ("defaults/off/High", 0x2c27_af3b_709b_82f4),
    ("defaults/off/Ultra", 0xda92_89d1_d6ae_3e0f),
    ("defaults/off/Custom", 0xda92_89d1_d6ae_3e0f),
    ("above/none/auto-Unknown", 0xb1d3_9453_16d5_2d4e),
    ("above/none/auto-Integrated", 0xd44c_c1b1_1ab1_e097),
    ("above/none/auto-EntryDiscrete", 0xb1d3_9453_16d5_2d4e),
    ("above/none/auto-MidDiscrete", 0xc9fe_d8f0_d935_770f),
    ("above/none/auto-HighDiscrete", 0x1239_fe32_a903_dfb3),
    ("above/none/Low", 0xd44c_c1b1_1ab1_e097),
    ("above/none/Medium", 0xb1d3_9453_16d5_2d4e),
    ("above/none/High", 0xc9fe_d8f0_d935_770f),
    ("above/none/Ultra", 0x1239_fe32_a903_dfb3),
    ("above/none/Custom", 0x7469_30da_3bf7_01df),
    ("above/every/auto-Unknown", 0x6552_9f9c_af7c_894a),
    ("above/every/auto-Integrated", 0xd641_987b_c823_fa20),
    ("above/every/auto-EntryDiscrete", 0x6552_9f9c_af7c_894a),
    ("above/every/auto-MidDiscrete", 0xc2ff_dd0d_98fe_a18c),
    ("above/every/auto-HighDiscrete", 0x9b11_3d73_3830_6ff5),
    ("above/every/Low", 0xd641_987b_c823_fa20),
    ("above/every/Medium", 0x6552_9f9c_af7c_894a),
    ("above/every/High", 0xc2ff_dd0d_98fe_a18c),
    ("above/every/Ultra", 0x9b11_3d73_3830_6ff5),
    ("above/every/Custom", 0xc91b_84b1_51c9_aac3),
    ("above/off/auto-Unknown", 0xc444_e46d_937b_90f0),
    ("above/off/auto-Integrated", 0xf725_0103_e47b_39a5),
    ("above/off/auto-EntryDiscrete", 0xc444_e46d_937b_90f0),
    ("above/off/auto-MidDiscrete", 0xffdc_a59c_8243_de20),
    ("above/off/auto-HighDiscrete", 0x2e90_ae7a_30f5_c4ec),
    ("above/off/Low", 0xf725_0103_e47b_39a5),
    ("above/off/Medium", 0xc444_e46d_937b_90f0),
    ("above/off/High", 0xffdc_a59c_8243_de20),
    ("above/off/Ultra", 0x2e90_ae7a_30f5_c4ec),
    ("above/off/Custom", 0xf30f_8cb4_8879_26d0),
    ("below/none/auto-Unknown", 0x1b94_aaa3_fcdc_9b12),
    ("below/none/auto-Integrated", 0x1b94_aaa3_fcdc_9b12),
    ("below/none/auto-EntryDiscrete", 0x1b94_aaa3_fcdc_9b12),
    ("below/none/auto-MidDiscrete", 0x1b94_aaa3_fcdc_9b12),
    ("below/none/auto-HighDiscrete", 0x1b94_aaa3_fcdc_9b12),
    ("below/none/Low", 0x1b94_aaa3_fcdc_9b12),
    ("below/none/Medium", 0x1b94_aaa3_fcdc_9b12),
    ("below/none/High", 0x1b94_aaa3_fcdc_9b12),
    ("below/none/Ultra", 0x1b94_aaa3_fcdc_9b12),
    ("below/none/Custom", 0x1b94_aaa3_fcdc_9b12),
    ("below/every/auto-Unknown", 0xb58d_ac42_3371_d457),
    ("below/every/auto-Integrated", 0xb58d_ac42_3371_d457),
    ("below/every/auto-EntryDiscrete", 0xb58d_ac42_3371_d457),
    ("below/every/auto-MidDiscrete", 0xb58d_ac42_3371_d457),
    ("below/every/auto-HighDiscrete", 0x7c83_86ec_5155_ce4b),
    ("below/every/Low", 0xb58d_ac42_3371_d457),
    ("below/every/Medium", 0xb58d_ac42_3371_d457),
    ("below/every/High", 0xb58d_ac42_3371_d457),
    ("below/every/Ultra", 0x7c83_86ec_5155_ce4b),
    ("below/every/Custom", 0x7c83_86ec_5155_ce4b),
    ("below/off/auto-Unknown", 0x17c6_ea53_3554_aac1),
    ("below/off/auto-Integrated", 0x17c6_ea53_3554_aac1),
    ("below/off/auto-EntryDiscrete", 0x17c6_ea53_3554_aac1),
    ("below/off/auto-MidDiscrete", 0x17c6_ea53_3554_aac1),
    ("below/off/auto-HighDiscrete", 0x17c6_ea53_3554_aac1),
    ("below/off/Low", 0x17c6_ea53_3554_aac1),
    ("below/off/Medium", 0x17c6_ea53_3554_aac1),
    ("below/off/High", 0x17c6_ea53_3554_aac1),
    ("below/off/Ultra", 0x17c6_ea53_3554_aac1),
    ("below/off/Custom", 0x17c6_ea53_3554_aac1),
    ("odd/none/auto-Unknown", 0x01c8_46bb_5ebf_5823),
    ("odd/none/auto-Integrated", 0x83ed_e00e_4ead_a7f3),
    ("odd/none/auto-EntryDiscrete", 0x01c8_46bb_5ebf_5823),
    ("odd/none/auto-MidDiscrete", 0x54d1_d2d7_2c34_287c),
    ("odd/none/auto-HighDiscrete", 0xe0b9_4b25_bbe6_bc0e),
    ("odd/none/Low", 0x83ed_e00e_4ead_a7f3),
    ("odd/none/Medium", 0x01c8_46bb_5ebf_5823),
    ("odd/none/High", 0x54d1_d2d7_2c34_287c),
    ("odd/none/Ultra", 0xe0b9_4b25_bbe6_bc0e),
    ("odd/none/Custom", 0x553b_e17a_51ab_64e2),
    ("odd/every/auto-Unknown", 0xb211_c9f0_c4c8_7854),
    ("odd/every/auto-Integrated", 0x6d74_85c7_14a5_0f90),
    ("odd/every/auto-EntryDiscrete", 0xb211_c9f0_c4c8_7854),
    ("odd/every/auto-MidDiscrete", 0xcf1e_e560_7ba5_5249),
    ("odd/every/auto-HighDiscrete", 0x39b1_6d46_c0ff_2ca9),
    ("odd/every/Low", 0x6d74_85c7_14a5_0f90),
    ("odd/every/Medium", 0xb211_c9f0_c4c8_7854),
    ("odd/every/High", 0xcf1e_e560_7ba5_5249),
    ("odd/every/Ultra", 0x39b1_6d46_c0ff_2ca9),
    ("odd/every/Custom", 0x8727_1486_1600_5e0f),
    ("odd/off/auto-Unknown", 0x69e5_c123_9ff8_6736),
    ("odd/off/auto-Integrated", 0x0ead_71ab_7061_a531),
    ("odd/off/auto-EntryDiscrete", 0x69e5_c123_9ff8_6736),
    ("odd/off/auto-MidDiscrete", 0x9261_e802_8388_deba),
    ("odd/off/auto-HighDiscrete", 0x4dc1_6eeb_65d3_24b9),
    ("odd/off/Low", 0x0ead_71ab_7061_a531),
    ("odd/off/Medium", 0x69e5_c123_9ff8_6736),
    ("odd/off/High", 0x9261_e802_8388_deba),
    ("odd/off/Ultra", 0x4dc1_6eeb_65d3_24b9),
    ("odd/off/Custom", 0x97c8_04e4_20d5_e8b5),
];

#[test]
fn the_settings_chain_resolves_every_case_as_pinned() {
    let actual = digests();
    let listing = actual
        .iter()
        .fold(String::new(), |mut listing, (name, digest)| {
            writeln!(listing, "    (\"{name}\", 0x{digest:016x}),").unwrap();
            listing
        });
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

// The values a `bare` case now resolves for the two fields absence used to
// move, written out from the rules: the render scale is the more aggressive of
// the `quality` default and the tier's minimum unless the user picked one, and
// two-pass occlusion takes its `true` default unless the user set it.
#[test]
fn a_bare_world_resolves_to_the_section_defaults() {
    let third = 2.0 / 3.0;
    let expected = |user: &str, ceiling: &str| -> (f32, bool) {
        if user == "every" {
            return (0.587, true);
        }
        let scale = match ceiling {
            "Low" | "auto-Integrated" => 0.5,
            "Medium" | "auto-Unknown" | "auto-EntryDiscrete" => 0.587,
            _ => third,
        };
        (scale, true)
    };
    let bare = WorldSettings::default();
    for (user_name, user) in users() {
        for (ceiling_name, preset, tier) in ceilings() {
            let profile = GpuProfile {
                tier,
                ..GpuProfile::UNKNOWN
            };
            let ceiling = resolve_ceiling(preset, &profile);
            let graphics = SettingsLayers::new(&bare, &ceiling, &user)
                .resolve()
                .graphics;
            let (scale, two_pass) = expected(user_name, &ceiling_name);
            let case = format!("{user_name}/{ceiling_name}");
            assert_eq!(graphics.upscale_scale(), scale, "{case}");
            assert_eq!(graphics.culling.two_pass_occlusion, two_pass, "{case}");
        }
    }
}

// Absent sections and sections setting every field to its default resolve to
// the same text; `every` differs only through the `StreamingConfig` the
// `defaults` world declares.
#[test]
fn a_bare_world_resolves_like_every_field_set_to_its_default() {
    let bare = WorldSettings::default();
    let defaults = Declared::default().world(Some(&StreamingConfig::default()));
    for (user_name, user) in users() {
        if user_name == "every" {
            continue;
        }
        for (ceiling_name, preset, tier) in ceilings() {
            assert_eq!(
                case_text("", &bare, &user, preset, tier),
                case_text("", &defaults, &user, preset, tier),
                "{user_name}/{ceiling_name}"
            );
        }
    }
}

// The exposure slider sets the resolved EV, which auto-exposure takes as its
// bias, and auto-exposure meters for the display the user resolved.
#[test]
fn user_overrides_reach_the_exposure_bias_and_metering() {
    use crate::gfx::auto_exposure::EV_LIMIT;
    for (world_name, world) in worlds() {
        for (user_name, user) in users() {
            let ceiling = crate::settings::QualityCeiling::NONE;
            let g = SettingsLayers::new(&world, &ceiling, &user)
                .resolve()
                .graphics;
            let w = &world.graphics;
            let case = format!("{world_name}/{user_name}");
            let ev = user
                .exposure_ev
                .map_or(w.exposure.ev, |v| v.clamp(-EV_LIMIT, EV_LIMIT));
            assert_eq!(g.auto_exposure_bias_ev(), ev, "{case}");
            let hdr = user.hdr_display.unwrap_or(w.display.hdr);
            assert_eq!(g.auto_exposure(), g.exposure.auto_settings(hdr), "{case}");
        }
    }
}
