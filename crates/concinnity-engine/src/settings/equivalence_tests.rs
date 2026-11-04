// Pins what the settings chain hands the backend for a matrix of inputs: every
// preset (Auto over each GPU tier), worlds authored above and below every
// ceiling, and the user's overrides on and off. Each case folds the resolved
// settings, what the backend derives from them, and a preset change's
// re-resolved quality group into one digest, so a change to any value the
// backend receives fails here by name, with the case's text to diff.

use crate::components::BackgroundOverrides;
use crate::components::PlanetDetailOverrides;
use alloc::format;
use alloc::string::String;
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
        customized_from: None,
        vsync: Some(true),
        fps_cap: Some(60),
        perf_stats: Some(false),
        show_fps: Some(false),
        show_vram: Some(true),
        window_mode: Some(WindowMode::Borderless),
        resolution: Some([1920, 1080, 60]),
        graphics_api: Some(concinnity_core::platform::Platform::Vulkan),
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

// Everything the settings chain resolved, field by field.
fn resolved_text(r: &ResolvedSettings) -> String {
    format!("{r:#?}")
}

// What the backend derives from the resolved settings at init.
fn backend_text(g: &ResolvedGraphics, launch: &LaunchRequest) -> String {
    format!(
        "{:#?}",
        (
            (g.post_process(), g.taa(), g.hdr_samples()),
            (g.ssao(), g.ssr(), g.ssgi(), g.rt_reflections()),
            (
                launch.resolve_rt_dynamic(),
                launch.resolve_rt_skinned_geometry(),
                g.reflection_blur_scale()
            ),
            (g.auto_exposure(), g.auto_exposure_bias_ev()),
            (g.upscale_scale(), g.shadow_cadence(), g.ambient_intensity()),
        )
    )
}

// What a live quality rebuild reads.
fn quality_text(g: &ResolvedGraphics) -> String {
    format!(
        "{:#?}",
        (
            g.taa(),
            g.ssao(),
            g.ssr(),
            g.rt_reflections(),
            g.ssgi(),
            g.reflection_blur_scale(),
            g.auto_exposure(),
            g.auto_exposure_bias_ev(),
        )
    )
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
        args: crate::launch::LaunchArgs {
            dlss_preset: Some(DlssPreset::M),
            ..crate::launch::LaunchArgs::NONE
        },
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
    if world_name == "bare" {
        let render_scale = user.render_scale.unwrap_or_else(|| {
            more_aggressive_upscale(UpscaleQuality::default(), ceiling.min_upscale)
        });
        let g = &resolved.graphics;
        assert_eq!(g.upscale_scale(), render_scale.scale());
        let two_pass = user.occlusion_two_pass.unwrap_or(true);
        assert_eq!(g.culling.two_pass_occlusion, two_pass);
        assert_eq!(after_preset.graphics.culling.two_pass_occlusion, two_pass);
    }
    format!(
        "{}\n{}\n{}\n{}",
        resolved_text(&resolved),
        backend_text(&resolved.graphics, &launch),
        quality_text(&resolved.graphics),
        resolved_text(&after_preset),
    )
}

// Every case's name, text and digest.
fn cases() -> Vec<(String, String, u64)> {
    let mut out = Vec::new();
    for (world_name, world) in worlds() {
        for (user_name, user) in users() {
            for (ceiling_name, preset, tier) in ceilings() {
                let text = case_text(world_name, &world, &user, preset, tier);
                let digest = Fnv1a::of(text.as_bytes());
                out.push((
                    format!("{world_name}/{user_name}/{ceiling_name}"),
                    text,
                    digest,
                ));
            }
        }
    }
    out
}

// Every case's digest as the settings chain resolves it.
const EXPECTED: &[(&str, u64)] = &[
    ("bare/none/auto-Unknown", 0xa267_9cdf_315e_ab16),
    ("bare/none/auto-Integrated", 0x0b32_dbb2_2e58_cba7),
    ("bare/none/auto-EntryDiscrete", 0xa267_9cdf_315e_ab16),
    ("bare/none/auto-MidDiscrete", 0x3ca8_f71b_0a1c_7ef4),
    ("bare/none/auto-HighDiscrete", 0x9808_3e64_1a34_de64),
    ("bare/none/Low", 0x0b32_dbb2_2e58_cba7),
    ("bare/none/Medium", 0xa267_9cdf_315e_ab16),
    ("bare/none/High", 0x3ca8_f71b_0a1c_7ef4),
    ("bare/none/Ultra", 0x9808_3e64_1a34_de64),
    ("bare/none/Custom", 0x9808_3e64_1a34_de64),
    ("bare/every/auto-Unknown", 0x6914_c4e7_a6a7_a3bf),
    ("bare/every/auto-Integrated", 0x547e_abcf_2212_fbca),
    ("bare/every/auto-EntryDiscrete", 0x6914_c4e7_a6a7_a3bf),
    ("bare/every/auto-MidDiscrete", 0x1042_f35b_38a3_0fc6),
    ("bare/every/auto-HighDiscrete", 0xc372_517a_2710_9eef),
    ("bare/every/Low", 0x547e_abcf_2212_fbca),
    ("bare/every/Medium", 0x6914_c4e7_a6a7_a3bf),
    ("bare/every/High", 0x1042_f35b_38a3_0fc6),
    ("bare/every/Ultra", 0xc372_517a_2710_9eef),
    ("bare/every/Custom", 0xc372_517a_2710_9eef),
    ("bare/off/auto-Unknown", 0xe389_46c5_cc7f_6125),
    ("bare/off/auto-Integrated", 0x7cb9_bd5d_5ff4_8625),
    ("bare/off/auto-EntryDiscrete", 0xe389_46c5_cc7f_6125),
    ("bare/off/auto-MidDiscrete", 0x0aee_b680_f635_a52e),
    ("bare/off/auto-HighDiscrete", 0x2f96_fddb_d473_0325),
    ("bare/off/Low", 0x7cb9_bd5d_5ff4_8625),
    ("bare/off/Medium", 0xe389_46c5_cc7f_6125),
    ("bare/off/High", 0x0aee_b680_f635_a52e),
    ("bare/off/Ultra", 0x2f96_fddb_d473_0325),
    ("bare/off/Custom", 0x2f96_fddb_d473_0325),
    ("defaults/none/auto-Unknown", 0xa267_9cdf_315e_ab16),
    ("defaults/none/auto-Integrated", 0x0b32_dbb2_2e58_cba7),
    ("defaults/none/auto-EntryDiscrete", 0xa267_9cdf_315e_ab16),
    ("defaults/none/auto-MidDiscrete", 0x3ca8_f71b_0a1c_7ef4),
    ("defaults/none/auto-HighDiscrete", 0x9808_3e64_1a34_de64),
    ("defaults/none/Low", 0x0b32_dbb2_2e58_cba7),
    ("defaults/none/Medium", 0xa267_9cdf_315e_ab16),
    ("defaults/none/High", 0x3ca8_f71b_0a1c_7ef4),
    ("defaults/none/Ultra", 0x9808_3e64_1a34_de64),
    ("defaults/none/Custom", 0x9808_3e64_1a34_de64),
    ("defaults/every/auto-Unknown", 0x03bb_f435_f9ad_b6f9),
    ("defaults/every/auto-Integrated", 0x9b0b_7c87_ce3c_2e18),
    ("defaults/every/auto-EntryDiscrete", 0x03bb_f435_f9ad_b6f9),
    ("defaults/every/auto-MidDiscrete", 0xfe46_3715_5103_a1ba),
    ("defaults/every/auto-HighDiscrete", 0x7c04_5e2d_7c4a_3bdd),
    ("defaults/every/Low", 0x9b0b_7c87_ce3c_2e18),
    ("defaults/every/Medium", 0x03bb_f435_f9ad_b6f9),
    ("defaults/every/High", 0xfe46_3715_5103_a1ba),
    ("defaults/every/Ultra", 0x7c04_5e2d_7c4a_3bdd),
    ("defaults/every/Custom", 0x7c04_5e2d_7c4a_3bdd),
    ("defaults/off/auto-Unknown", 0xe389_46c5_cc7f_6125),
    ("defaults/off/auto-Integrated", 0x7cb9_bd5d_5ff4_8625),
    ("defaults/off/auto-EntryDiscrete", 0xe389_46c5_cc7f_6125),
    ("defaults/off/auto-MidDiscrete", 0x0aee_b680_f635_a52e),
    ("defaults/off/auto-HighDiscrete", 0x2f96_fddb_d473_0325),
    ("defaults/off/Low", 0x7cb9_bd5d_5ff4_8625),
    ("defaults/off/Medium", 0xe389_46c5_cc7f_6125),
    ("defaults/off/High", 0x0aee_b680_f635_a52e),
    ("defaults/off/Ultra", 0x2f96_fddb_d473_0325),
    ("defaults/off/Custom", 0x2f96_fddb_d473_0325),
    ("above/none/auto-Unknown", 0x0c7b_ce1a_caea_b086),
    ("above/none/auto-Integrated", 0x613d_4768_3721_ad98),
    ("above/none/auto-EntryDiscrete", 0x0c7b_ce1a_caea_b086),
    ("above/none/auto-MidDiscrete", 0x5244_e5ef_f93f_8309),
    ("above/none/auto-HighDiscrete", 0x6b88_624a_3671_7d69),
    ("above/none/Low", 0x613d_4768_3721_ad98),
    ("above/none/Medium", 0x0c7b_ce1a_caea_b086),
    ("above/none/High", 0x5244_e5ef_f93f_8309),
    ("above/none/Ultra", 0x6b88_624a_3671_7d69),
    ("above/none/Custom", 0xebf9_4c70_95e8_e179),
    ("above/every/auto-Unknown", 0x7a36_08c8_7979_2d0e),
    ("above/every/auto-Integrated", 0x167b_42ce_717c_36d1),
    ("above/every/auto-EntryDiscrete", 0x7a36_08c8_7979_2d0e),
    ("above/every/auto-MidDiscrete", 0x7261_03f6_c6d5_63e4),
    ("above/every/auto-HighDiscrete", 0x1872_228f_04d6_a37e),
    ("above/every/Low", 0x167b_42ce_717c_36d1),
    ("above/every/Medium", 0x7a36_08c8_7979_2d0e),
    ("above/every/High", 0x7261_03f6_c6d5_63e4),
    ("above/every/Ultra", 0x1872_228f_04d6_a37e),
    ("above/every/Custom", 0x9f54_fc1d_e9a6_6a54),
    ("above/off/auto-Unknown", 0x57da_0e09_6c57_01d5),
    ("above/off/auto-Integrated", 0x2ceb_a6e8_6f0e_4263),
    ("above/off/auto-EntryDiscrete", 0x57da_0e09_6c57_01d5),
    ("above/off/auto-MidDiscrete", 0x5570_1911_c0a5_895d),
    ("above/off/auto-HighDiscrete", 0x2c75_c446_59d1_5235),
    ("above/off/Low", 0x2ceb_a6e8_6f0e_4263),
    ("above/off/Medium", 0x57da_0e09_6c57_01d5),
    ("above/off/High", 0x5570_1911_c0a5_895d),
    ("above/off/Ultra", 0x2c75_c446_59d1_5235),
    ("above/off/Custom", 0xed6f_e113_0c89_28b9),
    ("below/none/auto-Unknown", 0xc06d_1bc4_9333_95dc),
    ("below/none/auto-Integrated", 0xb784_0850_67a2_0210),
    ("below/none/auto-EntryDiscrete", 0xc06d_1bc4_9333_95dc),
    ("below/none/auto-MidDiscrete", 0x8af9_ab70_7066_f814),
    ("below/none/auto-HighDiscrete", 0x6cce_133b_50d3_3e76),
    ("below/none/Low", 0xb784_0850_67a2_0210),
    ("below/none/Medium", 0xc06d_1bc4_9333_95dc),
    ("below/none/High", 0x8af9_ab70_7066_f814),
    ("below/none/Ultra", 0x6cce_133b_50d3_3e76),
    ("below/none/Custom", 0x6cce_133b_50d3_3e76),
    ("below/every/auto-Unknown", 0x63ab_1c53_d4f8_c7a3),
    ("below/every/auto-Integrated", 0xbdeb_5a9d_9a50_6a93),
    ("below/every/auto-EntryDiscrete", 0x63ab_1c53_d4f8_c7a3),
    ("below/every/auto-MidDiscrete", 0xad7b_21ff_7ada_3a7d),
    ("below/every/auto-HighDiscrete", 0x296c_bb1d_3bc9_95bf),
    ("below/every/Low", 0xbdeb_5a9d_9a50_6a93),
    ("below/every/Medium", 0x63ab_1c53_d4f8_c7a3),
    ("below/every/High", 0xad7b_21ff_7ada_3a7d),
    ("below/every/Ultra", 0x296c_bb1d_3bc9_95bf),
    ("below/every/Custom", 0x296c_bb1d_3bc9_95bf),
    ("below/off/auto-Unknown", 0x61aa_3f40_849f_44c9),
    ("below/off/auto-Integrated", 0x072a_02bf_d853_5835),
    ("below/off/auto-EntryDiscrete", 0x61aa_3f40_849f_44c9),
    ("below/off/auto-MidDiscrete", 0x0392_f58c_0fb0_29ab),
    ("below/off/auto-HighDiscrete", 0x0b8a_19d0_19f2_3b71),
    ("below/off/Low", 0x072a_02bf_d853_5835),
    ("below/off/Medium", 0x61aa_3f40_849f_44c9),
    ("below/off/High", 0x0392_f58c_0fb0_29ab),
    ("below/off/Ultra", 0x0b8a_19d0_19f2_3b71),
    ("below/off/Custom", 0x0b8a_19d0_19f2_3b71),
    ("odd/none/auto-Unknown", 0xa00f_2988_2f25_23cd),
    ("odd/none/auto-Integrated", 0x0ae3_7817_dd53_791e),
    ("odd/none/auto-EntryDiscrete", 0xa00f_2988_2f25_23cd),
    ("odd/none/auto-MidDiscrete", 0x8f5a_8f0a_568e_cb99),
    ("odd/none/auto-HighDiscrete", 0x3e86_97b6_0368_689f),
    ("odd/none/Low", 0x0ae3_7817_dd53_791e),
    ("odd/none/Medium", 0xa00f_2988_2f25_23cd),
    ("odd/none/High", 0x8f5a_8f0a_568e_cb99),
    ("odd/none/Ultra", 0x3e86_97b6_0368_689f),
    ("odd/none/Custom", 0xbcc8_ad36_1849_806b),
    ("odd/every/auto-Unknown", 0xc598_f283_d2e3_6660),
    ("odd/every/auto-Integrated", 0xef45_60c9_b8be_240f),
    ("odd/every/auto-EntryDiscrete", 0xc598_f283_d2e3_6660),
    ("odd/every/auto-MidDiscrete", 0x45c5_4780_b866_dac5),
    ("odd/every/auto-HighDiscrete", 0xafe3_b77f_cecf_e786),
    ("odd/every/Low", 0xef45_60c9_b8be_240f),
    ("odd/every/Medium", 0xc598_f283_d2e3_6660),
    ("odd/every/High", 0x45c5_4780_b866_dac5),
    ("odd/every/Ultra", 0xafe3_b77f_cecf_e786),
    ("odd/every/Custom", 0x2a8e_6403_a471_77ac),
    ("odd/off/auto-Unknown", 0xa7e4_e397_7351_64ab),
    ("odd/off/auto-Integrated", 0xd848_668b_13dd_23f1),
    ("odd/off/auto-EntryDiscrete", 0xa7e4_e397_7351_64ab),
    ("odd/off/auto-MidDiscrete", 0xe029_7697_315d_bdb4),
    ("odd/off/auto-HighDiscrete", 0x0369_2962_5cbc_a659),
    ("odd/off/Low", 0xd848_668b_13dd_23f1),
    ("odd/off/Medium", 0xa7e4_e397_7351_64ab),
    ("odd/off/High", 0xe029_7697_315d_bdb4),
    ("odd/off/Ultra", 0x0369_2962_5cbc_a659),
    ("odd/off/Custom", 0xb8d7_00ed_6940_a8e9),
];

#[test]
fn the_settings_chain_resolves_every_case_as_pinned() {
    let actual = cases();
    assert_eq!(actual.len(), EXPECTED.len());
    let mut changed = String::new();
    let mut listing = String::new();
    for ((name, text, digest), (expected_name, expected)) in actual.iter().zip(EXPECTED) {
        assert_eq!(name, expected_name, "the case matrix changed order");
        writeln!(listing, "    (\"{name}\", 0x{digest:016x}),").unwrap();
        if digest != expected {
            writeln!(changed, "{name} now resolves to:\n{text}\n").unwrap();
        }
    }
    assert!(
        changed.is_empty(),
        "the resolved settings changed:\n{changed}\nre-pinned digests:\n{listing}"
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
