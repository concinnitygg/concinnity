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
use crate::render::config::ResolvedGraphics;
use crate::render::output::dlss::DlssPreset;
use crate::settings::quality::{more_aggressive_upscale, resolve_ceiling};
use crate::settings::{
    QualityPreset, ResolvedSettings, SettingsLayers, UserGraphicsSettings, WorldAssets,
    WorldSettings,
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
            far_distance: 6000,
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
            far_distance: 0,
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
    ("bare/none/auto-Unknown", 0xdf84_c4a5_d16c_7bd0),
    ("bare/none/auto-Integrated", 0xbde2_a076_9f46_da09),
    ("bare/none/auto-EntryDiscrete", 0xdf84_c4a5_d16c_7bd0),
    ("bare/none/auto-MidDiscrete", 0xf6f3_d454_1e9a_ca2e),
    ("bare/none/auto-HighDiscrete", 0xb9c6_300b_f313_a0ac),
    ("bare/none/Low", 0xbde2_a076_9f46_da09),
    ("bare/none/Medium", 0xdf84_c4a5_d16c_7bd0),
    ("bare/none/High", 0xf6f3_d454_1e9a_ca2e),
    ("bare/none/Ultra", 0xb9c6_300b_f313_a0ac),
    ("bare/none/Custom", 0xb9c6_300b_f313_a0ac),
    ("bare/every/auto-Unknown", 0xab31_0e70_4de2_4881),
    ("bare/every/auto-Integrated", 0x7a51_303c_4ff0_7438),
    ("bare/every/auto-EntryDiscrete", 0xab31_0e70_4de2_4881),
    ("bare/every/auto-MidDiscrete", 0x944a_ea94_3604_ab30),
    ("bare/every/auto-HighDiscrete", 0x0469_8175_a6b3_2fc5),
    ("bare/every/Low", 0x7a51_303c_4ff0_7438),
    ("bare/every/Medium", 0xab31_0e70_4de2_4881),
    ("bare/every/High", 0x944a_ea94_3604_ab30),
    ("bare/every/Ultra", 0x0469_8175_a6b3_2fc5),
    ("bare/every/Custom", 0x0469_8175_a6b3_2fc5),
    ("bare/off/auto-Unknown", 0xc843_9c39_f193_a197),
    ("bare/off/auto-Integrated", 0x1785_1cf3_9b37_acc7),
    ("bare/off/auto-EntryDiscrete", 0xc843_9c39_f193_a197),
    ("bare/off/auto-MidDiscrete", 0x5c2a_c1f6_0a7b_6544),
    ("bare/off/auto-HighDiscrete", 0x19d3_656d_b94e_8855),
    ("bare/off/Low", 0x1785_1cf3_9b37_acc7),
    ("bare/off/Medium", 0xc843_9c39_f193_a197),
    ("bare/off/High", 0x5c2a_c1f6_0a7b_6544),
    ("bare/off/Ultra", 0x19d3_656d_b94e_8855),
    ("bare/off/Custom", 0x19d3_656d_b94e_8855),
    ("defaults/none/auto-Unknown", 0xdf84_c4a5_d16c_7bd0),
    ("defaults/none/auto-Integrated", 0xbde2_a076_9f46_da09),
    ("defaults/none/auto-EntryDiscrete", 0xdf84_c4a5_d16c_7bd0),
    ("defaults/none/auto-MidDiscrete", 0xf6f3_d454_1e9a_ca2e),
    ("defaults/none/auto-HighDiscrete", 0xb9c6_300b_f313_a0ac),
    ("defaults/none/Low", 0xbde2_a076_9f46_da09),
    ("defaults/none/Medium", 0xdf84_c4a5_d16c_7bd0),
    ("defaults/none/High", 0xf6f3_d454_1e9a_ca2e),
    ("defaults/none/Ultra", 0xb9c6_300b_f313_a0ac),
    ("defaults/none/Custom", 0xb9c6_300b_f313_a0ac),
    ("defaults/every/auto-Unknown", 0x384a_9579_f842_a90b),
    ("defaults/every/auto-Integrated", 0xe61f_3350_7c49_157a),
    ("defaults/every/auto-EntryDiscrete", 0x384a_9579_f842_a90b),
    ("defaults/every/auto-MidDiscrete", 0x209e_7490_39d8_5868),
    ("defaults/every/auto-HighDiscrete", 0xa1ac_6903_4f5f_e0e3),
    ("defaults/every/Low", 0xe61f_3350_7c49_157a),
    ("defaults/every/Medium", 0x384a_9579_f842_a90b),
    ("defaults/every/High", 0x209e_7490_39d8_5868),
    ("defaults/every/Ultra", 0xa1ac_6903_4f5f_e0e3),
    ("defaults/every/Custom", 0xa1ac_6903_4f5f_e0e3),
    ("defaults/off/auto-Unknown", 0xc843_9c39_f193_a197),
    ("defaults/off/auto-Integrated", 0x1785_1cf3_9b37_acc7),
    ("defaults/off/auto-EntryDiscrete", 0xc843_9c39_f193_a197),
    ("defaults/off/auto-MidDiscrete", 0x5c2a_c1f6_0a7b_6544),
    ("defaults/off/auto-HighDiscrete", 0x19d3_656d_b94e_8855),
    ("defaults/off/Low", 0x1785_1cf3_9b37_acc7),
    ("defaults/off/Medium", 0xc843_9c39_f193_a197),
    ("defaults/off/High", 0x5c2a_c1f6_0a7b_6544),
    ("defaults/off/Ultra", 0x19d3_656d_b94e_8855),
    ("defaults/off/Custom", 0x19d3_656d_b94e_8855),
    ("above/none/auto-Unknown", 0xfe92_a307_9798_d543),
    ("above/none/auto-Integrated", 0x0d55_98fd_9b99_b0c1),
    ("above/none/auto-EntryDiscrete", 0xfe92_a307_9798_d543),
    ("above/none/auto-MidDiscrete", 0x3647_f464_56ae_3fd7),
    ("above/none/auto-HighDiscrete", 0x30a4_fa34_9723_449f),
    ("above/none/Low", 0x0d55_98fd_9b99_b0c1),
    ("above/none/Medium", 0xfe92_a307_9798_d543),
    ("above/none/High", 0x3647_f464_56ae_3fd7),
    ("above/none/Ultra", 0x30a4_fa34_9723_449f),
    ("above/none/Custom", 0x765a_8fe8_5d79_e53f),
    ("above/every/auto-Unknown", 0x2fe5_1adf_126d_6c17),
    ("above/every/auto-Integrated", 0x6899_09c7_96c4_1282),
    ("above/every/auto-EntryDiscrete", 0x2fe5_1adf_126d_6c17),
    ("above/every/auto-MidDiscrete", 0x799e_b759_5f1e_42d4),
    ("above/every/auto-HighDiscrete", 0x2ec7_a5c4_6b69_502e),
    ("above/every/Low", 0x6899_09c7_96c4_1282),
    ("above/every/Medium", 0x2fe5_1adf_126d_6c17),
    ("above/every/High", 0x799e_b759_5f1e_42d4),
    ("above/every/Ultra", 0x2ec7_a5c4_6b69_502e),
    ("above/every/Custom", 0xa2c2_862f_a3bb_90c4),
    ("above/off/auto-Unknown", 0xa89e_dcfa_5e97_a3cc),
    ("above/off/auto-Integrated", 0x589c_a6ce_609a_665a),
    ("above/off/auto-EntryDiscrete", 0xa89e_dcfa_5e97_a3cc),
    ("above/off/auto-MidDiscrete", 0x5ad4_02da_3bd4_f253),
    ("above/off/auto-HighDiscrete", 0x1e11_5f3a_57eb_5d91),
    ("above/off/Low", 0x589c_a6ce_609a_665a),
    ("above/off/Medium", 0xa89e_dcfa_5e97_a3cc),
    ("above/off/High", 0x5ad4_02da_3bd4_f253),
    ("above/off/Ultra", 0x1e11_5f3a_57eb_5d91),
    ("above/off/Custom", 0x8b66_d766_e3f0_f7e9),
    ("below/none/auto-Unknown", 0x44a6_a0d1_ed70_21d5),
    ("below/none/auto-Integrated", 0x2571_d168_56b0_b236),
    ("below/none/auto-EntryDiscrete", 0x44a6_a0d1_ed70_21d5),
    ("below/none/auto-MidDiscrete", 0x7ff2_1703_a84f_4dfe),
    ("below/none/auto-HighDiscrete", 0xdb40_d6ff_5643_7e00),
    ("below/none/Low", 0x2571_d168_56b0_b236),
    ("below/none/Medium", 0x44a6_a0d1_ed70_21d5),
    ("below/none/High", 0x7ff2_1703_a84f_4dfe),
    ("below/none/Ultra", 0xdb40_d6ff_5643_7e00),
    ("below/none/Custom", 0xdb40_d6ff_5643_7e00),
    ("below/every/auto-Unknown", 0xf16a_62b3_5584_c512),
    ("below/every/auto-Integrated", 0xdf64_5588_8da5_2fcf),
    ("below/every/auto-EntryDiscrete", 0xf16a_62b3_5584_c512),
    ("below/every/auto-MidDiscrete", 0xa520_798d_6f26_5dd5),
    ("below/every/auto-HighDiscrete", 0xdbd7_9ccb_5e51_8bff),
    ("below/every/Low", 0xdf64_5588_8da5_2fcf),
    ("below/every/Medium", 0xf16a_62b3_5584_c512),
    ("below/every/High", 0xa520_798d_6f26_5dd5),
    ("below/every/Ultra", 0xdbd7_9ccb_5e51_8bff),
    ("below/every/Custom", 0xdbd7_9ccb_5e51_8bff),
    ("below/off/auto-Unknown", 0x50a4_0b13_af27_2056),
    ("below/off/auto-Integrated", 0x6d35_30a3_fc93_af09),
    ("below/off/auto-EntryDiscrete", 0x50a4_0b13_af27_2056),
    ("below/off/auto-MidDiscrete", 0xeb31_6b66_d6dd_f225),
    ("below/off/auto-HighDiscrete", 0x15b6_006f_9217_fb97),
    ("below/off/Low", 0x6d35_30a3_fc93_af09),
    ("below/off/Medium", 0x50a4_0b13_af27_2056),
    ("below/off/High", 0xeb31_6b66_d6dd_f225),
    ("below/off/Ultra", 0x15b6_006f_9217_fb97),
    ("below/off/Custom", 0x15b6_006f_9217_fb97),
    ("odd/none/auto-Unknown", 0xe346_7455_736d_b255),
    ("odd/none/auto-Integrated", 0xacaf_ec40_60ac_bdc2),
    ("odd/none/auto-EntryDiscrete", 0xe346_7455_736d_b255),
    ("odd/none/auto-MidDiscrete", 0xc3eb_e384_6a77_5b45),
    ("odd/none/auto-HighDiscrete", 0xc297_5b50_dad0_83fd),
    ("odd/none/Low", 0xacaf_ec40_60ac_bdc2),
    ("odd/none/Medium", 0xe346_7455_736d_b255),
    ("odd/none/High", 0xc3eb_e384_6a77_5b45),
    ("odd/none/Ultra", 0xc297_5b50_dad0_83fd),
    ("odd/none/Custom", 0x2899_b2a1_9f52_ec41),
    ("odd/every/auto-Unknown", 0x6466_a4e7_55bf_247a),
    ("odd/every/auto-Integrated", 0x4661_8393_2b6e_cb65),
    ("odd/every/auto-EntryDiscrete", 0x6466_a4e7_55bf_247a),
    ("odd/every/auto-MidDiscrete", 0x6642_b2ed_93ed_13df),
    ("odd/every/auto-HighDiscrete", 0xe394_9a12_9609_06e4),
    ("odd/every/Low", 0x4661_8393_2b6e_cb65),
    ("odd/every/Medium", 0x6466_a4e7_55bf_247a),
    ("odd/every/High", 0x6642_b2ed_93ed_13df),
    ("odd/every/Ultra", 0xe394_9a12_9609_06e4),
    ("odd/every/Custom", 0xfdbf_3c10_c6fe_f1fe),
    ("odd/off/auto-Unknown", 0xa104_30d6_67df_5f6f),
    ("odd/off/auto-Integrated", 0x23dc_0e00_bc0d_9399),
    ("odd/off/auto-EntryDiscrete", 0xa104_30d6_67df_5f6f),
    ("odd/off/auto-MidDiscrete", 0x15ff_e73c_7af0_833c),
    ("odd/off/auto-HighDiscrete", 0x223e_ea57_9833_ad5b),
    ("odd/off/Low", 0x23dc_0e00_bc0d_9399),
    ("odd/off/Medium", 0xa104_30d6_67df_5f6f),
    ("odd/off/High", 0x15ff_e73c_7af0_833c),
    ("odd/off/Ultra", 0x223e_ea57_9833_ad5b),
    ("odd/off/Custom", 0x2816_d1bf_c0ea_7af3),
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
