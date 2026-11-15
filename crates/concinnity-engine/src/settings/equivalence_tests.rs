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
            streaming,
            ..Default::default()
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
            (
                g.upscaler_scale(),
                g.shadow_cadence(),
                g.ambient_intensity()
            ),
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
        let g = &resolved.graphics;
        // TAA at a preset that renders natively takes the native scale.
        let native_taa = user.temporal_upscaling.is_none() && g.anti_aliasing.mode == AaMode::Taa;
        let render_scale = user.render_scale.unwrap_or_else(|| {
            if native_taa {
                UpscaleQuality::Native
            } else {
                more_aggressive_upscale(UpscaleQuality::default(), ceiling.min_upscale)
            }
        });
        assert_eq!(g.anti_aliasing.upscale_quality, render_scale);
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
    ("bare/none/auto-Unknown", 0xde82_6c29_6290_3b42),
    ("bare/none/auto-Integrated", 0x2f89_abec_b199_8902),
    ("bare/none/auto-EntryDiscrete", 0xde82_6c29_6290_3b42),
    ("bare/none/auto-MidDiscrete", 0x08af_b43b_1e37_e40b),
    ("bare/none/auto-HighDiscrete", 0xbe50_fc85_244a_8a8d),
    ("bare/none/Low", 0x2f89_abec_b199_8902),
    ("bare/none/Medium", 0xde82_6c29_6290_3b42),
    ("bare/none/High", 0x08af_b43b_1e37_e40b),
    ("bare/none/Ultra", 0xbe50_fc85_244a_8a8d),
    ("bare/none/Custom", 0x2a73_2575_266a_0308),
    ("bare/every/auto-Unknown", 0xad4f_cf52_58d5_46a6),
    ("bare/every/auto-Integrated", 0xc2c6_a9d5_cc20_69db),
    ("bare/every/auto-EntryDiscrete", 0xad4f_cf52_58d5_46a6),
    ("bare/every/auto-MidDiscrete", 0x71e8_95f6_fbe9_167d),
    ("bare/every/auto-HighDiscrete", 0x3623_6d88_103a_d4ce),
    ("bare/every/Low", 0xc2c6_a9d5_cc20_69db),
    ("bare/every/Medium", 0xad4f_cf52_58d5_46a6),
    ("bare/every/High", 0x71e8_95f6_fbe9_167d),
    ("bare/every/Ultra", 0x3623_6d88_103a_d4ce),
    ("bare/every/Custom", 0x5b8c_0ea8_0a95_eab7),
    ("bare/off/auto-Unknown", 0x50ee_88fc_1f50_df67),
    ("bare/off/auto-Integrated", 0x22d5_f744_9141_e9a0),
    ("bare/off/auto-EntryDiscrete", 0x50ee_88fc_1f50_df67),
    ("bare/off/auto-MidDiscrete", 0xeef1_5214_6df3_ddda),
    ("bare/off/auto-HighDiscrete", 0xf937_4181_4127_6013),
    ("bare/off/Low", 0x22d5_f744_9141_e9a0),
    ("bare/off/Medium", 0x50ee_88fc_1f50_df67),
    ("bare/off/High", 0xeef1_5214_6df3_ddda),
    ("bare/off/Ultra", 0xf937_4181_4127_6013),
    ("bare/off/Custom", 0xc467_78c7_8d03_63b0),
    ("defaults/none/auto-Unknown", 0x176b_569d_64cc_690c),
    ("defaults/none/auto-Integrated", 0x2f89_abec_b199_8902),
    ("defaults/none/auto-EntryDiscrete", 0x176b_569d_64cc_690c),
    ("defaults/none/auto-MidDiscrete", 0xdd36_ace7_b5d5_f5e3),
    ("defaults/none/auto-HighDiscrete", 0x4e7f_eb8f_7ada_87ab),
    ("defaults/none/Low", 0x2f89_abec_b199_8902),
    ("defaults/none/Medium", 0x176b_569d_64cc_690c),
    ("defaults/none/High", 0xdd36_ace7_b5d5_f5e3),
    ("defaults/none/Ultra", 0x4e7f_eb8f_7ada_87ab),
    ("defaults/none/Custom", 0x4d31_82f6_6e87_ab9a),
    ("defaults/every/auto-Unknown", 0xa343_e9dc_4dfe_2944),
    ("defaults/every/auto-Integrated", 0x9476_aa95_14d9_e783),
    ("defaults/every/auto-EntryDiscrete", 0xa343_e9dc_4dfe_2944),
    ("defaults/every/auto-MidDiscrete", 0xf0b6_75f6_4da9_15d1),
    ("defaults/every/auto-HighDiscrete", 0xe92e_b797_8752_6f30),
    ("defaults/every/Low", 0x9476_aa95_14d9_e783),
    ("defaults/every/Medium", 0xa343_e9dc_4dfe_2944),
    ("defaults/every/High", 0xf0b6_75f6_4da9_15d1),
    ("defaults/every/Ultra", 0xe92e_b797_8752_6f30),
    ("defaults/every/Custom", 0x5fc5_22d3_e20a_2c39),
    ("defaults/off/auto-Unknown", 0x4cb0_a186_2bea_eb5f),
    ("defaults/off/auto-Integrated", 0x22d5_f744_9141_e9a0),
    ("defaults/off/auto-EntryDiscrete", 0x4cb0_a186_2bea_eb5f),
    ("defaults/off/auto-MidDiscrete", 0x8bf7_9a18_292c_8a57),
    ("defaults/off/auto-HighDiscrete", 0x524f_579e_1b27_ed8c),
    ("defaults/off/Low", 0x22d5_f744_9141_e9a0),
    ("defaults/off/Medium", 0x4cb0_a186_2bea_eb5f),
    ("defaults/off/High", 0x8bf7_9a18_292c_8a57),
    ("defaults/off/Ultra", 0x524f_579e_1b27_ed8c),
    ("defaults/off/Custom", 0x7f9a_e367_0c81_6883),
    ("above/none/auto-Unknown", 0xd366_6692_e989_1418),
    ("above/none/auto-Integrated", 0x31f2_ade1_6552_71c1),
    ("above/none/auto-EntryDiscrete", 0xd366_6692_e989_1418),
    ("above/none/auto-MidDiscrete", 0x6dc3_1554_59a0_9922),
    ("above/none/auto-HighDiscrete", 0x5cc1_4ba9_f10d_1a08),
    ("above/none/Low", 0x31f2_ade1_6552_71c1),
    ("above/none/Medium", 0xd366_6692_e989_1418),
    ("above/none/High", 0x6dc3_1554_59a0_9922),
    ("above/none/Ultra", 0x5cc1_4ba9_f10d_1a08),
    ("above/none/Custom", 0x30c5_930b_6ecf_5363),
    ("above/every/auto-Unknown", 0x1f44_f1e1_8141_5862),
    ("above/every/auto-Integrated", 0x5b5b_6eeb_1b09_fa2f),
    ("above/every/auto-EntryDiscrete", 0x1f44_f1e1_8141_5862),
    ("above/every/auto-MidDiscrete", 0x0cd9_b7ba_a149_18c1),
    ("above/every/auto-HighDiscrete", 0x6c9b_4fbc_e3b4_5efd),
    ("above/every/Low", 0x5b5b_6eeb_1b09_fa2f),
    ("above/every/Medium", 0x1f44_f1e1_8141_5862),
    ("above/every/High", 0x0cd9_b7ba_a149_18c1),
    ("above/every/Ultra", 0x6c9b_4fbc_e3b4_5efd),
    ("above/every/Custom", 0x4c94_db2f_47a1_8c9e),
    ("above/off/auto-Unknown", 0xe774_0268_9586_4216),
    ("above/off/auto-Integrated", 0x18b5_ae57_2ce2_72c8),
    ("above/off/auto-EntryDiscrete", 0xe774_0268_9586_4216),
    ("above/off/auto-MidDiscrete", 0xa1e9_cf28_755c_7ab5),
    ("above/off/auto-HighDiscrete", 0x1dd1_5052_c282_39cf),
    ("above/off/Low", 0x18b5_ae57_2ce2_72c8),
    ("above/off/Medium", 0xe774_0268_9586_4216),
    ("above/off/High", 0xa1e9_cf28_755c_7ab5),
    ("above/off/Ultra", 0x1dd1_5052_c282_39cf),
    ("above/off/Custom", 0x6ec8_8690_898c_a482),
    ("below/none/auto-Unknown", 0x4ee7_ded9_75f3_8e06),
    ("below/none/auto-Integrated", 0x1e3f_644f_a66a_a6cb),
    ("below/none/auto-EntryDiscrete", 0x4ee7_ded9_75f3_8e06),
    ("below/none/auto-MidDiscrete", 0x2947_b59b_3bfe_bcc1),
    ("below/none/auto-HighDiscrete", 0xcb8c_2fce_1fcc_e429),
    ("below/none/Low", 0x1e3f_644f_a66a_a6cb),
    ("below/none/Medium", 0x4ee7_ded9_75f3_8e06),
    ("below/none/High", 0x2947_b59b_3bfe_bcc1),
    ("below/none/Ultra", 0xcb8c_2fce_1fcc_e429),
    ("below/none/Custom", 0x8656_22db_7d7c_9560),
    ("below/every/auto-Unknown", 0xb5d4_cc51_4ab2_33b3),
    ("below/every/auto-Integrated", 0xe7be_6e0b_46fc_9ca4),
    ("below/every/auto-EntryDiscrete", 0xb5d4_cc51_4ab2_33b3),
    ("below/every/auto-MidDiscrete", 0x07c1_1e60_2e1c_8a7c),
    ("below/every/auto-HighDiscrete", 0xfb9c_e80c_272b_a3d8),
    ("below/every/Low", 0xe7be_6e0b_46fc_9ca4),
    ("below/every/Medium", 0xb5d4_cc51_4ab2_33b3),
    ("below/every/High", 0x07c1_1e60_2e1c_8a7c),
    ("below/every/Ultra", 0xfb9c_e80c_272b_a3d8),
    ("below/every/Custom", 0x365d_4944_b1d5_4185),
    ("below/off/auto-Unknown", 0x125f_931b_c040_5945),
    ("below/off/auto-Integrated", 0x5ee5_058b_0dca_e8fe),
    ("below/off/auto-EntryDiscrete", 0x125f_931b_c040_5945),
    ("below/off/auto-MidDiscrete", 0xa402_bf87_3e8d_18f6),
    ("below/off/auto-HighDiscrete", 0x392a_ab97_3bc2_c646),
    ("below/off/Low", 0x5ee5_058b_0dca_e8fe),
    ("below/off/Medium", 0x125f_931b_c040_5945),
    ("below/off/High", 0xa402_bf87_3e8d_18f6),
    ("below/off/Ultra", 0x392a_ab97_3bc2_c646),
    ("below/off/Custom", 0xde20_9d96_c825_3ed9),
    ("odd/none/auto-Unknown", 0xdf90_ae2a_1908_bc25),
    ("odd/none/auto-Integrated", 0x34ce_22c7_0872_19d3),
    ("odd/none/auto-EntryDiscrete", 0xdf90_ae2a_1908_bc25),
    ("odd/none/auto-MidDiscrete", 0x8818_88a6_b6e6_1a94),
    ("odd/none/auto-HighDiscrete", 0x2527_c15e_f5af_f4dc),
    ("odd/none/Low", 0x34ce_22c7_0872_19d3),
    ("odd/none/Medium", 0xdf90_ae2a_1908_bc25),
    ("odd/none/High", 0x8818_88a6_b6e6_1a94),
    ("odd/none/Ultra", 0x2527_c15e_f5af_f4dc),
    ("odd/none/Custom", 0x61fa_d83d_8d8a_a4d7),
    ("odd/every/auto-Unknown", 0xf35b_e25b_f0f6_591f),
    ("odd/every/auto-Integrated", 0xd3f1_f278_053d_f99e),
    ("odd/every/auto-EntryDiscrete", 0xf35b_e25b_f0f6_591f),
    ("odd/every/auto-MidDiscrete", 0xa8ba_be66_39e5_0500),
    ("odd/every/auto-HighDiscrete", 0x12db_9f96_0e89_e819),
    ("odd/every/Low", 0xd3f1_f278_053d_f99e),
    ("odd/every/Medium", 0xf35b_e25b_f0f6_591f),
    ("odd/every/High", 0xa8ba_be66_39e5_0500),
    ("odd/every/Ultra", 0x12db_9f96_0e89_e819),
    ("odd/every/Custom", 0xda00_5cea_b553_ad3a),
    ("odd/off/auto-Unknown", 0xb86d_848e_bcc4_38a7),
    ("odd/off/auto-Integrated", 0x147f_f689_5f4d_9dc6),
    ("odd/off/auto-EntryDiscrete", 0xb86d_848e_bcc4_38a7),
    ("odd/off/auto-MidDiscrete", 0x383a_d307_b3c2_cdf3),
    ("odd/off/auto-HighDiscrete", 0xbb94_3413_c854_c9b6),
    ("odd/off/Low", 0x147f_f689_5f4d_9dc6),
    ("odd/off/Medium", 0xb86d_848e_bcc4_38a7),
    ("odd/off/High", 0x383a_d307_b3c2_cdf3),
    ("odd/off/Ultra", 0xbb94_3413_c854_c9b6),
    ("odd/off/Custom", 0x50e0_1d31_d591_cb41),
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
// the `quality` default and the tier's minimum unless the user picked one or
// TAA takes the native scale at a preset that renders natively, and two-pass
// occlusion takes its `true` default unless the user set it.
#[test]
fn a_bare_world_resolves_to_the_section_defaults() {
    let third = 2.0 / 3.0;
    let expected = |user: &str, ceiling: &str| -> (f32, bool) {
        if user == "every" {
            return (0.587, true);
        }
        let fxaa_tier = matches!(ceiling, "Low" | "auto-Integrated");
        if user == "none" && !fxaa_tier {
            return (1.0, true);
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
            assert_eq!(
                graphics.anti_aliasing.upscale_quality.scale(),
                scale,
                "{case}"
            );
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
    let mut defaults = Declared::default().world(Some(&StreamingConfig::default()));
    // Leaving upscaling unset hands it to the preset, which an explicit
    // `false` refuses, so only its value is a default.
    defaults.upscaling = None;
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
