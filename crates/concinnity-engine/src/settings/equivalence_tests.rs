// Pins what the settings chain hands the backend across every preset, worlds
// above and below every ceiling, and user overrides on and off. Each case
// digests the resolved settings, the backend's derivations and a preset
// change's re-resolution, so any change fails here by name.

use crate::components::BackgroundOverrides;
use crate::components::{PlanetDetailOverrides, RelevanceOverrides};
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
            relevance: RelevanceOverrides::default(),
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
    ("bare/none/auto-Unknown", 0x09ff_abcb_0c21_0aa8),
    ("bare/none/auto-Integrated", 0x4a2b_73be_feaf_7da8),
    ("bare/none/auto-EntryDiscrete", 0x09ff_abcb_0c21_0aa8),
    ("bare/none/auto-MidDiscrete", 0x841c_13f1_37d3_85e5),
    ("bare/none/auto-HighDiscrete", 0xe810_6a96_21c8_396d),
    ("bare/none/Low", 0x4a2b_73be_feaf_7da8),
    ("bare/none/Medium", 0x09ff_abcb_0c21_0aa8),
    ("bare/none/High", 0x841c_13f1_37d3_85e5),
    ("bare/none/Ultra", 0xe810_6a96_21c8_396d),
    ("bare/none/Custom", 0xf28e_f747_1435_d200),
    ("bare/every/auto-Unknown", 0xca63_f9d0_71fa_3066),
    ("bare/every/auto-Integrated", 0x5697_4d17_4c28_2849),
    ("bare/every/auto-EntryDiscrete", 0xca63_f9d0_71fa_3066),
    ("bare/every/auto-MidDiscrete", 0x6acc_bade_52a3_4dcf),
    ("bare/every/auto-HighDiscrete", 0x1412_dd5a_3536_6d5e),
    ("bare/every/Low", 0x5697_4d17_4c28_2849),
    ("bare/every/Medium", 0xca63_f9d0_71fa_3066),
    ("bare/every/High", 0x6acc_bade_52a3_4dcf),
    ("bare/every/Ultra", 0x1412_dd5a_3536_6d5e),
    ("bare/every/Custom", 0xef55_ec6a_436e_649f),
    ("bare/off/auto-Unknown", 0xcc9f_4c9f_961c_c35f),
    ("bare/off/auto-Integrated", 0x07e2_4920_27c7_4b4c),
    ("bare/off/auto-EntryDiscrete", 0xcc9f_4c9f_961c_c35f),
    ("bare/off/auto-MidDiscrete", 0xfce9_abd3_9b54_43f2),
    ("bare/off/auto-HighDiscrete", 0x8699_4f1e_e5a2_e551),
    ("bare/off/Low", 0x07e2_4920_27c7_4b4c),
    ("bare/off/Medium", 0xcc9f_4c9f_961c_c35f),
    ("bare/off/High", 0xfce9_abd3_9b54_43f2),
    ("bare/off/Ultra", 0x8699_4f1e_e5a2_e551),
    ("bare/off/Custom", 0xd8d9_027c_909a_e2b2),
    ("defaults/none/auto-Unknown", 0x5056_f266_9f63_655c),
    ("defaults/none/auto-Integrated", 0x4a2b_73be_feaf_7da8),
    ("defaults/none/auto-EntryDiscrete", 0x5056_f266_9f63_655c),
    ("defaults/none/auto-MidDiscrete", 0xcb0d_dd8f_1fc1_cc95),
    ("defaults/none/auto-HighDiscrete", 0x036e_b372_be99_0633),
    ("defaults/none/Low", 0x4a2b_73be_feaf_7da8),
    ("defaults/none/Medium", 0x5056_f266_9f63_655c),
    ("defaults/none/High", 0xcb0d_dd8f_1fc1_cc95),
    ("defaults/none/Ultra", 0x036e_b372_be99_0633),
    ("defaults/none/Custom", 0xd1da_aa3e_31d6_fe82),
    ("defaults/every/auto-Unknown", 0x813f_c468_0f34_eede),
    ("defaults/every/auto-Integrated", 0x40bc_cd24_0aff_f7eb),
    ("defaults/every/auto-EntryDiscrete", 0x813f_c468_0f34_eede),
    ("defaults/every/auto-MidDiscrete", 0x0af5_6960_a985_8179),
    ("defaults/every/auto-HighDiscrete", 0xf021_881b_1124_dc66),
    ("defaults/every/Low", 0x40bc_cd24_0aff_f7eb),
    ("defaults/every/Medium", 0x813f_c468_0f34_eede),
    ("defaults/every/High", 0x0af5_6960_a985_8179),
    ("defaults/every/Ultra", 0xf021_881b_1124_dc66),
    ("defaults/every/Custom", 0x52fa_dec3_4876_3ea3),
    ("defaults/off/auto-Unknown", 0x369e_c150_d7b5_eb61),
    ("defaults/off/auto-Integrated", 0x07e2_4920_27c7_4b4c),
    ("defaults/off/auto-EntryDiscrete", 0x369e_c150_d7b5_eb61),
    ("defaults/off/auto-MidDiscrete", 0xe9a4_9177_0374_123b),
    ("defaults/off/auto-HighDiscrete", 0x5797_7c29_8c46_8b0a),
    ("defaults/off/Low", 0x07e2_4920_27c7_4b4c),
    ("defaults/off/Medium", 0x369e_c150_d7b5_eb61),
    ("defaults/off/High", 0xe9a4_9177_0374_123b),
    ("defaults/off/Ultra", 0x5797_7c29_8c46_8b0a),
    ("defaults/off/Custom", 0xb0af_4390_b0c8_c655),
    ("above/none/auto-Unknown", 0x87d2_d8ea_e915_e532),
    ("above/none/auto-Integrated", 0x5f70_ff68_c1cd_9459),
    ("above/none/auto-EntryDiscrete", 0x87d2_d8ea_e915_e532),
    ("above/none/auto-MidDiscrete", 0x78a3_2738_2b8f_40b6),
    ("above/none/auto-HighDiscrete", 0xeb78_095e_cea5_a446),
    ("above/none/Low", 0x5f70_ff68_c1cd_9459),
    ("above/none/Medium", 0x87d2_d8ea_e915_e532),
    ("above/none/High", 0x78a3_2738_2b8f_40b6),
    ("above/none/Ultra", 0xeb78_095e_cea5_a446),
    ("above/none/Custom", 0x5c69_d895_7bec_da39),
    ("above/every/auto-Unknown", 0xa10a_1b77_8824_100a),
    ("above/every/auto-Integrated", 0x563e_df33_9dcf_692d),
    ("above/every/auto-EntryDiscrete", 0xa10a_1b77_8824_100a),
    ("above/every/auto-MidDiscrete", 0x25e2_b1d7_cc21_139b),
    ("above/every/auto-HighDiscrete", 0xd2a6_8e93_a246_5cfd),
    ("above/every/Low", 0x563e_df33_9dcf_692d),
    ("above/every/Medium", 0xa10a_1b77_8824_100a),
    ("above/every/High", 0x25e2_b1d7_cc21_139b),
    ("above/every/Ultra", 0xd2a6_8e93_a246_5cfd),
    ("above/every/Custom", 0xa773_28c0_cdb5_8682),
    ("above/off/auto-Unknown", 0x70ce_abbe_2c71_82fa),
    ("above/off/auto-Integrated", 0xa87e_12e9_0d44_d8a2),
    ("above/off/auto-EntryDiscrete", 0x70ce_abbe_2c71_82fa),
    ("above/off/auto-MidDiscrete", 0x2db6_eabc_4359_656f),
    ("above/off/auto-HighDiscrete", 0x4e05_c9be_6f66_b623),
    ("above/off/Low", 0xa87e_12e9_0d44_d8a2),
    ("above/off/Medium", 0x70ce_abbe_2c71_82fa),
    ("above/off/High", 0x2db6_eabc_4359_656f),
    ("above/off/Ultra", 0x4e05_c9be_6f66_b623),
    ("above/off/Custom", 0x9d74_d89b_583f_13b6),
    ("below/none/auto-Unknown", 0x7026_437f_cff4_62bc),
    ("below/none/auto-Integrated", 0xb2ca_f0b7_d7d9_11a1),
    ("below/none/auto-EntryDiscrete", 0x7026_437f_cff4_62bc),
    ("below/none/auto-MidDiscrete", 0xff7b_9b0a_0e77_8a57),
    ("below/none/auto-HighDiscrete", 0xf671_49b2_a583_f865),
    ("below/none/Low", 0xb2ca_f0b7_d7d9_11a1),
    ("below/none/Medium", 0x7026_437f_cff4_62bc),
    ("below/none/High", 0xff7b_9b0a_0e77_8a57),
    ("below/none/Ultra", 0xf671_49b2_a583_f865),
    ("below/none/Custom", 0x4783_4c34_4e2d_2e5c),
    ("below/every/auto-Unknown", 0xc7c2_33e6_db06_3dd3),
    ("below/every/auto-Integrated", 0xd708_635b_db63_e588),
    ("below/every/auto-EntryDiscrete", 0xc7c2_33e6_db06_3dd3),
    ("below/every/auto-MidDiscrete", 0xbed5_300f_56ef_dee8),
    ("below/every/auto-HighDiscrete", 0xdf88_78f1_6970_4642),
    ("below/every/Low", 0xd708_635b_db63_e588),
    ("below/every/Medium", 0xc7c2_33e6_db06_3dd3),
    ("below/every/High", 0xbed5_300f_56ef_dee8),
    ("below/every/Ultra", 0xdf88_78f1_6970_4642),
    ("below/every/Custom", 0xbd8f_c5b4_2337_7a83),
    ("below/off/auto-Unknown", 0xc4b7_4b83_dbfb_17cd),
    ("below/off/auto-Integrated", 0x80c5_0649_d03e_5e9e),
    ("below/off/auto-EntryDiscrete", 0xc4b7_4b83_dbfb_17cd),
    ("below/off/auto-MidDiscrete", 0x5671_a437_38c6_a2de),
    ("below/off/auto-HighDiscrete", 0xa594_221a_ae63_3978),
    ("below/off/Low", 0x80c5_0649_d03e_5e9e),
    ("below/off/Medium", 0xc4b7_4b83_dbfb_17cd),
    ("below/off/High", 0x5671_a437_38c6_a2de),
    ("below/off/Ultra", 0xa594_221a_ae63_3978),
    ("below/off/Custom", 0xf4ab_5cb0_3603_738f),
    ("odd/none/auto-Unknown", 0x9cdf_6518_b717_72a5),
    ("odd/none/auto-Integrated", 0x28ff_9f09_2ee6_79b5),
    ("odd/none/auto-EntryDiscrete", 0x9cdf_6518_b717_72a5),
    ("odd/none/auto-MidDiscrete", 0x6339_2c53_8b39_ad6e),
    ("odd/none/auto-HighDiscrete", 0xfd58_17c9_b1de_c08c),
    ("odd/none/Low", 0x28ff_9f09_2ee6_79b5),
    ("odd/none/Medium", 0x9cdf_6518_b717_72a5),
    ("odd/none/High", 0x6339_2c53_8b39_ad6e),
    ("odd/none/Ultra", 0xfd58_17c9_b1de_c08c),
    ("odd/none/Custom", 0x673d_ad60_e967_3863),
    ("odd/every/auto-Unknown", 0x83f5_adbd_ec7f_db09),
    ("odd/every/auto-Integrated", 0xe6d5_bacc_dc77_9d0a),
    ("odd/every/auto-EntryDiscrete", 0x83f5_adbd_ec7f_db09),
    ("odd/every/auto-MidDiscrete", 0x5b90_2ba3_cde5_76b8),
    ("odd/every/auto-HighDiscrete", 0x01f5_9e33_1b28_30f7),
    ("odd/every/Low", 0xe6d5_bacc_dc77_9d0a),
    ("odd/every/Medium", 0x83f5_adbd_ec7f_db09),
    ("odd/every/High", 0x5b90_2ba3_cde5_76b8),
    ("odd/every/Ultra", 0x01f5_9e33_1b28_30f7),
    ("odd/every/Custom", 0xb0cf_bddb_c9df_944c),
    ("odd/off/auto-Unknown", 0xfc03_7848_e98e_d279),
    ("odd/off/auto-Integrated", 0x47d5_8fa0_9bbd_f142),
    ("odd/off/auto-EntryDiscrete", 0xfc03_7848_e98e_d279),
    ("odd/off/auto-MidDiscrete", 0xa519_f0b5_85d8_4b5f),
    ("odd/off/auto-HighDiscrete", 0x7b28_8e28_24cb_24ec),
    ("odd/off/Low", 0x47d5_8fa0_9bbd_f142),
    ("odd/off/Medium", 0xfc03_7848_e98e_d279),
    ("odd/off/High", 0xa519_f0b5_85d8_4b5f),
    ("odd/off/Ultra", 0x7b28_8e28_24cb_24ec),
    ("odd/off/Custom", 0x9a86_b6f1_eca6_df7f),
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

// What a `bare` case resolves for render scale and two-pass occlusion,
// written out from the rules.
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
