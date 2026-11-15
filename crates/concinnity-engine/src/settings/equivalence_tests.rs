// Pins what the settings chain hands the backend for a matrix of inputs: every
// preset (Auto over each GPU tier), worlds authored above and below every
// ceiling, and the user's overrides on and off. Each case folds the resolved
// settings, what the backend derives from them, and a preset change's
// re-resolved quality group into one digest, so a change to any value the
// backend receives fails here by name, with the case's text to diff.

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
    ("bare/none/auto-Unknown", 0x8fc2_6342_5553_e80c),
    ("bare/none/auto-Integrated", 0x2e4f_9105_0feb_2a28),
    ("bare/none/auto-EntryDiscrete", 0x8fc2_6342_5553_e80c),
    ("bare/none/auto-MidDiscrete", 0x5e2f_53a0_c8af_f09d),
    ("bare/none/auto-HighDiscrete", 0x841d_eada_11c0_75a1),
    ("bare/none/Low", 0x2e4f_9105_0feb_2a28),
    ("bare/none/Medium", 0x8fc2_6342_5553_e80c),
    ("bare/none/High", 0x5e2f_53a0_c8af_f09d),
    ("bare/none/Ultra", 0x841d_eada_11c0_75a1),
    ("bare/none/Custom", 0x6b56_b7f9_09cc_36f4),
    ("bare/every/auto-Unknown", 0xa0d4_19fd_9ab8_3f02),
    ("bare/every/auto-Integrated", 0xf88c_bf30_4ed6_d67d),
    ("bare/every/auto-EntryDiscrete", 0xa0d4_19fd_9ab8_3f02),
    ("bare/every/auto-MidDiscrete", 0x297b_f7a1_b58a_251b),
    ("bare/every/auto-HighDiscrete", 0xc16c_e237_dde2_1c56),
    ("bare/every/Low", 0xf88c_bf30_4ed6_d67d),
    ("bare/every/Medium", 0xa0d4_19fd_9ab8_3f02),
    ("bare/every/High", 0x297b_f7a1_b58a_251b),
    ("bare/every/Ultra", 0xc16c_e237_dde2_1c56),
    ("bare/every/Custom", 0x0529_156c_fd68_f527),
    ("bare/off/auto-Unknown", 0x6a25_8e31_def2_e76f),
    ("bare/off/auto-Integrated", 0x4844_89e9_3d44_11a0),
    ("bare/off/auto-EntryDiscrete", 0x6a25_8e31_def2_e76f),
    ("bare/off/auto-MidDiscrete", 0x4fac_2988_55f9_811e),
    ("bare/off/auto-HighDiscrete", 0xc017_dd93_1cc0_f601),
    ("bare/off/Low", 0x4844_89e9_3d44_11a0),
    ("bare/off/Medium", 0x6a25_8e31_def2_e76f),
    ("bare/off/High", 0x4fac_2988_55f9_811e),
    ("bare/off/Ultra", 0xc017_dd93_1cc0_f601),
    ("bare/off/Custom", 0xb99b_0479_409c_2b62),
    ("defaults/none/auto-Unknown", 0xe748_703f_8512_203c),
    ("defaults/none/auto-Integrated", 0x2e4f_9105_0feb_2a28),
    ("defaults/none/auto-EntryDiscrete", 0xe748_703f_8512_203c),
    ("defaults/none/auto-MidDiscrete", 0xc032_c4d0_ab30_c1e1),
    ("defaults/none/auto-HighDiscrete", 0xc265_f32d_001a_3463),
    ("defaults/none/Low", 0x2e4f_9105_0feb_2a28),
    ("defaults/none/Medium", 0xe748_703f_8512_203c),
    ("defaults/none/High", 0xc032_c4d0_ab30_c1e1),
    ("defaults/none/Ultra", 0xc265_f32d_001a_3463),
    ("defaults/none/Custom", 0xd830_e80c_f076_fada),
    ("defaults/every/auto-Unknown", 0x8bcc_0e7b_515b_e1be),
    ("defaults/every/auto-Integrated", 0x44b0_4ddc_3da9_0e43),
    ("defaults/every/auto-EntryDiscrete", 0x8bcc_0e7b_515b_e1be),
    ("defaults/every/auto-MidDiscrete", 0x7fd4_1d59_9864_f159),
    ("defaults/every/auto-HighDiscrete", 0x7faf_df2f_0dc8_8112),
    ("defaults/every/Low", 0x44b0_4ddc_3da9_0e43),
    ("defaults/every/Medium", 0x8bcc_0e7b_515b_e1be),
    ("defaults/every/High", 0x7fd4_1d59_9864_f159),
    ("defaults/every/Ultra", 0x7faf_df2f_0dc8_8112),
    ("defaults/every/Custom", 0x883e_27ca_5ad6_3fc7),
    ("defaults/off/auto-Unknown", 0x0d39_65bd_3cc7_b725),
    ("defaults/off/auto-Integrated", 0x4844_89e9_3d44_11a0),
    ("defaults/off/auto-EntryDiscrete", 0x0d39_65bd_3cc7_b725),
    ("defaults/off/auto-MidDiscrete", 0x7245_2609_750e_3093),
    ("defaults/off/auto-HighDiscrete", 0xf432_e88c_675a_beb6),
    ("defaults/off/Low", 0x4844_89e9_3d44_11a0),
    ("defaults/off/Medium", 0x0d39_65bd_3cc7_b725),
    ("defaults/off/High", 0x7245_2609_750e_3093),
    ("defaults/off/Ultra", 0xf432_e88c_675a_beb6),
    ("defaults/off/Custom", 0x1699_0bda_fa24_4939),
    ("above/none/auto-Unknown", 0x6f89_d938_da67_ff8e),
    ("above/none/auto-Integrated", 0x7c2e_d65b_70f6_5941),
    ("above/none/auto-EntryDiscrete", 0x6f89_d938_da67_ff8e),
    ("above/none/auto-MidDiscrete", 0x89b9_3f2b_9a95_cb86),
    ("above/none/auto-HighDiscrete", 0x9eaa_a59c_77a6_efea),
    ("above/none/Low", 0x7c2e_d65b_70f6_5941),
    ("above/none/Medium", 0x6f89_d938_da67_ff8e),
    ("above/none/High", 0x89b9_3f2b_9a95_cb86),
    ("above/none/Ultra", 0x9eaa_a59c_77a6_efea),
    ("above/none/Custom", 0x96b7_a455_66e2_ada1),
    ("above/every/auto-Unknown", 0xd4c9_8809_bbc1_f6a2),
    ("above/every/auto-Integrated", 0x0d66_3d7b_1d0b_193d),
    ("above/every/auto-EntryDiscrete", 0xd4c9_8809_bbc1_f6a2),
    ("above/every/auto-MidDiscrete", 0x6506_a2db_1176_706b),
    ("above/every/auto-HighDiscrete", 0xccc3_622d_65cf_cec1),
    ("above/every/Low", 0x0d66_3d7b_1d0b_193d),
    ("above/every/Medium", 0xd4c9_8809_bbc1_f6a2),
    ("above/every/High", 0x6506_a2db_1176_706b),
    ("above/every/Ultra", 0xccc3_622d_65cf_cec1),
    ("above/every/Custom", 0x7996_fe2d_0b4e_27d2),
    ("above/off/auto-Unknown", 0xae0f_185a_88ff_4b4a),
    ("above/off/auto-Integrated", 0x7aff_781d_ffc2_9f36),
    ("above/off/auto-EntryDiscrete", 0xae0f_185a_88ff_4b4a),
    ("above/off/auto-MidDiscrete", 0x3a14_5fbc_4e03_d58b),
    ("above/off/auto-HighDiscrete", 0xb9e1_b554_160c_61e3),
    ("above/off/Low", 0x7aff_781d_ffc2_9f36),
    ("above/off/Medium", 0xae0f_185a_88ff_4b4a),
    ("above/off/High", 0x3a14_5fbc_4e03_d58b),
    ("above/off/Ultra", 0xb9e1_b554_160c_61e3),
    ("above/off/Custom", 0xb857_b914_f456_e12a),
    ("below/none/auto-Unknown", 0xb476_3f58_bda6_ee18),
    ("below/none/auto-Integrated", 0x6fbc_62c5_4d37_925d),
    ("below/none/auto-EntryDiscrete", 0xb476_3f58_bda6_ee18),
    ("below/none/auto-MidDiscrete", 0x12cb_7c7e_8c38_af9b),
    ("below/none/auto-HighDiscrete", 0x8883_ab53_dab0_c9f1),
    ("below/none/Low", 0x6fbc_62c5_4d37_925d),
    ("below/none/Medium", 0xb476_3f58_bda6_ee18),
    ("below/none/High", 0x12cb_7c7e_8c38_af9b),
    ("below/none/Ultra", 0x8883_ab53_dab0_c9f1),
    ("below/none/Custom", 0x1712_2316_84c1_6868),
    ("below/every/auto-Unknown", 0x8692_72a8_e765_8463),
    ("below/every/auto-Integrated", 0x00de_1e7b_64bc_c7b8),
    ("below/every/auto-EntryDiscrete", 0x8692_72a8_e765_8463),
    ("below/every/auto-MidDiscrete", 0xbed7_0a4e_1c06_d600),
    ("below/every/auto-HighDiscrete", 0xbc7e_8f44_5801_0262),
    ("below/every/Low", 0x00de_1e7b_64bc_c7b8),
    ("below/every/Medium", 0x8692_72a8_e765_8463),
    ("below/every/High", 0xbed7_0a4e_1c06_d600),
    ("below/every/Ultra", 0xbc7e_8f44_5801_0262),
    ("below/every/Custom", 0xd049_04c0_90c9_e2a3),
    ("below/off/auto-Unknown", 0xe96c_a0cb_66c9_6bd1),
    ("below/off/auto-Integrated", 0x437d_a921_cb3b_bcc2),
    ("below/off/auto-EntryDiscrete", 0xe96c_a0cb_66c9_6bd1),
    ("below/off/auto-MidDiscrete", 0x82ae_4037_f41a_b5c2),
    ("below/off/auto-HighDiscrete", 0xd65d_f9a2_9b12_8d84),
    ("below/off/Low", 0x437d_a921_cb3b_bcc2),
    ("below/off/Medium", 0xe96c_a0cb_66c9_6bd1),
    ("below/off/High", 0x82ae_4037_f41a_b5c2),
    ("below/off/Ultra", 0xd65d_f9a2_9b12_8d84),
    ("below/off/Custom", 0x8940_0551_2b31_2103),
    ("odd/none/auto-Unknown", 0xe2f0_16b1_fd34_6c85),
    ("odd/none/auto-Integrated", 0xbd75_9dd6_ac5b_b135),
    ("odd/none/auto-EntryDiscrete", 0xe2f0_16b1_fd34_6c85),
    ("odd/none/auto-MidDiscrete", 0x84d3_6161_8b0b_dc12),
    ("odd/none/auto-HighDiscrete", 0x3461_d53e_41a5_0ebc),
    ("odd/none/Low", 0xbd75_9dd6_ac5b_b135),
    ("odd/none/Medium", 0xe2f0_16b1_fd34_6c85),
    ("odd/none/High", 0x84d3_6161_8b0b_dc12),
    ("odd/none/Ultra", 0x3461_d53e_41a5_0ebc),
    ("odd/none/Custom", 0x6d89_86cd_b04d_7ce7),
    ("odd/every/auto-Unknown", 0x1270_0eeb_3282_490d),
    ("odd/every/auto-Integrated", 0x9951_dbab_dfb9_fac6),
    ("odd/every/auto-EntryDiscrete", 0x1270_0eeb_3282_490d),
    ("odd/every/auto-MidDiscrete", 0xeab5_5261_4f3d_91c4),
    ("odd/every/auto-HighDiscrete", 0x079b_90fa_f758_3fb7),
    ("odd/every/Low", 0x9951_dbab_dfb9_fac6),
    ("odd/every/Medium", 0x1270_0eeb_3282_490d),
    ("odd/every/High", 0xeab5_5261_4f3d_91c4),
    ("odd/every/Ultra", 0x079b_90fa_f758_3fb7),
    ("odd/every/Custom", 0x7e07_c558_d316_4f40),
    ("odd/off/auto-Unknown", 0xaaca_d749_ce38_a749),
    ("odd/off/auto-Integrated", 0x4423_1464_8b14_4d1a),
    ("odd/off/auto-EntryDiscrete", 0xaaca_d749_ce38_a749),
    ("odd/off/auto-MidDiscrete", 0x4a97_6a6f_975a_7c33),
    ("odd/off/auto-HighDiscrete", 0x54c1_c97d_fcf5_216c),
    ("odd/off/Low", 0x4423_1464_8b14_4d1a),
    ("odd/off/Medium", 0xaaca_d749_ce38_a749),
    ("odd/off/High", 0x4a97_6a6f_975a_7c33),
    ("odd/off/Ultra", 0x54c1_c97d_fcf5_216c),
    ("odd/off/Custom", 0x597d_4172_3a03_fa63),
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
