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

use concinnity_core::components::{
    AaMode, AmbientOcclusion, AntiAliasing, Bloom, Culling, Display, Exposure, FramePacing,
    GraphicsConfig, IndirectLighting, IndirectTechnique, PassResolution, Reflections, ShadowUpdate,
    Shadows, StreamingConfig, TextureFiltering, UpscaleQuality, UpscalerBackend, Vignette,
    WindowMode,
};
use concinnity_core::render::backend::{GpuProfile, GpuTier};
use concinnity_core::render::dlss::DlssPreset;

use concinnity_core::settings::quality::{more_aggressive_upscale, resolve_ceiling};
use concinnity_core::settings::{
    QualityPreset, ResolvedGraphics, ResolvedSettings, UserGraphicsSettings, WorldAssets,
    WorldSettings,
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
    // The world setting every field of every section, and `streaming` where
    // given.
    fn world(&self, streaming: Option<&StreamingConfig>) -> WorldSettings {
        let graphics = GraphicsConfig {
            frame_pacing: self.frame_pacing.clone().into(),
            shadows: self.shadows.clone().into(),
            texture_filtering: self.texture_filtering.clone().into(),
            background: Default::default(),
            ambient_occlusion: self.ambient_occlusion.clone().into(),
            reflections: self.reflections.clone().into(),
            indirect_lighting: self.indirect_lighting.clone().into(),
            anti_aliasing: self.anti_aliasing.clone().into(),
            exposure: self.exposure.clone().into(),
            bloom: self.bloom.clone().into(),
            vignette: self.vignette.clone().into(),
            display: self.display.clone().into(),
            culling: self.culling.clone().into(),
            planet_detail: Default::default(),
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
        bare.unwrap_or((g.upscale_scale(), g.culling.two_pass_occlusion));
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
    let mut state = SettingsState::new();
    state.authored = world.clone();
    state.session = launch.session_overrides();
    let resolved = state.resolve(user, &ceiling);

    let cleared = preset_cleared(user);
    let mut after_preset = resolved.clone();
    after_preset.set_quality(state.resolve_quality(&cleared, &ceiling));
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
                let text = case_text(world_name, &world, &user, preset, tier);
                out.push((
                    format!("{world_name}/{user_name}/{ceiling_name}"),
                    fnv1a(&text),
                ));
            }
        }
    }
    out
}

// Every case's digest as the settings chain resolves it.
const EXPECTED: &[(&str, u64)] = &[
    ("bare/none/auto-Unknown", 0x121928c9c7bf0305),
    ("bare/none/auto-Integrated", 0x716248889a98ce8e),
    ("bare/none/auto-EntryDiscrete", 0x121928c9c7bf0305),
    ("bare/none/auto-MidDiscrete", 0x85a85ac353507765),
    ("bare/none/auto-HighDiscrete", 0x195696c3b4bdba55),
    ("bare/none/Low", 0x716248889a98ce8e),
    ("bare/none/Medium", 0x121928c9c7bf0305),
    ("bare/none/High", 0x85a85ac353507765),
    ("bare/none/Ultra", 0x195696c3b4bdba55),
    ("bare/none/Custom", 0x195696c3b4bdba55),
    ("bare/every/auto-Unknown", 0x380a2cf857b5ecc4),
    ("bare/every/auto-Integrated", 0x5b62893b2ee2ea74),
    ("bare/every/auto-EntryDiscrete", 0x380a2cf857b5ecc4),
    ("bare/every/auto-MidDiscrete", 0x6a957cbb4fb7120d),
    ("bare/every/auto-HighDiscrete", 0x34b814aa97d2b7c8),
    ("bare/every/Low", 0x5b62893b2ee2ea74),
    ("bare/every/Medium", 0x380a2cf857b5ecc4),
    ("bare/every/High", 0x6a957cbb4fb7120d),
    ("bare/every/Ultra", 0x34b814aa97d2b7c8),
    ("bare/every/Custom", 0x34b814aa97d2b7c8),
    ("bare/off/auto-Unknown", 0x4623db63a8df603d),
    ("bare/off/auto-Integrated", 0x56961da738fded2b),
    ("bare/off/auto-EntryDiscrete", 0x4623db63a8df603d),
    ("bare/off/auto-MidDiscrete", 0x9d29d18748d7f6ac),
    ("bare/off/auto-HighDiscrete", 0x3553e5717f1f8a41),
    ("bare/off/Low", 0x56961da738fded2b),
    ("bare/off/Medium", 0x4623db63a8df603d),
    ("bare/off/High", 0x9d29d18748d7f6ac),
    ("bare/off/Ultra", 0x3553e5717f1f8a41),
    ("bare/off/Custom", 0x3553e5717f1f8a41),
    ("defaults/none/auto-Unknown", 0xfce9bea1f2fca2bd),
    ("defaults/none/auto-Integrated", 0x4e1a38227921a157),
    ("defaults/none/auto-EntryDiscrete", 0xfce9bea1f2fca2bd),
    ("defaults/none/auto-MidDiscrete", 0xb9da0d9b6e1d5d6e),
    ("defaults/none/auto-HighDiscrete", 0xae4d178d9b1f9430),
    ("defaults/none/Low", 0x4e1a38227921a157),
    ("defaults/none/Medium", 0xfce9bea1f2fca2bd),
    ("defaults/none/High", 0xb9da0d9b6e1d5d6e),
    ("defaults/none/Ultra", 0xae4d178d9b1f9430),
    ("defaults/none/Custom", 0xae4d178d9b1f9430),
    ("defaults/every/auto-Unknown", 0x9f744af69b060b9b),
    ("defaults/every/auto-Integrated", 0x65cc0eccdc4a1a4b),
    ("defaults/every/auto-EntryDiscrete", 0x9f744af69b060b9b),
    ("defaults/every/auto-MidDiscrete", 0xd24cb9785decccd4),
    ("defaults/every/auto-HighDiscrete", 0x36dfbf1784cd15ef),
    ("defaults/every/Low", 0x65cc0eccdc4a1a4b),
    ("defaults/every/Medium", 0x9f744af69b060b9b),
    ("defaults/every/High", 0xd24cb9785decccd4),
    ("defaults/every/Ultra", 0x36dfbf1784cd15ef),
    ("defaults/every/Custom", 0x36dfbf1784cd15ef),
    ("defaults/off/auto-Unknown", 0x4d0c63c0ae4f58d7),
    ("defaults/off/auto-Integrated", 0x732c13ee51dd99cc),
    ("defaults/off/auto-EntryDiscrete", 0x4d0c63c0ae4f58d7),
    ("defaults/off/auto-MidDiscrete", 0x252eac3328c77213),
    ("defaults/off/auto-HighDiscrete", 0xf75656d48e739288),
    ("defaults/off/Low", 0x732c13ee51dd99cc),
    ("defaults/off/Medium", 0x4d0c63c0ae4f58d7),
    ("defaults/off/High", 0x252eac3328c77213),
    ("defaults/off/Ultra", 0xf75656d48e739288),
    ("defaults/off/Custom", 0xf75656d48e739288),
    ("above/none/auto-Unknown", 0xb9f0fc9edec998b1),
    ("above/none/auto-Integrated", 0x38750b52c3b9fb40),
    ("above/none/auto-EntryDiscrete", 0xb9f0fc9edec998b1),
    ("above/none/auto-MidDiscrete", 0x6a2973fbc9558a70),
    ("above/none/auto-HighDiscrete", 0xfe0892bb2013767e),
    ("above/none/Low", 0x38750b52c3b9fb40),
    ("above/none/Medium", 0xb9f0fc9edec998b1),
    ("above/none/High", 0x6a2973fbc9558a70),
    ("above/none/Ultra", 0xfe0892bb2013767e),
    ("above/none/Custom", 0x6a0fdeadefe52bca),
    ("above/every/auto-Unknown", 0x55a4d63d22bdf210),
    ("above/every/auto-Integrated", 0xd9387bd6fcaa8d06),
    ("above/every/auto-EntryDiscrete", 0x55a4d63d22bdf210),
    ("above/every/auto-MidDiscrete", 0xc1c59e98bdc5c3ca),
    ("above/every/auto-HighDiscrete", 0x5dd0318080a1b473),
    ("above/every/Low", 0xd9387bd6fcaa8d06),
    ("above/every/Medium", 0x55a4d63d22bdf210),
    ("above/every/High", 0xc1c59e98bdc5c3ca),
    ("above/every/Ultra", 0x5dd0318080a1b473),
    ("above/every/Custom", 0x29fc698630720765),
    ("above/off/auto-Unknown", 0x7252356afc406259),
    ("above/off/auto-Integrated", 0x477da4df117e02e0),
    ("above/off/auto-EntryDiscrete", 0x7252356afc406259),
    ("above/off/auto-MidDiscrete", 0xcebede7030bb288d),
    ("above/off/auto-HighDiscrete", 0x59439dd60969f039),
    ("above/off/Low", 0x477da4df117e02e0),
    ("above/off/Medium", 0x7252356afc406259),
    ("above/off/High", 0xcebede7030bb288d),
    ("above/off/Ultra", 0x59439dd60969f039),
    ("above/off/Custom", 0xd481fc3f62a217dd),
    ("below/none/auto-Unknown", 0x8769d42a7605f94f),
    ("below/none/auto-Integrated", 0x8769d42a7605f94f),
    ("below/none/auto-EntryDiscrete", 0x8769d42a7605f94f),
    ("below/none/auto-MidDiscrete", 0x8769d42a7605f94f),
    ("below/none/auto-HighDiscrete", 0x8769d42a7605f94f),
    ("below/none/Low", 0x8769d42a7605f94f),
    ("below/none/Medium", 0x8769d42a7605f94f),
    ("below/none/High", 0x8769d42a7605f94f),
    ("below/none/Ultra", 0x8769d42a7605f94f),
    ("below/none/Custom", 0x8769d42a7605f94f),
    ("below/every/auto-Unknown", 0xaa88e72f2458a7f9),
    ("below/every/auto-Integrated", 0xaa88e72f2458a7f9),
    ("below/every/auto-EntryDiscrete", 0xaa88e72f2458a7f9),
    ("below/every/auto-MidDiscrete", 0xaa88e72f2458a7f9),
    ("below/every/auto-HighDiscrete", 0xaa88e72f2458a7f9),
    ("below/every/Low", 0xaa88e72f2458a7f9),
    ("below/every/Medium", 0xaa88e72f2458a7f9),
    ("below/every/High", 0xaa88e72f2458a7f9),
    ("below/every/Ultra", 0xaa88e72f2458a7f9),
    ("below/every/Custom", 0xaa88e72f2458a7f9),
    ("below/off/auto-Unknown", 0xc79919a3fae0fac4),
    ("below/off/auto-Integrated", 0xc79919a3fae0fac4),
    ("below/off/auto-EntryDiscrete", 0xc79919a3fae0fac4),
    ("below/off/auto-MidDiscrete", 0xc79919a3fae0fac4),
    ("below/off/auto-HighDiscrete", 0xc79919a3fae0fac4),
    ("below/off/Low", 0xc79919a3fae0fac4),
    ("below/off/Medium", 0xc79919a3fae0fac4),
    ("below/off/High", 0xc79919a3fae0fac4),
    ("below/off/Ultra", 0xc79919a3fae0fac4),
    ("below/off/Custom", 0xc79919a3fae0fac4),
    ("odd/none/auto-Unknown", 0x693ac13d2ce03156),
    ("odd/none/auto-Integrated", 0xc08c90844f677852),
    ("odd/none/auto-EntryDiscrete", 0x693ac13d2ce03156),
    ("odd/none/auto-MidDiscrete", 0x467cebb240c89633),
    ("odd/none/auto-HighDiscrete", 0x0f70bf5ef9b2dfa7),
    ("odd/none/Low", 0xc08c90844f677852),
    ("odd/none/Medium", 0x693ac13d2ce03156),
    ("odd/none/High", 0x467cebb240c89633),
    ("odd/none/Ultra", 0x0f70bf5ef9b2dfa7),
    ("odd/none/Custom", 0xa611d25fe042da43),
    ("odd/every/auto-Unknown", 0x48de43cfd02c9668),
    ("odd/every/auto-Integrated", 0x9fa280e3f299913c),
    ("odd/every/auto-EntryDiscrete", 0x48de43cfd02c9668),
    ("odd/every/auto-MidDiscrete", 0xd3354817ea692065),
    ("odd/every/auto-HighDiscrete", 0x2fb5c608c203eb31),
    ("odd/every/Low", 0x9fa280e3f299913c),
    ("odd/every/Medium", 0x48de43cfd02c9668),
    ("odd/every/High", 0xd3354817ea692065),
    ("odd/every/Ultra", 0x2fb5c608c203eb31),
    ("odd/every/Custom", 0xfe1c2c54b64a89d7),
    ("odd/off/auto-Unknown", 0xb0b7188825607fbb),
    ("odd/off/auto-Integrated", 0x62247d4700c2d2a4),
    ("odd/off/auto-EntryDiscrete", 0xb0b7188825607fbb),
    ("odd/off/auto-MidDiscrete", 0x369ff872f73c2da9),
    ("odd/off/auto-HighDiscrete", 0x0b31176f099a3416),
    ("odd/off/Low", 0x62247d4700c2d2a4),
    ("odd/off/Medium", 0xb0b7188825607fbb),
    ("odd/off/High", 0x369ff872f73c2da9),
    ("odd/off/Ultra", 0x0b31176f099a3416),
    ("odd/off/Custom", 0x014d30bac601776e),
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
            let mut state = SettingsState::new();
            state.authored = bare.clone();
            let graphics = state.resolve(&user, &ceiling).graphics;
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
    use concinnity_core::gfx::auto_exposure::EV_LIMIT;
    for (world_name, world) in worlds() {
        for (user_name, user) in users() {
            let mut state = SettingsState::new();
            state.authored = world.clone();
            let g = state
                .resolve(&user, &concinnity_core::settings::QualityCeiling::NONE)
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
