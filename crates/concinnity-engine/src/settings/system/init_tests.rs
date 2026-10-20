// The settings state graphics init resolves: the stored choices, the detected
// GPU's ceiling and the launch's overrides layered over the world's authored
// config, built the way init feeds `SettingsState::from_init`.

use crate::components::{
    AaMode, GraphicsConfig, IndirectTechnique, PassResolution, StreamingConfig, UpscaleQuality,
    UpscalerBackend, WindowMode,
};
use crate::ecs::Resources;
use crate::launch::{LaunchArgs, LaunchRequest};
use crate::render::backend::{DeviceCapabilities, GpuProfile, GpuTier, GpuVendor};
use crate::settings::quality::resolve_ceiling;
use crate::settings::{
    QualityPreset, ResolvedGraphics, StoredSettings, WorldAssets, WorldSettings,
};
use crate::window::display_mode::DisplayMode;
use alloc::vec::Vec;

use super::{SettingsInit, SettingsState};

// What graphics init feeds the settings state.
pub(super) struct Launch {
    pub(super) stored: StoredSettings,
    pub(super) profile: GpuProfile,
    pub(super) request: LaunchRequest,
    pub(super) graphics: GraphicsConfig,
    pub(super) streaming: Option<StreamingConfig>,
}

impl Launch {
    // A first launch on an unclassified GPU of a world authoring `graphics`.
    pub(super) fn new(graphics: GraphicsConfig) -> Self {
        Self {
            stored: StoredSettings::default(),
            profile: GpuProfile::UNKNOWN,
            request: LaunchRequest::default(),
            graphics,
            streaming: None,
        }
    }

    // The launch with the stored choices `stored`.
    pub(super) fn stored(self, stored: StoredSettings) -> Self {
        Self { stored, ..self }
    }

    // The launch on a GPU profiled as `profile`.
    pub(super) fn on(self, profile: GpuProfile) -> Self {
        Self { profile, ..self }
    }

    // The state init builds: the preset the request and store resolve to, its
    // ceiling on this GPU, and the request's session overrides.
    pub(super) fn state(&self) -> SettingsState {
        let preset = self
            .request
            .resolve_quality_preset(&self.stored, &mut Resources::default());
        let ceiling = resolve_ceiling(preset, &self.profile);
        SettingsState::from_init(SettingsInit {
            stored: &self.stored,
            gpu_profile: self.profile,
            capabilities: DeviceCapabilities::ALL,
            preset,
            ceiling: &ceiling,
            window: None,
            authored: WorldSettings::new(WorldAssets {
                graphics: Some(&self.graphics),
                lut_strength: None,
                streaming: self.streaming.as_ref(),
            }),
            session: self.request.session_overrides(),
        })
    }
}

// A `GraphicsConfig` setting every quality-governed section to the defaults
// adjusted by `f`.
pub(super) fn effects(f: impl FnOnce(&mut ResolvedGraphics)) -> GraphicsConfig {
    let mut e = ResolvedGraphics::default();
    f(&mut e);
    GraphicsConfig {
        ambient_occlusion: e.ambient_occlusion.into(),
        reflections: e.reflections.into(),
        indirect_lighting: e.indirect_lighting.into(),
        anti_aliasing: e.anti_aliasing.into(),
        exposure: e.exposure.into(),
        ..Default::default()
    }
}

// A discrete GPU at `tier`.
pub(super) fn profile_at(tier: GpuTier) -> GpuProfile {
    GpuProfile {
        vendor: GpuVendor::Amd,
        tier,
        memory_budget_bytes: 8 << 30,
        unified_memory: false,
        discrete: true,
    }
}

fn stored_preset(preset: QualityPreset) -> StoredSettings {
    let mut stored = StoredSettings::default();
    stored.graphics.quality_preset = Some(preset);
    stored
}

// Every stored post-process choice overrides the world's authored value. Under
// `Custom` there is no ceiling, so each override stands exactly as stored.
#[test]
fn persisted_post_process_overrides_win_over_authored_config() {
    let mut stored = stored_preset(QualityPreset::Custom);
    let g = &mut stored.graphics;
    g.exposure_ev = Some(2.0);
    g.bloom_intensity = Some(0.25);
    g.bloom_threshold = Some(1.5);
    g.bloom_knee = Some(0.2);
    g.vignette = Some(0.4);
    g.lut_strength = Some(0.3);
    g.ambient_intensity = Some(2.0);
    g.ssao = Some(true);
    g.ssr = Some(true);
    g.ray_traced_reflections = Some(true);
    g.ssgi = Some(true);
    g.auto_exposure = Some(true);
    g.aa_mode = Some(AaMode::Taa);
    g.ssgi_resolution = Some(PassResolution::Full);
    g.ssgi_rays = Some(2);
    g.reflection_blur_resolution = Some(PassResolution::Full);
    g.ssao_radius = Some(0.9);
    g.ssao_intensity = Some(2.0);
    g.ssr_intensity = Some(0.4);
    g.ssr_max_distance = Some(20.0);
    g.ssgi_intensity = Some(1.5);
    g.ssgi_max_distance = Some(4.0);
    g.auto_exposure_min_ev = Some(-4.0);
    g.auto_exposure_max_ev = Some(4.0);
    g.auto_exposure_speed = Some(3.0);
    // Deliberately the opposite of the overrides above, so a resolved value
    // can only have come from the store.
    let live = Launch::new(effects(|e| {
        e.anti_aliasing.mode = AaMode::Off;
        e.ambient_occlusion.enabled = false;
        e.exposure.ev = -3.0;
    }))
    .stored(stored)
    .state();

    let r = &live.resolved().graphics;
    assert!(r.taa(), "the stored TAA mode is what the backend builds");
    assert!(r.ssao().is_some(), "the stored SSAO toggle sizes SSAO");
    assert!(r.ambient_occlusion.enabled);
    assert!(r.reflections.screen_space);
    assert!(r.reflections.ray_traced);
    assert_eq!(r.indirect_lighting.technique, IndirectTechnique::Ssgi);
    assert!(r.exposure.auto);
    assert_eq!(r.anti_aliasing.mode, AaMode::Taa);
    assert_eq!(r.indirect_lighting.ssgi_resolution, PassResolution::Full);
    assert_eq!(r.indirect_lighting.ssgi_rays, 2);
    assert_eq!(r.reflections.blur_resolution, PassResolution::Full);
    assert_eq!(r.ambient_occlusion.radius, 0.9);
    assert_eq!(r.ambient_occlusion.intensity, 2.0);
    assert_eq!(r.reflections.intensity, 0.4);
    assert_eq!(r.reflections.max_distance, 20.0);
    assert_eq!(r.indirect_lighting.ssgi_intensity, 1.5);
    assert_eq!(r.indirect_lighting.ssgi_max_distance, 4.0);
    assert_eq!(r.exposure.auto_min_ev, -4.0);
    assert_eq!(r.exposure.auto_max_ev, 4.0);
    assert_eq!(r.exposure.auto_speed, 3.0);
    // Exposure is stored as EV and applied as 2^EV.
    assert_eq!(
        r.post_process().exposure,
        4.0,
        "2^2 EV, not the authored -3"
    );
    assert_eq!(r.post_process().bloom_intensity, 0.25);
    assert_eq!(r.post_process().bloom_threshold, 1.5);
    assert_eq!(r.post_process().bloom_knee, 0.2);
    assert_eq!(r.post_process().vignette, 0.4);
    assert_eq!(r.post_process().lut_strength, 0.3);
    assert_eq!(r.ambient_intensity(), 2.0);
    assert_eq!(r.post_process().fxaa, 1.0, "TAA keeps the FXAA cleanup");
    // The pristine authored config is the baseline a preset change re-clamps
    // from, untouched by the overrides.
    assert_eq!(live.authored().graphics.anti_aliasing.mode, AaMode::Off);
    assert!(!live.authored().graphics.ambient_occlusion.enabled);
}

// A weak-tier ceiling forces the authored effects off where the user expressed
// no preference, and keeps the authored config for a later up-shift.
#[test]
fn low_preset_forces_authored_effects_off_but_keeps_the_baseline() {
    let live = Launch::new(effects(|e| {
        e.ambient_occlusion.enabled = true;
        e.reflections.screen_space = true;
        e.reflections.ray_traced = true;
        e.indirect_lighting.technique = IndirectTechnique::Ssgi;
        e.exposure.auto = true;
        e.anti_aliasing.mode = AaMode::Taa;
    }))
    .stored(stored_preset(QualityPreset::Low))
    .on(profile_at(GpuTier::Integrated))
    .state();

    let r = &live.resolved().graphics;
    assert!(r.ssao().is_none(), "the ceiling trims SSAO");
    assert!(!r.taa(), "TAA clamps down to a cheaper AA mode");
    assert!(!r.ambient_occlusion.enabled);
    assert!(!r.reflections.screen_space);
    assert!(!r.reflections.ray_traced);
    assert_eq!(r.indirect_lighting.technique, IndirectTechnique::Ibl);
    assert_ne!(r.anti_aliasing.mode, AaMode::Taa);
    // Even the weakest ceiling permits auto-exposure.
    assert!(r.exposure.auto);
    let authored = &live.authored().graphics;
    assert!(authored.ambient_occlusion.enabled);
    assert!(authored.reflections.screen_space);
    assert_eq!(authored.anti_aliasing.mode, AaMode::Taa);
}

// An explicit per-row choice beats the preset ceiling; rows the user never
// touched still clamp.
#[test]
fn an_explicit_override_survives_the_preset_ceiling() {
    let mut stored = stored_preset(QualityPreset::Low);
    stored.graphics.ssao = Some(true);
    stored.graphics.aa_mode = Some(AaMode::Taa);
    let live = Launch::new(effects(|e| {
        e.ambient_occlusion.enabled = true;
        e.reflections.screen_space = true;
        e.anti_aliasing.mode = AaMode::Taa;
    }))
    .stored(stored)
    .on(profile_at(GpuTier::Integrated))
    .state();

    let r = &live.resolved().graphics;
    assert!(r.ambient_occlusion.enabled, "the explicit SSAO choice wins");
    assert_eq!(r.anti_aliasing.mode, AaMode::Taa, "explicit AA mode wins");
    assert!(
        !r.reflections.screen_space,
        "a row the user never touched still clamps under the ceiling"
    );
}

// A ceiling only ever reduces: a high tier does not turn back on what the
// world turned off, and the stored preset is the one the state runs.
#[test]
fn a_high_ceiling_never_enables_what_the_world_turned_off() {
    let live = Launch::new(effects(|e| {
        e.ambient_occlusion.enabled = false;
        e.reflections.screen_space = false;
        e.reflections.ray_traced = false;
    }))
    .stored(stored_preset(QualityPreset::Ultra))
    .on(profile_at(GpuTier::HighDiscrete))
    .state();

    assert_eq!(live.quality_preset(), QualityPreset::Ultra);
    let r = &live.resolved().graphics;
    assert!(
        !r.ambient_occlusion.enabled,
        "a feature turned off stays off"
    );
    assert!(!r.reflections.screen_space);
    assert!(!r.reflections.ray_traced);
}

// Under a first launch's `Auto` the ceiling follows the detected tier: a weaker
// GPU shadows a smaller map, and the top tier leaves the authored size alone.
#[test]
fn auto_preset_shadow_ceiling_tracks_the_detected_tier() {
    let sizes: Vec<u32> = [
        GpuTier::Integrated,
        GpuTier::EntryDiscrete,
        GpuTier::MidDiscrete,
        GpuTier::HighDiscrete,
    ]
    .into_iter()
    .map(|tier| {
        let live = Launch::new(GraphicsConfig::default())
            .on(profile_at(tier))
            .state();
        assert_eq!(live.quality_preset(), QualityPreset::Auto);
        live.resolved().graphics.shadow_params().map_size
    })
    .collect();

    assert!(
        sizes.windows(2).all(|w| w[0] <= w[1]),
        "a stronger tier never shadows at a smaller map: {sizes:?}"
    );
    assert_eq!(
        sizes.last().copied(),
        Some(4096),
        "the top tier leaves the world's authored 4096 unclamped"
    );
}

// An integrated GPU under the first-launch `Auto` preset takes the Low ceiling.
#[test]
fn auto_preset_resolves_ceiling_from_gpu_tier() {
    let live = Launch::new(GraphicsConfig::default())
        .on(GpuProfile {
            vendor: GpuVendor::Intel,
            tier: GpuTier::Integrated,
            memory_budget_bytes: 0,
            unified_memory: false,
            discrete: false,
        })
        .state();
    assert_eq!(live.quality_preset(), QualityPreset::Auto);
    assert_eq!(live.resolved().graphics.shadow_params().map_size, 1024);
}

// The Low ceiling clamps the authored shadow and filtering defaults, and the
// authored baselines are kept for a later preset up-shift.
#[test]
fn low_preset_ceiling_clamps_quality_knobs() {
    let live = Launch::new(GraphicsConfig::default())
        .stored(stored_preset(QualityPreset::Low))
        .state();
    let shadows = live.resolved().graphics.shadow_params();
    assert_eq!(shadows.map_size, 1024);
    assert_eq!(shadows.cadence.distance, 40);
    assert_eq!(shadows.cadence.cascades, 2);
    assert_eq!(live.resolved().graphics.texture_filtering.anisotropy, 4);
    let authored = &live.authored().graphics;
    assert_eq!(authored.shadows.map_size, 4096);
    assert_eq!(authored.shadows.distance, 80);
    assert_eq!(authored.shadows.cascades, 4);
    assert_eq!(authored.texture_filtering.anisotropy, 16);
}

// The stored choices beat the world's authored pacing and shadow values.
#[test]
fn persisted_settings_override_authored_config() {
    let mut stored = stored_preset(QualityPreset::Custom);
    stored.graphics.vsync = Some(true);
    stored.graphics.fps_cap = Some(30);
    stored.graphics.frames_in_flight = Some(3);
    stored.graphics.shadow_map_size = Some(1024);
    let live = Launch::new(GraphicsConfig::default())
        .stored(stored)
        .state();

    assert_eq!(live.quality_preset(), QualityPreset::Custom);
    let r = &live.resolved().graphics;
    assert_eq!(
        r.frame_pacing.fps_cap, 30,
        "the stored cap overrides the world's 0"
    );
    assert!(
        r.frame_pacing.vsync,
        "stored vsync overrides the authored false"
    );
    assert_eq!(r.frames_in_flight(), 3);
    assert_eq!(r.shadow_params().map_size, 1024, "explicit override wins");
}

// `--vsync` wins over the stored choice and the world's.
#[test]
fn launch_vsync_overrides_persisted_and_authored_config() {
    let mut stored = StoredSettings::default();
    stored.graphics.vsync = Some(false);
    let mut launch = Launch::new(GraphicsConfig::default()).stored(stored);
    launch.request = LaunchRequest {
        args: LaunchArgs {
            vsync: true,
            ..LaunchArgs::NONE
        },
        ..Default::default()
    };
    assert!(launch.state().resolved().graphics.frame_pacing.vsync);
}

// Display-output, upscaling and system/streaming choices resolve from the
// world's config overridden by the stored choice, and the stored window and
// display modes are held for the startup apply.
#[test]
fn persisted_display_and_system_overrides_resolve() {
    let mut stored = stored_preset(QualityPreset::Custom);
    let g = &mut stored.graphics;
    g.window_mode = Some(WindowMode::Fullscreen);
    g.resolution = Some([1920, 1080, 60]);
    g.render_scale = Some(UpscaleQuality::Performance);
    g.upscale_backend = Some(UpscalerBackend::Fsr3);
    g.temporal_upscaling = Some(true);
    g.hdr_display = Some(true);
    g.hdr_pq = Some(true);
    g.occlusion_two_pass = Some(true);
    g.texture_cap = Some(192);
    g.texture_budget = Some(8);
    g.perf_stats = Some(false);
    g.show_fps = Some(false);
    g.show_vram = Some(false);
    let mut launch = Launch::new(effects(|_| {})).stored(stored);
    launch.streaming = Some(StreamingConfig::default());
    let live = launch.state();

    let r = live.resolved();
    assert_eq!(
        r.graphics.anti_aliasing.upscale_quality,
        UpscaleQuality::Performance
    );
    assert_eq!(r.graphics.anti_aliasing.upscaler, UpscalerBackend::Fsr3);
    assert!(r.graphics.anti_aliasing.upscaling);
    assert!(r.graphics.display.hdr);
    assert!(r.graphics.display.pq);
    assert!(r.graphics.culling.two_pass_occlusion);
    assert_eq!(r.texture_cap, 192);
    assert_eq!(r.texture_budget, 8);
    assert!(!r.perf_stats);
    assert!(!r.show_fps);
    assert!(!r.show_vram);
    assert_eq!(live.window().mode, WindowMode::Fullscreen);
    assert_eq!(
        live.resolution(),
        Some(DisplayMode {
            width: 1920,
            height: 1080,
            refresh_hz: 60,
        })
    );
}
