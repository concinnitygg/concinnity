//! Each cycle setting's options: the label a row shows at each option index and
//! the value that index stands for, held side by side so the two never drift.
//! An authored value off a discrete level shows as the nearest one.

use crate::components::{
    AaMode, PassResolution, ShadowUpdate, UpscaleQuality, UpscalerBackend, WindowMode,
};
use crate::render::output::dlss::DlssPreset;
use crate::settings::QualityPreset;

/// A cycle row's options in menu order: a label and a value for each index.
///
/// ```rust
/// # use concinnity_core::settings::levels::FPS_CAP;
/// assert_eq!(FPS_CAP.labels()[2], "60");
/// assert_eq!(FPS_CAP.at(2), 60);
/// assert_eq!(FPS_CAP.index_of(58), 2);
/// ```
pub struct Levels<T: 'static> {
    labels: &'static [&'static str],
    values: &'static [T],
    fallback: usize,
    find: fn(&[T], T) -> Option<usize>,
}

impl<T: Copy> Levels<T> {
    // Options labeled `labels` standing for `values`, index for index. An index
    // past the end reads as the value at `fallback`, and a value `find` places
    // nowhere shows as `fallback`.
    const fn new(
        labels: &'static [&'static str],
        values: &'static [T],
        fallback: usize,
        find: fn(&[T], T) -> Option<usize>,
    ) -> Self {
        assert!(labels.len() == values.len(), "a label for every value");
        assert!(
            fallback < values.len(),
            "the fallback is one of the options"
        );
        Self {
            labels,
            values,
            fallback,
            find,
        }
    }

    /// The labels, in option order.
    pub const fn labels(&self) -> &'static [&'static str] {
        self.labels
    }

    #[cfg(test)]
    pub(crate) const fn values(&self) -> &'static [T] {
        self.values
    }

    /// The value at option `index`.
    pub fn at(&self, index: usize) -> T {
        self.values
            .get(index)
            .copied()
            .unwrap_or(self.values[self.fallback])
    }

    /// The option index `value` shows as.
    pub fn index_of(&self, value: T) -> usize {
        (self.find)(self.values, value).unwrap_or(self.fallback)
    }
}

// The index holding exactly `value`.
fn exact<T: PartialEq + Copy>(values: &[T], value: T) -> Option<usize> {
    values.iter().position(|v| *v == value)
}

// The index of the level closest to `count`; a tie picks the lower level.
fn nearest(values: &[u32], count: u32) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .min_by_key(|&(_, &v)| v.abs_diff(count))
        .map(|(i, _)| i)
}

// The level whose pool cap is closest to `level`'s; the budget is not compared.
fn nearest_cap(values: &[(u32, u32)], level: (u32, u32)) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .min_by_key(|&(_, &(cap, _))| cap.abs_diff(level.0))
        .map(|(i, _)| i)
}

fn near_gain(values: &[f32], gain: f32) -> Option<usize> {
    values.iter().position(|g| (g - gain).abs() < 1.0e-4)
}

/// The master graphics-quality presets. The `Auto` row is relabeled with its
/// resolved tier at runtime, which this static table cannot express.
pub const GRAPHICS_QUALITY: Levels<QualityPreset> = Levels::new(
    &["Auto", "Low", "Medium", "High", "Ultra", "Custom"],
    &QualityPreset::ALL,
    0,
    exact,
);

/// Anti-aliasing modes, in ascending cost, so the index doubles as the rank a
/// quality ceiling clamps against.
pub const AA_MODE: Levels<AaMode> = Levels::new(
    &["Off", "FXAA", "TAA"],
    &[AaMode::Off, AaMode::Fxaa, AaMode::Taa],
    1,
    exact,
);

/// Render-scale (upscaling quality) presets: a higher index is a lower internal
/// resolution.
pub const RENDER_SCALE: Levels<UpscaleQuality> = Levels::new(
    &["Quality", "Balanced", "Performance", "Ultra"],
    &[
        UpscaleQuality::Quality,
        UpscaleQuality::Balanced,
        UpscaleQuality::Performance,
        UpscaleQuality::UltraPerformance,
    ],
    0,
    exact,
);

/// Pass resolutions (SSGI trace, ray-traced reflection trace, reflection blur),
/// finest first.
pub const PASS_RESOLUTION: Levels<PassResolution> =
    Levels::new(&["Full", "Half", "Quarter"], PassResolution::ALL, 1, exact);

/// SSGI rays per pixel.
pub const SSGI_RAYS: Levels<u32> = Levels::new(&["1", "2", "4"], &[1, 2, 4], 0, nearest);

/// Window modes.
pub const WINDOW_MODE: Levels<WindowMode> = Levels::new(
    &["Windowed", "Borderless", "Fullscreen"],
    &[
        WindowMode::Windowed,
        WindowMode::Borderless,
        WindowMode::Fullscreen,
    ],
    0,
    exact,
);

/// Upscaler backends, in `UpscalerBackend` order. DirectX / Vulkan only (Metal
/// uses `MetalFX`).
pub const UPSCALE_BACKEND: Levels<UpscalerBackend> = Levels::new(
    &["Auto", "FSR 3", "DLSS", "XeSS"],
    &[
        UpscalerBackend::Auto,
        UpscalerBackend::Fsr3,
        UpscalerBackend::Dlss,
        UpscalerBackend::Xess,
    ],
    0,
    exact,
);

/// DLSS presets, in `DlssPreset::ALL` order.
pub const DLSS_PRESET: Levels<DlssPreset> =
    Levels::new(&DlssPreset::LABELS, &DlssPreset::ALL, 0, exact);

/// Frame-rate caps; "Unlimited" is no cap.
pub const FPS_CAP: Levels<u32> = Levels::new(
    &["Unlimited", "30", "60", "120", "144", "240", "300", "360"],
    &[0, 30, 60, 120, 144, 240, 300, 360],
    0,
    nearest,
);

/// Shadow-map cascade resolutions in texels; "Off" disables shadows.
pub const SHADOW_RESOLUTION: Levels<u32> = Levels::new(
    &["Off", "1024", "2048", "4096"],
    &[0, 1024, 2048, 4096],
    2,
    nearest,
);

/// Shadow re-render cadences, best (most expensive) first.
pub const SHADOW_UPDATE: Levels<ShadowUpdate> = Levels::new(
    &["Every Frame", "Hybrid"],
    &[ShadowUpdate::EveryFrame, ShadowUpdate::Hybrid],
    1,
    exact,
);

/// Shadow distances in world units the cascades cover.
pub const SHADOW_DISTANCE: Levels<u32> = Levels::new(
    &["40 m", "80 m", "160 m", "320 m"],
    &[40, 80, 160, 320],
    1,
    nearest,
);

/// Shadow cascade counts.
pub const SHADOW_CASCADES: Levels<u32> = Levels::new(&["2", "3", "4"], &[2, 3, 4], 2, nearest);

/// Anisotropic-filtering degrees for the scene sampler; "Off" is 1x.
pub const ANISOTROPY: Levels<u32> = Levels::new(
    &["Off", "2x", "4x", "8x", "16x"],
    &[1, 2, 4, 8, 16],
    3,
    nearest,
);

/// Frames in flight: lower is less latency, higher is smoother pacing.
pub const FRAMES_IN_FLIGHT: Levels<u32> = Levels::new(&["1", "2", "3"], &[1, 2, 3], 1, nearest);

/// Texture quality: each level's (streaming pool cap, per-frame upload budget),
/// found by its pool cap.
pub const TEXTURE_QUALITY: Levels<(u32, u32)> = Levels::new(
    &["Low", "Medium", "High", "Ultra"],
    &[(48, 2), (96, 4), (192, 8), (384, 12)],
    3,
    nearest_cap,
);

/// Volume gains shared by the master and per-bus rows; a gain off the levels
/// shows as full.
pub const VOLUME: Levels<f32> = Levels::new(
    &["Off", "25%", "50%", "75%", "100%"],
    &[0.0, 0.25, 0.5, 0.75, 1.0],
    4,
    near_gain,
);

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trips<T: Copy + PartialEq + core::fmt::Debug>(levels: &Levels<T>) {
        for (i, &v) in levels.values().iter().enumerate() {
            assert_eq!(levels.index_of(v), i, "{v:?}");
            assert_eq!(levels.at(i), v);
        }
    }

    #[test]
    fn every_level_round_trips() {
        round_trips(&GRAPHICS_QUALITY);
        round_trips(&AA_MODE);
        round_trips(&RENDER_SCALE);
        round_trips(&PASS_RESOLUTION);
        round_trips(&SSGI_RAYS);
        round_trips(&WINDOW_MODE);
        round_trips(&UPSCALE_BACKEND);
        round_trips(&DLSS_PRESET);
        round_trips(&FPS_CAP);
        round_trips(&SHADOW_RESOLUTION);
        round_trips(&SHADOW_UPDATE);
        round_trips(&SHADOW_DISTANCE);
        round_trips(&SHADOW_CASCADES);
        round_trips(&ANISOTROPY);
        round_trips(&FRAMES_IN_FLIGHT);
        round_trips(&TEXTURE_QUALITY);
        round_trips(&VOLUME);
    }

    // An index past the end reads as the fallback value.
    #[test]
    fn an_index_past_the_end_reads_the_fallback() {
        assert_eq!(AA_MODE.at(9), AaMode::Fxaa);
        assert_eq!(FPS_CAP.at(99), 0);
        assert_eq!(SHADOW_RESOLUTION.at(9), 2048);
        assert_eq!(SHADOW_UPDATE.at(5), ShadowUpdate::Hybrid);
        assert_eq!(TEXTURE_QUALITY.at(7), (384, 12));
        assert_eq!(
            PASS_RESOLUTION.at(PassResolution::ALL.len()),
            PassResolution::default()
        );
        assert_eq!(DLSS_PRESET.at(9), DlssPreset::default());
        assert_eq!(GRAPHICS_QUALITY.at(99), QualityPreset::Auto);
    }

    // A count off the discrete levels shows as the nearest one; a tie picks the
    // lower level.
    #[test]
    fn a_count_off_the_levels_snaps_to_the_nearest() {
        assert_eq!(FPS_CAP.index_of(58), FPS_CAP.index_of(60));
        assert_eq!(FPS_CAP.index_of(1000), FPS_CAP.values().len() - 1);
        assert_eq!(SSGI_RAYS.index_of(3), 1);
        assert_eq!(SSGI_RAYS.index_of(9), 2);
        assert_eq!(TEXTURE_QUALITY.index_of((100, 0)), 1);
    }

    // A gain that is not a preset shows as full volume.
    #[test]
    fn a_gain_off_the_levels_shows_as_full() {
        assert_eq!(VOLUME.index_of(0.33), 4);
        assert_eq!(VOLUME.at(VOLUME.index_of(1.0)), 1.0);
    }

    // The engine defaults each have a row of their own.
    #[test]
    fn the_defaults_have_rows() {
        let cap = crate::components::FramePacing::default().fps_cap;
        assert_eq!(FPS_CAP.at(FPS_CAP.index_of(cap)), cap);
    }
}
