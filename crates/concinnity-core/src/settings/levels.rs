//! The value each option index of a quality cycle row stands for, and the index
//! a value shows as. The orders match the option labels in this module's parent
//! (`AA_MODE_OPTIONS`, `RENDER_SCALE_OPTIONS`, `PASS_RESOLUTION_OPTIONS`,
//! `SSGI_RAYS_OPTIONS`); an authored value off a discrete level shows as the
//! nearest one.

use crate::components::{AaMode, PassResolution, UpscaleQuality};

/// SSGI rays-per-pixel levels, in `SSGI_RAYS_OPTIONS` order.
pub const SSGI_RAYS_COUNTS: [u32; 3] = [1, 2, 4];

/// The anti-aliasing mode at an option index. Order matches `AA_MODE_OPTIONS`
/// (off, FXAA, TAA), which is also ascending cost, so the index doubles as the
/// rank a quality ceiling clamps against.
pub fn aa_mode_at(index: usize) -> AaMode {
    match index {
        0 => AaMode::Off,
        2 => AaMode::Taa,
        _ => AaMode::Fxaa,
    }
}

/// The option index of an anti-aliasing mode.
pub fn aa_mode_index(mode: AaMode) -> usize {
    match mode {
        AaMode::Off => 0,
        AaMode::Fxaa => 1,
        AaMode::Taa => 2,
    }
}

/// The render-scale preset at an option index, least aggressive first.
pub fn render_scale_at(index: usize) -> UpscaleQuality {
    match index {
        1 => UpscaleQuality::Balanced,
        2 => UpscaleQuality::Performance,
        3 => UpscaleQuality::UltraPerformance,
        _ => UpscaleQuality::Quality,
    }
}

/// The option index of a render-scale preset: higher is more aggressive (a
/// lower internal resolution).
pub fn render_scale_index(quality: UpscaleQuality) -> usize {
    match quality {
        UpscaleQuality::Quality => 0,
        UpscaleQuality::Balanced => 1,
        UpscaleQuality::Performance => 2,
        UpscaleQuality::UltraPerformance => 3,
    }
}

/// The pass resolution at an option index, finest first.
pub fn pass_resolution_at(index: usize) -> PassResolution {
    PassResolution::ALL.get(index).copied().unwrap_or_default()
}

/// The option index of a pass resolution.
pub fn pass_resolution_index(res: PassResolution) -> usize {
    res as usize
}

/// The SSGI ray count at an option index.
pub fn ssgi_rays_at(index: usize) -> u32 {
    *SSGI_RAYS_COUNTS.get(index).unwrap_or(&SSGI_RAYS_COUNTS[0])
}

/// The option index nearest an SSGI ray count.
pub fn ssgi_rays_index(count: u32) -> usize {
    nearest_count_index(&SSGI_RAYS_COUNTS, count)
}

/// The index of the level closest to `count`; a tie picks the lower level.
pub fn nearest_count_index(levels: &[u32], count: u32) -> usize {
    levels
        .iter()
        .enumerate()
        .min_by_key(|&(_, &v)| v.abs_diff(count))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
