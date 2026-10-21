//! The master graphics-quality preset and the performance ceiling it resolves
//! to over a world's authored look.
//!
//! A preset produces a [`QualityCeiling`] that clamps the performance-relevant
//! settings DOWN where the chosen tier (or the detected GPU, under `Auto`)
//! cannot honor them. A ceiling never turns a feature on, it only reduces, so a
//! world authored conservatively is never upgraded, and an explicit per-row
//! user override always wins over it. The component defaults author the
//! top-tier look, so under `Auto` the ceiling is what picks a world's quality:
//! every tier below Ultra is reached by clamping, not by opting in.

use alloc::format;
use alloc::string::{String, ToString};

use crate::components::{AaMode, PassResolution, ShadowUpdate, UpscaleQuality};
use crate::render::backend::{GpuProfile, GpuTier};
use crate::render::environment::water::ocean::OceanQuality;
use crate::render::planar_reflection;
use crate::settings::levels::{AA_MODE, RENDER_SCALE};

/// The master graphics-quality choice. `Auto` resolves from the detected GPU
/// tier each launch; a named tier (Low..Ultra) is a fixed ceiling; `Custom`
/// imposes no ceiling (the user's per-row overrides drive).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QualityPreset {
    /// Resolve the ceiling from the GPU tier detected at launch.
    Auto,
    /// Fixed low-tier ceiling.
    Low,
    /// Fixed mid-tier ceiling.
    Medium,
    /// Fixed high-tier ceiling.
    High,
    /// Fixed top-tier ceiling; the only one that permits ray-traced reflections.
    Ultra,
    /// No ceiling: the world's authored look and the per-row overrides stand.
    Custom,
}

impl QualityPreset {
    /// The presets in menu-cycle order, matching
    /// [`levels::GRAPHICS_QUALITY`](crate::settings::levels::GRAPHICS_QUALITY).
    pub const ALL: [Self; 6] = [
        Self::Auto,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Ultra,
        Self::Custom,
    ];

    /// The display name, without an `Auto` tier suffix (see [`preset_label`]).
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Ultra => "Ultra",
            Self::Custom => "Custom",
        }
    }
}

/// A ceiling on the performance-relevant settings: which features are permitted
/// (`false` forces one off), and the least aggressive value each knob may take.
/// Each cap's no-ceiling value is the engine maximum, so an authored value
/// always stands under [`QualityCeiling::NONE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one ceiling per independent quality feature"
)]
pub struct QualityCeiling {
    /// The most expensive anti-aliasing mode permitted, by cost rank (Off <
    /// FXAA < TAA). A lower tier can force TAA to FXAA but never enables a mode
    /// the world did not author.
    pub aa_mode: AaMode,
    /// Whether screen-space ambient occlusion is permitted.
    pub ssao: bool,
    /// Whether screen-space reflections are permitted.
    pub ssr: bool,
    /// Whether ray-traced reflections are permitted.
    pub ray_traced_reflections: bool,
    /// Whether ray-traced reflections may cast sun-shadow rays from their hits.
    pub rt_reflection_shadows: bool,
    /// Whether the screen-space indirect bounce is permitted.
    pub ssgi: bool,
    /// Whether auto-exposure is permitted.
    pub auto_exposure: bool,
    /// The least aggressive upscaling permitted: the effective render scale is
    /// the more aggressive of the world's choice and this. `Quality` forces
    /// nothing.
    pub min_upscale: UpscaleQuality,
    /// The finest SSGI trace resolution permitted.
    pub ssgi_resolution: PassResolution,
    /// The most SSGI rays per pixel per frame permitted.
    pub ssgi_rays: u32,
    /// The finest roughness-aware reflection blur permitted.
    pub reflection_blur_resolution: PassResolution,
    /// The finest ray-traced reflection trace permitted.
    pub rt_reflection_resolution: PassResolution,
    /// The largest shadow map permitted, in texels.
    pub shadow_map_size: u32,
    /// Whether the `EveryFrame` shadow cadence is permitted; when not, it is
    /// clamped to `Hybrid`.
    pub allow_every_frame_shadows: bool,
    /// The highest scene-sampler anisotropy permitted.
    pub anisotropy: u32,
    /// The farthest shadow distance permitted, in world units.
    pub shadow_distance: u32,
    /// The most shadow cascades permitted.
    pub shadow_cascades: u32,
    /// The planar reflection planes rendered each frame. No world authors this:
    /// the budget is the cap itself, and reflectors past it fall back to the
    /// box-projected probe cube.
    pub planar_reflection_planes: u32,
    /// The resolution each planar mirror renders at, relative to the render
    /// resolution. No world authors this either.
    pub planar_reflection_resolution: PassResolution,
    /// How finely large water's simulated ocean runs. No world authors this:
    /// a lower tier runs fewer or coarser cascades, leaving more of the sea to
    /// the ripples.
    pub ocean: OceanQuality,
    /// The smallest a planet's finest terrain cells may be, in meters: the
    /// effective size is the coarser of the world's and this.
    pub min_planet_cell_size: u32,
}

// The engine maxima each cap takes where a tier imposes none.
const SSGI_RAYS_MAX: u32 = crate::render::post::ssgi::settings::MAX_RAYS;
const ANISO_MAX: u32 = 16;
#[expect(
    clippy::cast_possible_truncation,
    reason = "a single-digit cascade count"
)]
const SHADOW_CASCADES_MAX: u32 = crate::gfx::render_types::NUM_SHADOW_CASCADES as u32;
#[expect(
    clippy::cast_possible_truncation,
    reason = "a single-digit plane count"
)]
const PLANAR_PLANES_MAX: u32 = planar_reflection::MAX_PLANAR_PLANES as u32;

impl QualityCeiling {
    /// No ceiling: everything permitted, no forced upscaling. What `Custom`
    /// resolves to.
    pub const NONE: Self = Self {
        aa_mode: AaMode::Taa,
        ssao: true,
        ssr: true,
        ray_traced_reflections: true,
        rt_reflection_shadows: true,
        ssgi: true,
        auto_exposure: true,
        min_upscale: UpscaleQuality::Quality,
        ssgi_resolution: PassResolution::Full,
        ssgi_rays: SSGI_RAYS_MAX,
        reflection_blur_resolution: PassResolution::Full,
        rt_reflection_resolution: PassResolution::Full,
        shadow_map_size: u32::MAX,
        allow_every_frame_shadows: true,
        anisotropy: ANISO_MAX,
        shadow_distance: u32::MAX,
        shadow_cascades: SHADOW_CASCADES_MAX,
        planar_reflection_planes: PLANAR_PLANES_MAX,
        planar_reflection_resolution: PassResolution::Full,
        ocean: OceanQuality::FULL,
        min_planet_cell_size: 0,
    };

    /// The integrated / weakest tier.
    pub const LOW: Self = Self {
        aa_mode: AaMode::Fxaa,
        ssao: false,
        ssr: false,
        ray_traced_reflections: false,
        rt_reflection_shadows: false,
        ssgi: false,
        auto_exposure: true,
        min_upscale: UpscaleQuality::Performance,
        ssgi_resolution: PassResolution::Quarter,
        ssgi_rays: 1,
        reflection_blur_resolution: PassResolution::Quarter,
        rt_reflection_resolution: PassResolution::Quarter,
        shadow_map_size: 1024,
        allow_every_frame_shadows: false,
        anisotropy: 4,
        shadow_distance: 40,
        shadow_cascades: 2,
        // The two most impactful mirror planes (a floor plus one wall).
        planar_reflection_planes: 2,
        planar_reflection_resolution: PassResolution::Half,
        ocean: OceanQuality {
            size: 128,
            cascades: 2,
        },
        min_planet_cell_size: 4,
    };

    /// The entry discrete tier.
    pub const MEDIUM: Self = Self {
        aa_mode: AaMode::Taa,
        ssao: true,
        ssr: false,
        ray_traced_reflections: false,
        rt_reflection_shadows: false,
        ssgi: false,
        auto_exposure: true,
        min_upscale: UpscaleQuality::Balanced,
        ssgi_resolution: PassResolution::Half,
        ssgi_rays: 1,
        reflection_blur_resolution: PassResolution::Half,
        rt_reflection_resolution: PassResolution::Half,
        shadow_map_size: 2048,
        allow_every_frame_shadows: false,
        anisotropy: 8,
        shadow_distance: 80,
        shadow_cascades: 3,
        planar_reflection_planes: 3,
        planar_reflection_resolution: PassResolution::Half,
        ocean: OceanQuality {
            size: 128,
            cascades: 3,
        },
        min_planet_cell_size: 2,
    };

    /// The mainstream discrete tier.
    pub const HIGH: Self = Self {
        aa_mode: AaMode::Taa,
        ssao: true,
        ssr: true,
        ray_traced_reflections: false,
        rt_reflection_shadows: true,
        ssgi: true,
        auto_exposure: true,
        min_upscale: UpscaleQuality::Quality,
        ssgi_resolution: PassResolution::Half,
        ssgi_rays: 1,
        reflection_blur_resolution: PassResolution::Half,
        rt_reflection_resolution: PassResolution::Half,
        shadow_map_size: 4096,
        allow_every_frame_shadows: false,
        anisotropy: 16,
        shadow_distance: 160,
        shadow_cascades: 4,
        planar_reflection_planes: PLANAR_PLANES_MAX,
        planar_reflection_resolution: PassResolution::Full,
        ocean: OceanQuality::FULL,
        min_planet_cell_size: 1,
    };

    /// The enthusiast tier: everything but a second SSGI ray past two.
    pub const ULTRA: Self = Self {
        aa_mode: AaMode::Taa,
        ssao: true,
        ssr: true,
        ray_traced_reflections: true,
        rt_reflection_shadows: true,
        ssgi: true,
        auto_exposure: true,
        min_upscale: UpscaleQuality::Quality,
        ssgi_resolution: PassResolution::Full,
        ssgi_rays: 2,
        reflection_blur_resolution: PassResolution::Full,
        rt_reflection_resolution: PassResolution::Full,
        shadow_map_size: u32::MAX,
        allow_every_frame_shadows: true,
        anisotropy: ANISO_MAX,
        shadow_distance: u32::MAX,
        shadow_cascades: SHADOW_CASCADES_MAX,
        planar_reflection_planes: PLANAR_PLANES_MAX,
        planar_reflection_resolution: PassResolution::Full,
        ocean: OceanQuality::FULL,
        min_planet_cell_size: 1,
    };
}

/// The authored anti-aliasing mode under a cap: the cheaper of the two by cost
/// rank (Off < FXAA < TAA). Never raises the authored mode.
pub fn clamp_aa_mode(authored: AaMode, cap: AaMode) -> AaMode {
    if AA_MODE.index_of(authored) <= AA_MODE.index_of(cap) {
        authored
    } else {
        cap
    }
}

/// The authored shadow cadence under the ceiling: `Hybrid` where the tier
/// forbids `EveryFrame`, otherwise the authored cadence. Never raises.
pub fn clamp_shadow_update(authored: ShadowUpdate, ceiling: &QualityCeiling) -> ShadowUpdate {
    if ceiling.allow_every_frame_shadows {
        authored
    } else {
        ShadowUpdate::Hybrid
    }
}

/// The more aggressive (lower internal resolution) of two upscale qualities.
pub fn more_aggressive_upscale(a: UpscaleQuality, b: UpscaleQuality) -> UpscaleQuality {
    if RENDER_SCALE.index_of(a) >= RENDER_SCALE.index_of(b) {
        a
    } else {
        b
    }
}

/// The active ceiling for a preset on this GPU. `Auto` maps the GPU tier to a
/// named tier; `Custom` imposes none; a named tier is fixed.
pub fn resolve_ceiling(preset: QualityPreset, profile: &GpuProfile) -> QualityCeiling {
    match preset {
        QualityPreset::Custom => QualityCeiling::NONE,
        QualityPreset::Low => QualityCeiling::LOW,
        QualityPreset::Medium => QualityCeiling::MEDIUM,
        QualityPreset::High => QualityCeiling::HIGH,
        QualityPreset::Ultra => QualityCeiling::ULTRA,
        QualityPreset::Auto => auto_tier(profile).1,
    }
}

/// The named tier `Auto` resolves to on this GPU, as (label, ceiling). One
/// table, so the menu label can never disagree with the ceiling applied.
fn auto_tier(profile: &GpuProfile) -> (&'static str, QualityCeiling) {
    match profile.tier {
        // A discrete GPU whose driver would not report its VRAM: the mid tier
        // keeps the cheap wins and drops what an unknown GPU might not carry,
        // rather than handing it the whole top-tier look the defaults author.
        GpuTier::Integrated => ("Low", QualityCeiling::LOW),
        GpuTier::Unknown | GpuTier::EntryDiscrete => ("Medium", QualityCeiling::MEDIUM),
        GpuTier::MidDiscrete => ("High", QualityCeiling::HIGH),
        GpuTier::HighDiscrete => ("Ultra", QualityCeiling::ULTRA),
    }
}

/// The master row's text for a preset: a named tier shows its name, while
/// `Auto` names the tier it resolved to on this GPU (e.g. "Auto (High)").
pub fn preset_label(preset: QualityPreset, profile: &GpuProfile) -> String {
    match preset {
        QualityPreset::Auto => format!("Auto ({})", auto_tier(profile).0),
        other => other.name().to_string(),
    }
}

#[cfg(test)]
mod tests;
