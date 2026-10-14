// Indirect-lighting schema.

use crate::components::{PassResolution, Vocabulary};
use crate::render::post::ssgi;

/// Indirect-diffuse lighting technique for `GraphicsConfig.indirect_lighting.technique`. `Ibl`
/// is the image-based-lighting-only ambient term; `Ssgi` layers a screen-space
/// global-illumination bounce on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default, Vocabulary)]
pub enum IndirectTechnique {
    /// Image-based lighting only.
    #[default]
    #[vocab("ibl")]
    Ibl,
    /// Image-based lighting plus a screen-space bounce.
    #[vocab("ssgi")]
    Ssgi,
}

/// Default SSGI hemisphere rays per pixel per frame for the authored
/// `ssgi_rays` field. Defined here (the schema default) and read by
/// `render::post::ssgi::settings` for its runtime clamp path, so the authored
/// default and the runtime code stay a single source of truth.
pub const DEFAULT_SSGI_RAYS: u32 = 1;

/// Indirect (ambient) lighting: how strong the environment's ambient term is,
/// and whether a screen-space bounce lets nearby lit surfaces bleed color onto
/// one another. The `indirect_lighting` section of
/// [GraphicsConfig](#graphicsconfig); a field left unset takes the default
/// below.
///
/// The screen-space bounce is on by default, which is the look on capable
/// hardware: the `Auto` graphics quality preset clamps it back to `ibl` below
/// the high tier.
///
/// ```rust
/// # use concinnity_core::components::{GraphicsConfig, IndirectLightingOverrides};
/// GraphicsConfig {
///     indirect_lighting: IndirectLightingOverrides {
///         ambient_intensity: Some(1.5),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[asset(section)]
#[serde(default)]
pub struct IndirectLighting {
    /// Indirect-diffuse lighting source. `ibl` uses the environment map's
    /// ambient alone. `ssgi` (default) adds a screen-space global-illumination
    /// pass on top, so nearby lit surfaces bleed color onto one another; the
    /// environment ambient still covers the off-screen / sky fallback.
    #[asset(default = IndirectTechnique::Ssgi, apply = rebuild)]
    pub technique: IndirectTechnique,
    /// Multiplier on the indirect (ambient / IBL) lighting term. 1.0 (default)
    /// leaves the environment-derived ambient at its physical level. Raising it
    /// lifts fill light in areas the directional light cannot reach (shadowed
    /// facades, alleys) without brightening directly lit surfaces, which the
    /// sun already dominates. Scales the diffuse and specular IBL together, so
    /// reflections stay consistent with the brighter ambient. Useful for
    /// high-contrast exterior scenes where a strong sun would otherwise crush
    /// shadows to black.
    #[asset(default = 1.0, min = 0.0, max = 16.0, apply = live)]
    pub ambient_intensity: f32,
    /// Indirect-bounce strength. Scales the gathered indirect light added on
    /// top of the existing shading; 0 makes it a no-op. Only matters when
    /// `technique` is `ssgi`.
    #[asset(default = 0.5, min = 0.0, max = 4.0, apply = live)]
    pub ssgi_intensity: f32,
    /// How far the indirect-light rays reach, in world units. A near-field
    /// effect, so it defaults well below the reflection reach. Only matters
    /// when `technique` is `ssgi`.
    #[asset(default = 8.0, min = 0.5, max = 100.0, apply = live)]
    pub ssgi_max_distance: f32,
    /// Internal resolution of the SSGI trace. `half` (default) trades a little
    /// sharpness for a large performance saving; `full` is native; `quarter` is
    /// the cheapest. Only matters when `technique` is `ssgi`.
    #[asset(apply = rebuild)]
    pub ssgi_resolution: PassResolution,
    /// Hemisphere rays traced per pixel each frame. The indirect light is
    /// accumulated over frames, so more rays settle it faster after the camera
    /// reveals new surfaces or the lighting changes, at a higher cost. Only
    /// matters when `technique` is `ssgi`.
    #[asset(
        default = DEFAULT_SSGI_RAYS,
        min = 1,
        max = ssgi::settings::MAX_RAYS,
        apply = rebuild
    )]
    pub ssgi_rays: u32,
}

impl IndirectLighting {
    /// Clamp the authored `ambient_intensity` to a safe `[0, 16]` multiplier the
    /// backend folds into `LightUniforms` to scale the indirect (ambient / IBL)
    /// term.
    pub fn resolved_ambient(&self) -> f32 {
        self.ambient_intensity.clamp(0.0, 16.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambient_intensity_clamps_to_a_safe_multiplier() {
        let hot = IndirectLighting {
            ambient_intensity: 100.0,
            ..Default::default()
        };
        assert_eq!(hot.resolved_ambient(), 16.0);
        let neg = IndirectLighting {
            ambient_intensity: -2.0,
            ..Default::default()
        };
        assert_eq!(neg.resolved_ambient(), 0.0);
    }

    #[test]
    fn technique_names_parse_in_snake_case() {
        assert_eq!(
            serde_json::from_str::<IndirectTechnique>(r#""ssgi""#).unwrap(),
            IndirectTechnique::Ssgi
        );
        assert_eq!(
            serde_json::from_str::<IndirectTechnique>(r#""ibl""#).unwrap(),
            IndirectTechnique::Ibl
        );
    }
}
