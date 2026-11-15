// View-relevance schema.

/// When an effect stops contributing to what the camera sees, so its passes
/// can stand down. The `relevance` section of [`GraphicsConfig`](#graphicsconfig);
/// a field left unset takes the default below.
///
/// Each rule reads the view rather than the world, measuring against the
/// nearest surface the camera sees of what the effect acts on, so the same
/// thresholds hold for a pebble at arm's length and a planet seen from orbit:
///
/// - Ambient occlusion, the screen-space bounce and the reflections run while
///   their reach (the occlusion's radius, each trace's distance) spans
///   `min_reach_pixels` at that surface.
/// - The sun's cascades run while something they shade lies within their
///   shadow distance: a surface in view, fog thick enough near the camera, or
///   a raymarched volume in view. They have no threshold of their own.
/// - The ocean simulation runs while its longest visible wave spans
///   `min_wave_pixels` at the nearest water in view.
/// - The reflection probe a prop rides captures while the prop is in view and
///   spans `min_subject_pixels` across.
/// - The disocclusion mark runs while something moves `min_motion_pixels`
///   against the view in a frame.
///
/// An effect that stood down comes back once its rule holds with a quarter to
/// spare, and stands down only after its rule has failed for half a second, so
/// a camera resting at a threshold does not toggle it. A threshold of 0 keeps
/// its effects on, and `enabled` false keeps every effect on.
///
/// ```rust
/// # use concinnity_engine::components::{GraphicsConfig, RelevanceOverrides};
/// GraphicsConfig {
///     relevance: RelevanceOverrides {
///         min_reach_pixels: Some(2.0),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[asset(section)]
#[serde(default)]
pub struct Relevance {
    /// Whether effects stand down where they do not contribute. Off runs every
    /// effect the other settings ask for, whatever the view.
    #[asset(default = true, apply = live)]
    pub enabled: bool,
    /// How many pixels an effect's reach must span at the nearest surface it
    /// acts on: ambient occlusion's radius, the screen-space bounce's trace and
    /// the reflections' trace. Below it, what the effect adds is smaller than
    /// a pixel.
    #[asset(default = 1.0, min = 0.0, max = 64.0, apply = live)]
    pub min_reach_pixels: f32,
    /// How many pixels the longest wave the ocean simulation shows (the
    /// longest still holding a thousandth of its sea's peak energy) must span
    /// at the nearest visible water for the simulation to run. The water draws
    /// no wave narrower than 3 pixels.
    #[asset(default = 3.0, min = 0.0, max = 64.0, apply = live)]
    pub min_wave_pixels: f32,
    /// How many pixels across the prop a reflection probe rides must appear
    /// for the probe to capture.
    #[asset(default = 16.0, min = 0.0, max = 1024.0, apply = live)]
    pub min_subject_pixels: f32,
    /// How many pixels a surface must move against the view in a frame for
    /// the disocclusion mark to run: the camera's own travel seen at the
    /// nearest still surface, or anything moving in view, wind-swayed grass
    /// included. The mark itself
    /// passes over anything uncovered by less than a pixel.
    #[asset(default = 1.0, min = 0.0, max = 64.0, apply = live)]
    pub min_motion_pixels: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rule_is_on_by_default() {
        let parsed: Relevance = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, Relevance::default());
        assert!(parsed.enabled);
        assert_eq!(parsed.min_reach_pixels, 1.0);
        assert_eq!(parsed.min_wave_pixels, 3.0);
        assert_eq!(parsed.min_subject_pixels, 16.0);
        assert_eq!(parsed.min_motion_pixels, 1.0);
    }
}
