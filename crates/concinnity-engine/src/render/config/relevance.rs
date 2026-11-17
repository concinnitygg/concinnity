// View-relevance schema.

/// When an effect stops contributing to what the camera sees, so its passes
/// can stand down. The `relevance` section of [`GraphicsConfig`](#graphicsconfig);
/// a field left unset takes the default below.
///
/// Each rule reads the view rather than the world. Every surface the camera
/// sees counts at its nearest visible part: the planet's ground, each water
/// surface, the prop a reflection probe rides, and everything else. An
/// effect runs while any surface it acts on asks for it, so the same
/// thresholds hold for a pebble at arm's length and a planet seen from orbit,
/// and a near prop keeps an effect the far planet behind it would drop:
///
/// - Ambient occlusion, the screen-space bounce and the reflections run while
///   their reach (the occlusion's radius, each trace's distance) spans
///   `min_reach_pixels` at a surface they act on.
/// - The sun's cascades run while something they shade lies within their
///   shadow distance: a surface in view, fog thick enough near the camera, or
///   a raymarched volume in view. They have no threshold of their own.
/// - Ambient occlusion, the bounce and the cascades act between two
///   surfaces, so the prop counts for them only while the ground or water
///   lies within their reach of it. A prop floating far above everything
///   asks for none of them, however near the camera.
/// - Where the prop's reflection probe captures what the prop reflects (not
///   under ray-traced reflections, which take its place), it captures
///   everything from farther off, so the reflections count the prop only
///   while the ground or water lies within their reach of it too. Where the
///   prop alone asks for them, they run over only the part of the screen it
///   covers, fading out over a few pixels past it.
/// - The screen-space bounce also runs only while the auto-exposure shows the
///   view's lit surfaces (its brightest twentieth) at least
///   `min_bounce_light` as bright as it shows a view it has caught up with:
///   the bounce carries a share of the light those surfaces send, so in a
///   night the exposure shows that dark it adds less than a display step. A
///   night the exposure lifts to full, or a pool of lamplight in a dark view,
///   keeps it. Without auto-exposure this part does not apply.
/// - The ocean simulation runs while its longest visible wave spans
///   `min_wave_pixels` at the nearest water in view.
/// - The grass runs while its blades stand tall enough on screen for the
///   field to draw one at the nearest ground in view.
/// - The reflection probe a prop rides captures while the prop is in view and
///   spans `min_subject_pixels` across.
/// - The disocclusion mark runs while something moves `min_motion_pixels`
///   against the view in a frame.
///
/// A surface a recent look at the view did not see, hidden or out of view,
/// asks for nothing. A lower quality preset asks for more pixels before an
/// effect runs: twice the thresholds on low and half again on medium, the
/// motion threshold aside.
///
/// Once an effect's rule fails, ambient occlusion, the screen-space bounce
/// and the reflections fade out over half a second; every effect stands down
/// once its rule has failed for half a second without a break. A surface a
/// look at the view missed still counts until enough looks in a row have
/// missed it to have found a band of it a pixel thin, so a sliver of sea is
/// not taken for hidden. An effect that stood down comes back once its rule
/// holds with a quarter to spare, the three that fade starting from nothing
/// and fading in over a quarter of a second. A camera resting at a threshold
/// does not toggle an effect, and one crossing it sees the occlusion, the
/// bounce and the reflections fade rather than pop. A threshold of 0 keeps
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
    /// How bright the auto-exposure must show the view's lit surfaces for the
    /// screen-space bounce to run, as a share of how bright it shows a view it
    /// has caught up with at no bias: about 1 for an evenly lit view, 0.125
    /// for a night it biases 3 stops down.
    #[asset(default = 0.02, min = 0.0, max = 1.0, apply = live)]
    pub min_bounce_light: f32,
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
        assert_eq!(parsed.min_bounce_light, 0.02);
    }
}
