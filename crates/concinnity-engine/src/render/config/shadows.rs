// Cascaded shadow-map schema.

use crate::components::Vocabulary;

/// How often each cascaded-shadow-map slice is re-rendered. The shadow pass
/// re-rasterizes all scene geometry into every cascade, so it is one of the
/// heavier passes; updating distant cascades less often cuts that cost.
///
/// `hybrid` re-renders every cascade covering the first few meters in front of
/// the camera, or out past what a follow camera or a camera track films, each
/// frame (so close shadows stay crisp and keep up with whatever moves there)
/// and rotates through the farther cascades one per frame. At a short reach
/// that can be every cascade. The shadows of things moving far away then lag a
/// few frames, which is imperceptible at that range. `every_frame` re-renders all cascades every
/// frame: pick it for scenes with fast-moving shadow casters where even distant
/// shadow lag is unacceptable. Each cascade is always primed (rendered once)
/// before it is sampled, so there is never missing shadow data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default, Vocabulary)]
pub enum ShadowUpdate {
    /// Re-render every cascade every frame.
    #[vocab("every_frame")]
    EveryFrame,
    /// Re-render the near cascades every frame and the distant ones on a
    /// rotation.
    #[default]
    #[vocab("hybrid")]
    Hybrid,
}

/// Cascaded shadow maps for the world's directional lights: their resolution,
/// how far from the camera they reach, how many cascades split that range, how
/// often each is re-rendered, and how far a coarse far shadow reaches past
/// them. The `shadows` section of [`GraphicsConfig`](#graphicsconfig); a field
/// left unset takes the default below.
///
/// The defaults describe the quality capable hardware runs, not what every GPU
/// runs: the `Auto` graphics quality preset resolves the detected GPU into a
/// ceiling that caps them tier by tier, and scales `distance` down on the lower
/// tiers.
///
/// ```rust
/// # use concinnity_engine::components::{GraphicsConfig, ShadowsOverrides};
/// GraphicsConfig {
///     shadows: ShadowsOverrides {
///         map_size: Some(2048),
///         distance: Some(120),
///         far_distance: Some(2000),
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
pub struct Shadows {
    /// Shadow map resolution in texels. `4096` by default, capped by the quality
    /// preset down to 1024 on the lowest tier. Set to 0 to disable shadows.
    #[asset(default = 4096, apply = restart)]
    pub map_size: u32,
    /// How often shadow cascades are re-rendered. `every_frame` (default)
    /// refreshes them all every frame; `hybrid` amortizes the far cascades
    /// across frames. Only the top quality tier permits `every_frame`, so
    /// everything below it runs `hybrid`.
    #[asset(default = ShadowUpdate::EveryFrame, apply = live)]
    pub update: ShadowUpdate,
    /// How far from the camera shadows are cast, in world units (e.g. 80). The
    /// cascades cover from the near plane out to this distance, and the shadow
    /// fades out over the last 15% of it rather than ending at a line. The
    /// cascades are weighted toward the camera, so a larger value keeps close
    /// shadows nearly as sharp and spends the extra reach on coarser far
    /// cascades. The quality preset scales it: the lowest tier covers half of
    /// it, the next three quarters, and the higher tiers all of it. Capped at
    /// the camera far plane.
    #[asset(default = 80, apply = live)]
    pub distance: u32,
    /// How far from the camera a coarse far shadow reaches past the cascades, in
    /// world units (e.g. 4000), so hills and other large casters still shadow
    /// the ground beyond `distance`. Only large casters draw into it. `0` (the
    /// default) means no far shadow: the cascades fade to fully lit. Unlike
    /// `distance`, the quality preset does not scale it, so far views look the
    /// same on every tier.
    #[asset(default = 0, apply = live)]
    pub far_distance: u32,
    /// Number of shadow cascades (`4` is the default and the maximum).
    /// More cascades keep distant shadows sharper by splitting the view range
    /// into finer slices, at the cost of an extra shadow-map render per cascade;
    /// fewer is cheaper but blockier far from the camera. The slice count covers
    /// the same `distance` regardless.
    #[asset(default = 4, min = 1, max = 4, apply = live)]
    pub cascades: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cascade_refreshes_each_frame_by_default() {
        let parsed: Shadows = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.update, ShadowUpdate::EveryFrame);
        assert_eq!(parsed.map_size, 4096);
        assert_eq!(parsed.cascades, 4);
        assert_eq!(parsed.far_distance, 0);
    }

    #[test]
    fn far_distance_parses_beside_the_cascade_reach() {
        let parsed: Shadows =
            serde_json::from_str(r#"{"distance": 250, "far_distance": 4000}"#).unwrap();
        assert_eq!(parsed.distance, 250);
        assert_eq!(parsed.far_distance, 4000);
    }

    #[test]
    fn shadow_update_names_parse_in_snake_case() {
        assert_eq!(
            serde_json::from_str::<ShadowUpdate>(r#""every_frame""#).unwrap(),
            ShadowUpdate::EveryFrame
        );
        assert_eq!(
            serde_json::from_str::<ShadowUpdate>(r#""hybrid""#).unwrap(),
            ShadowUpdate::Hybrid
        );
        assert_eq!(
            serde_json::to_string(&ShadowUpdate::EveryFrame).unwrap(),
            r#""every_frame""#
        );
    }
}
