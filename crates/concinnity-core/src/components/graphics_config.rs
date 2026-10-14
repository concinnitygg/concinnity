// World rendering configuration schema.

use crate::components::Vocabulary;

/// How often each cascaded-shadow-map slice is re-rendered. The shadow pass
/// re-rasterizes all scene geometry into every cascade, so it is one of the
/// heavier passes; updating distant cascades less often cuts that cost.
///
/// `hybrid` (the default) re-renders the nearest cascade every frame (so close
/// shadows stay crisp) and rotates through the farther cascades one per frame.
/// Distant shadows then lag a few frames while the camera moves, which is
/// imperceptible at that range. `every_frame` re-renders all cascades every
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

/// Rendering settings for the world: frame pacing, shadows, and clear color.
/// One per world. The GPU backend is chosen by the engine for the platform and
/// is not user-configurable.
///
/// The shadow and anisotropy defaults describe the quality capable hardware
/// runs, not what every GPU runs: the `Auto` graphics quality preset resolves
/// the detected GPU into a ceiling that caps them tier by tier. Frame pacing
/// (`vsync`, `fps_cap`, `frames_in_flight`) is a user preference rather than a
/// quality tier, so no preset touches it.
///
/// ```json
/// ["GraphicsConfig", { "clear_color": [0.1, 0.1, 0.15, 1.0], "frames_in_flight": 2 }]
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct GraphicsConfig {
    /// Cap the render loop at this many frames, then exit. Unset runs until the
    /// window is closed.
    #[asset(apply = restart)]
    pub max_frames: Option<u64>,
    /// Preferred number of frames in flight. Higher can smooth pacing at the
    /// cost of input latency.
    #[asset(default = 2, min = 1, max = 3, apply = restart)]
    pub frames_in_flight: u32,
    /// Cap the frame rate to the display refresh (vsync). Defaults to `false`:
    /// presentation is not tied to the refresh (DirectX presents with tearing
    /// allowed, Vulkan uses a mailbox present mode) and `fps_cap` alone bounds
    /// the frame rate. Set to `true` to lock presentation to the monitor
    /// refresh, eliminating tearing and the wasted frames that never reach the
    /// screen.
    #[asset(apply = live)]
    pub vsync: bool,
    /// Cap the frame rate to this many frames per second, `300` by default.
    /// `0` leaves the loop uncapped, which is what GPU cost measurement wants: a
    /// capped GPU idles between frames and lowers its clocks. The cap is a
    /// CPU-side frame pacer, so it composes with `vsync`: the more restrictive
    /// of the two wins. Lower caps limit heat, fan noise, and power draw.
    #[asset(default = 300, apply = live)]
    pub fps_cap: u32,
    /// Background clear color [r, g, b, a] in linear 0..1 space.
    #[asset(default = [0.01, 0.01, 0.02, 1.0], apply = restart)]
    pub clear_color: [f32; 4],
    /// Shadow map resolution in texels. `4096` by default, capped by the quality
    /// preset down to 1024 on the lowest tier. Set to 0 to disable shadows.
    #[asset(default = 4096, apply = restart)]
    pub shadow_map_size: u32,
    /// How often shadow cascades are re-rendered. `every_frame` (default)
    /// refreshes them all every frame; `hybrid` amortizes the far cascades
    /// across frames. Only the top quality tier permits `every_frame`, so
    /// everything below it runs `hybrid`.
    #[asset(default = ShadowUpdate::EveryFrame, apply = live)]
    pub shadow_update: ShadowUpdate,
    /// How far from the camera shadows are cast, in world units (e.g. 80). The
    /// cascades cover from the near plane out to this distance; a larger value
    /// shadows more of the scene but spreads the same shadow-map resolution over
    /// more area (softer, blockier shadows). Capped at the camera far plane.
    #[asset(default = 80, apply = live)]
    pub shadow_distance: u32,
    /// Number of shadow cascades (`4` is the default and the maximum).
    /// More cascades keep distant shadows sharper by splitting the view range
    /// into finer slices, at the cost of an extra shadow-map render per cascade;
    /// fewer is cheaper but blockier far from the camera. The slice count covers
    /// the same `shadow_distance` regardless.
    #[asset(default = 4, min = 1, max = 4, apply = live)]
    pub shadow_cascades: u32,
    /// Maximum anisotropic-filtering degree for the scene texture sampler
    /// (albedo + normal maps), e.g. 8. Higher keeps textures viewed at a grazing
    /// angle (floors, walls receding into the distance) sharp instead of blurring
    /// along the minor axis, at a small sampling cost. `1` disables anisotropy
    /// (plain trilinear). `16` by default, capped by the quality preset down to
    /// 4 on the lowest tier.
    #[asset(default = 16, min = 1, max = 16, apply = restart)]
    pub anisotropy: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rate_is_capped_by_default() {
        assert_eq!(GraphicsConfig::default().fps_cap, 300);
        let parsed: GraphicsConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.fps_cap, 300);
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
