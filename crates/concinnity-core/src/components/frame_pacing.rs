// Frame-pacing schema.

/// How the render loop paces its frames: how many may be in flight, whether
/// presentation waits for the display refresh, and the frame-rate cap. One per
/// world; with none declared, the defaults below apply.
///
/// Pacing is a preference of whoever runs the build rather than a quality
/// tier, so no graphics quality preset changes it.
///
/// ```rust
/// # use concinnity_core::components::FramePacing;
/// FramePacing {
///     vsync: true,
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
#[serde(default)]
pub struct FramePacing {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rate_is_capped_by_default() {
        assert_eq!(FramePacing::default().fps_cap, 300);
        let parsed: FramePacing = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.fps_cap, 300);
        assert_eq!(parsed.frames_in_flight, 2);
    }
}
