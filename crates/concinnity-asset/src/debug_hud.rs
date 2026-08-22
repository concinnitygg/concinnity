// Developer debug HUD schema.

use crate::{AssetId, de_opt_asset_ref};

/// Requests the developer debug HUD: a set of [TextLabel](#textlabel) chips
/// with diagnostic readouts, anchored to the top-right of the window and
/// toggled with F1 (hidden by default).
///
/// Each label field, when set, receives one chip: `passes_label` a multi-line
/// list of the heaviest rendering steps of the last frame, `mouse_label` the
/// cursor position in window pixels, `camera_label` the live camera pose
/// (position, yaw, pitch) in the exact form a fixed viewpoint is reproduced
/// with, and `sys_label` the worker-thread and host-memory budgets (the job
/// pool's thread count, and the process resident set against the memory
/// budget). Chips whose stat is unavailable stay blank. The chips stack
/// vertically from the top-right corner in the order cursor, then camera, then
/// system, then passes (passes is last because its height varies with the
/// frame's step count), so their on-screen position is fixed by the engine
/// rather than the authored coordinates.
///
/// The always-on frame-rate and GPU-memory readouts live on the separate
/// [StatHud](#stathud).
///
/// Every rendering world receives a `DebugHud` and its chip labels at build
/// time when it declares none, so the example below is only needed to restyle
/// the chips. The HUD only activates in developer contexts: a debug build of
/// the host binary, or a world launched through `cn debug`; release builds
/// leave it inert even when declared. Declare an
/// [EngineDefaults](#enginedefaults) with `"debug_hud": false` to remove it
/// from the build entirely.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DebugHud {
    /// [TextLabel](#textlabel) that receives the per-step GPU-timing chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub passes_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the cursor-position chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub mouse_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the live camera-pose chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub camera_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the thread / memory budget chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub sys_label: Option<AssetId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_hud_claims_no_labels() {
        // Every chip is opt-in: an unset slot means that readout is suppressed
        // rather than drawn somewhere arbitrary.
        let h = DebugHud::default();
        assert!(h.passes_label.is_none());
        assert!(h.mouse_label.is_none());
        assert!(h.camera_label.is_none());
        assert!(h.sys_label.is_none());
    }

    #[test]
    fn each_chip_binds_its_own_label_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let h: DebugHud = serde_json::from_str(
            r#"{"passes_label":"passes_chip","mouse_label":"","camera_label":"cam","sys_label":6}"#,
        )
        .unwrap();
        assert_eq!(h.passes_label, Some(AssetId(11)));
        assert_eq!(h.mouse_label, None);
        assert_eq!(h.camera_label, Some(AssetId(3)));
        assert_eq!(h.sys_label, Some(AssetId(6)));

        let bytes = postcard::to_allocvec(&h).unwrap();
        let back: DebugHud = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.passes_label, Some(AssetId(11)));
        assert_eq!(back.mouse_label, None);
        assert_eq!(back.sys_label, Some(AssetId(6)));
    }
}
