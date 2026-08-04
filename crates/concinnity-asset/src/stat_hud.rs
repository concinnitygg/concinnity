// Stats HUD schema.

use crate::{AssetId, de_opt_asset_ref};

/// Requests the default on-screen stats HUD. Drives a set of
/// [TextLabel](#textlabel) chips with live engine stats, refreshed on a fixed
/// interval.
///
/// Each label field, when set, receives one chip: `fps_label` the averaged
/// frame rate, `vram_label` the GPU-memory use, `ram_label` the host process
/// memory (resident set size, against the memory budget when known), `ev_label`
/// the auto-exposure value, and `edr_label` the HDR headroom multiplier. Chips
/// whose stat is unavailable stay blank. The frame-rate and GPU-memory chips
/// are shown or hidden from the in-game video settings ("Display performance
/// stats"); the host-memory, exposure, and HDR chips show whenever their
/// reading is available.
///
/// The chips are packed into a tight strip anchored at the top-left of the
/// window, left to right in the order fps, vram, ram, ev, edr; a blank chip
/// reserves no width, so hidden readouts leave no gap. Their on-screen position
/// is fixed by the engine rather than the authored coordinates.
///
/// Developer-facing readouts (per-pass GPU timings, cursor position, live
/// camera pose) live on the separate [DebugHud](#debughud), toggled with F1.
///
/// A world that declares a [MainMenu](#mainmenu) receives a `StatHud`, its
/// chip labels, and their font at build time when it declares none (the
/// menu's performance-stats toggles drive the chips), so the example below is
/// only needed to restyle the chips or run a HUD without a menu. Declare an
/// [EngineDefaults](#enginedefaults) with `"hud": false` to remove the
/// injection entirely.
///
/// ```jsonl
/// {"type":"Font","name":"hud_font","args":{"size_px":20}}
/// {"type":"TextLabel","name":"fps_chip","args":{"font":"hud_font","x":10,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.18,0.32,0.85],"padding":5}}
/// {"type":"TextLabel","name":"vram_chip","args":{"font":"hud_font","x":92,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.18,0.32,0.85],"padding":5}}
/// {"type":"TextLabel","name":"ram_chip","args":{"font":"hud_font","x":192,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.18,0.32,0.85],"padding":5}}
/// {"type":"TextLabel","name":"ev_chip","args":{"font":"hud_font","x":330,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.18,0.32,0.85],"padding":5}}
/// {"type":"TextLabel","name":"edr_chip","args":{"font":"hud_font","x":410,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.18,0.32,0.85],"padding":5}}
/// {"type":"StatHud","name":"hud","args":{"fps_label":"fps_chip","vram_label":"vram_chip","ram_label":"ram_chip","ev_label":"ev_chip","edr_label":"edr_chip"}}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StatHud {
    /// [TextLabel](#textlabel) that receives the frame-rate chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub fps_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the GPU-memory chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub vram_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the host-memory (RSS) chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub ram_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the auto-exposure chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub ev_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the HDR-headroom chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub edr_label: Option<AssetId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_hud_claims_no_labels() {
        // Each chip is opt-in, so an unset slot suppresses that readout instead
        // of drawing it somewhere arbitrary.
        let h = StatHud::default();
        assert!(h.fps_label.is_none());
        assert!(h.vram_label.is_none());
        assert!(h.ram_label.is_none());
        assert!(h.ev_label.is_none());
        assert!(h.edr_label.is_none());
    }

    #[test]
    fn each_chip_binds_its_own_label_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let h: StatHud = serde_json::from_str(
            r#"{"fps_label":"fps_chip","vram_label":"vram","ram_label":"","ev_label":3,
                "edr_label":"edr_chip"}"#,
        )
        .unwrap();
        assert_eq!(h.fps_label, Some(AssetId(8)));
        assert_eq!(h.vram_label, Some(AssetId(4)));
        assert_eq!(h.ram_label, None);
        assert_eq!(h.ev_label, Some(AssetId(3)));
        assert_eq!(h.edr_label, Some(AssetId(8)));

        let bytes = postcard::to_allocvec(&h).unwrap();
        let back: StatHud = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.fps_label, Some(AssetId(8)));
        assert_eq!(back.ram_label, None);
        assert_eq!(back.ev_label, Some(AssetId(3)));
    }
}
