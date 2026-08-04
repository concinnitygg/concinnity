// FpsCounter component schema.

use crate::{AssetId, de_opt_asset_ref};

/// Requests a frames-per-second counter; optionally writes it to a
/// [TextLabel](#textlabel).
///
/// Declaring an `FpsCounter` updates the named [TextLabel](#textlabel) with the
/// current rate once per second. Omit `label` to suppress on-screen display.
///
/// To display an FPS overlay, declare a [Font](#font), a
/// [TextLabel](#textlabel), and an `FpsCounter` that references the label:
///
/// ```jsonl
/// {"$include":"assets/fps_font.json"}
/// {"$include":"assets/fps_text.json"}
/// {"type":"FpsCounter","name":"fps","args":{"label":"fps_text"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct FpsCounter {
    /// A [TextLabel](#textlabel) to update with the current FPS each second.
    /// Leave unset to suppress on-screen display.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub label: Option<AssetId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unset_label_suppresses_the_on_screen_readout() {
        assert!(FpsCounter::default().label.is_none());
        assert!(
            serde_json::from_str::<FpsCounter>(r#"{"label":""}"#)
                .unwrap()
                .label
                .is_none()
        );
    }

    #[test]
    fn a_named_label_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let c: FpsCounter = serde_json::from_str(r#"{"label":"fps_chip"}"#).unwrap();
        assert_eq!(c.label, Some(AssetId(8)));

        let bytes = postcard::to_allocvec(&c).unwrap();
        let back: FpsCounter = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.label, Some(AssetId(8)));
    }
}
