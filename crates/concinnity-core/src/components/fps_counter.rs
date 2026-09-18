// FpsCounter component schema.

use crate::components::TextLabel;
use crate::ecs::{Ref, de_opt_ref};

/// Requests a frames-per-second counter; optionally writes it to a
/// [TextLabel](#textlabel).
///
/// Declaring an `FpsCounter` updates the named [TextLabel](#textlabel) with the
/// current rate once per second. Omit `label` to suppress on-screen display.
///
/// To display an FPS overlay, declare a [Font](#font), a
/// [TextLabel](#textlabel), and an `FpsCounter` that references the label:
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
#[derive(Default)]
pub struct FpsCounter {
    /// A [TextLabel](#textlabel) to update with the current FPS each second.
    /// Leave unset to suppress on-screen display.
    #[serde(deserialize_with = "de_opt_ref")]
    pub label: Option<Ref<TextLabel>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::AssetId;

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
        let c: FpsCounter = crate::test_support::from_json(r#"{"label":"fps_chip"}"#);
        assert_eq!(c.label, Some(Ref::new(AssetId(8))));

        let bytes = postcard::to_allocvec(&c).unwrap();
        let back: FpsCounter = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.label, Some(Ref::new(AssetId(8))));
    }
}
