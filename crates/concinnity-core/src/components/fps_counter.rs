// FpsCounter component schema.

use crate::components::TextLabel;
use crate::ecs::Ref;

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
    pub label: Option<Ref<TextLabel>>,
}
