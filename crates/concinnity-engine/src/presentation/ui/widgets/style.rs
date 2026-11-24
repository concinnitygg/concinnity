// The look every widget shares: font, text colors, and the fills it draws
// rows, highlights, and frames in.

use crate::ecs::FontHandle;

/// How a widget draws: its font, its text colors, and the fills of its
/// background, frame, and highlights. Every field has a default, so a widget
/// that sets none draws in the engine's dark UI theme.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct WidgetStyle {
    /// The [Font](#font) the widget's text draws in. Unset draws with the
    /// engine's built-in face.
    pub font: Option<FontHandle>,
    /// Text scale relative to the font's native size.
    #[asset(default = 0.85)]
    pub scale: f32,
    /// Linear-space RGB color of the widget's text.
    #[asset(default = [0.90, 0.90, 0.92])]
    pub text: [f32; 3],
    /// Linear-space RGB color of secondary text: details, hints, and dimmed
    /// rows.
    #[asset(default = [0.60, 0.60, 0.66])]
    pub dim_text: [f32; 3],
    /// RGBA fill behind the widget.
    #[asset(default = [0.11, 0.11, 0.14, 0.97])]
    pub background: [f32; 4],
    /// RGBA color of the hairline framing the widget.
    #[asset(default = [0.30, 0.32, 0.40, 1.0])]
    pub border: [f32; 4],
    /// RGBA fill of the row or control under the cursor.
    #[asset(default = [0.24, 0.27, 0.36, 0.9])]
    pub hover: [f32; 4],
    /// RGBA fill of the selected row or tab.
    #[asset(default = [0.18, 0.28, 0.46, 1.0])]
    pub selected: [f32; 4],
    /// RGBA fill of an active control: a checked box, a pressed button.
    #[asset(default = [0.26, 0.42, 0.66, 1.0])]
    pub accent: [f32; 4],
    /// Corner rounding of the widget's frame and highlights, in pixels.
    #[asset(default = 5.0)]
    pub corner_radius: f32,
}
