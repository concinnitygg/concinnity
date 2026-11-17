// Screen-space hit-region schema.

use crate::components::{Binding, Screen, Sprite, SpriteFit, TextLabel, UiAction};
use crate::ecs::Ref;
use crate::settings::SettingKey;

/// A responsive invisible rectangular region in screen space.
///
/// When clicked, fires an `action`. When hovered, it optionally restyles a
/// referenced [`TextLabel`](#textlabel) (color and/or scale).
///
/// The cursor must be free (not captured for camera control) for events to fire.
///
/// ```rust
/// # use concinnity_engine::components::HitRegion;
/// HitRegion {
///     x: 430.0,
///     y: 330.0,
///     width: 220.0,
///     height: 40.0,
///     hover_color: Some([1.0, 0.85, 0.3]),
///     hover_scale: Some(1.08),
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
pub struct HitRegion {
    /// Left edge of the region in window pixels.
    pub x: f32,
    /// Top edge of the region in window pixels.
    pub y: f32,
    /// Width of the region in window pixels.
    #[asset(default = 100.0)]
    pub width: f32,
    /// Height of the region in window pixels.
    #[asset(default = 40.0)]
    pub height: f32,
    /// A [`TextLabel`](#textlabel) to style on hover. `None` = no label effect.
    pub label: Option<Ref<TextLabel>>,
    /// RGB color applied to the label while hovered. `None` = no change.
    pub hover_color: Option<[f32; 3]>,
    /// Scale applied to the label while hovered. None = no change.
    pub hover_scale: Option<f32>,
    /// What a click fires. Unset fires nothing.
    pub action: Option<UiAction>,
    /// A number, 0 to 1, that dragging across the region sets to the fraction
    /// of its width under the cursor: a `setting:` slider, or a `state:` or
    /// `var:` float. The `drag_handle` sprite follows the value.
    #[serde(default)]
    pub slider: Option<Binding>,
    /// The [Sprite](#sprite) that slides along the region to show its
    /// `slider` value.
    #[serde(default)]
    pub drag_handle: Option<Ref<Sprite>>,
    /// [Screen](#screen) this region belongs to. While a capturing screen is
    /// open, only the topmost one's regions fire; with none open, persistent
    /// screens' regions do. Unset, the region joins the world's HUD: a
    /// persistent window-space screen made for the elements that name none.
    #[serde(default)]
    pub screen: Option<Ref<Screen>>,
    /// Whether this region is inert. A disabled region never hovers or fires.
    /// Set by the engine at runtime (e.g. a settings row whose feature the GPU
    /// cannot provide is disabled and grayed out); you don't set this directly.
    #[serde(default)]
    pub disabled: bool,
    /// When set, this region tracks its referenced [`label`](#hitregion): it
    /// follows the label's vertical position (so a menu the engine lays out at
    /// runtime keeps its buttons clickable) and is inert while the label's text
    /// is empty (so a hidden menu entry does not catch clicks). Requires
    /// `label`.
    #[serde(default)]
    pub follow_label: bool,
    /// How a screen-owned region maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    /// `Bottom` keeps a region aligned with bottom-anchored furniture it
    /// covers. A region spanning the whole reference canvas always covers the
    /// full window regardless of `fit`.
    #[serde(default)]
    pub fit: SpriteFit,
}

impl HitRegion {
    /// The setting this region operates, when it operates one: through a
    /// `setting:` slider, a `Value` action on a `setting:` binding, or a
    /// `Setting` action.
    #[must_use]
    pub fn setting(&self) -> Option<SettingKey> {
        if let Some(Binding::Setting(key)) = &self.slider {
            return Some(*key);
        }
        match &self.action {
            Some(
                UiAction::Setting { key, .. }
                | UiAction::Value {
                    binding: Binding::Setting(key),
                    ..
                },
            ) => Some(*key),
            _ => None,
        }
    }
}
