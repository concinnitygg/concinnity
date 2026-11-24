// A strip of tabs, one of them selected.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{Binding, Screen, SpriteFit};
use crate::ecs::Ref;

use super::WidgetStyle;

/// A strip of tabs sharing its width, one of them selected.
///
/// A click on a tab sets the whole number `selected` names to the tab's index
/// and sends a `UiEvent` named by `on_select` carrying it. What each tab shows
/// is the app's: its screens or systems read the same value.
///
/// ```rust
/// # use concinnity_engine::components::{Binding, Tabs};
/// Tabs {
///     tabs: vec!["Scene".into(), "Assets".into(), "Console".into()],
///     selected: Some(Binding::State("tab".into())),
///     width: 360.0,
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
pub struct Tabs {
    /// Each tab's title, left to right.
    pub tabs: Vec<String>,
    /// A whole number set to the selected tab's index: a `state:` or `var:`
    /// binding. Unset, the first tab shows selected.
    pub selected: Option<Binding>,
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width, shared evenly between the tabs.
    #[asset(default = 300.0)]
    pub width: f32,
    /// Height.
    #[asset(default = 28.0)]
    pub height: f32,
    /// The event a click on a tab sends.
    pub on_select: String,
    /// How the tabs draw.
    pub style: WidgetStyle,
    /// When false the tabs are not drawn and take no input.
    #[asset(default = true)]
    pub visible: bool,
    /// How screen-owned tabs map from the reference canvas to the window when
    /// their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) the tabs belong to. Unset, they join the world's HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime index of the tab under the cursor. Not authored.
    #[serde(skip)]
    pub hovered: Option<usize>,
}

impl Tabs {
    // Tab `i`'s rectangle: the strip shared evenly.
    pub(crate) fn tab_rect(&self, i: usize) -> [f32; 4] {
        let w = self.width / self.tabs.len().max(1) as f32;
        [self.x + i as f32 * w, self.y, w, self.height]
    }

    // The tab under `(x, y)`, if any.
    pub(crate) fn tab_at(&self, x: f32, y: f32) -> Option<usize> {
        (0..self.tabs.len()).find(|&i| super::point_in(x, y, self.tab_rect(i)))
    }
}
