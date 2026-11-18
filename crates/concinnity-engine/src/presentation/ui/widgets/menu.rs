// A popup menu of entries, each firing an action.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{Binding, Screen, SpriteFit, UiAction};
use crate::ecs::Ref;

use super::WidgetStyle;

/// One entry of a [Menu](#menu).
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct MenuEntry {
    /// The entry's text.
    pub label: String,
    /// Dimmer text at the entry's right end, such as its key shortcut.
    pub shortcut: String,
    /// What picking the entry fires.
    pub action: Option<UiAction>,
    /// Whether the entry draws dimmed and cannot be picked.
    pub disabled: bool,
    /// Whether a divider line draws above the entry.
    pub separator: bool,
}

/// A popup menu: a column of entries drawn over the rest of its
/// [Screen](#screen) while the boolean `open` names is true.
///
/// A [`HitRegion`](#hitregion) or [`KeyBinding`](#keybinding) opens it by
/// setting or flipping that boolean, such as `{"flip": "state:file_menu"}`.
/// It opens at `x` and `y`, or at the cursor when `at_cursor` is set, as a
/// context menu does, and moves in to stay on screen. Picking an entry fires
/// its `action` and closes the menu; a click outside it or Escape closes it
/// without a pick. While it is open, it takes the clicks over its screen.
///
/// Its entries are `entries`, or the items of the list in the app's UI state
/// that `items` names, picking one of which sends a `UiEvent` named by
/// `on_pick` carrying its index.
///
/// ```rust
/// # use concinnity_engine::components::{Binding, Menu, MenuEntry, UiAction};
/// Menu {
///     open: Some(Binding::State("file_menu".into())),
///     entries: vec![MenuEntry {
///         label: "Quit".into(),
///         shortcut: "Ctrl+Q".into(),
///         action: Some(UiAction::Quit),
///         ..Default::default()
///     }],
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
pub struct Menu {
    /// The boolean that shows the menu while true: a `state:` or `var:`
    /// binding. Closing the menu sets it false.
    pub open: Option<Binding>,
    /// The entries, top to bottom.
    pub entries: Vec<MenuEntry>,
    /// A list in the app's UI state whose items are the entries instead.
    pub items: String,
    /// The event picking an entry of `items` sends.
    pub on_pick: String,
    /// Left edge of the open menu.
    pub x: f32,
    /// Top edge of the open menu.
    pub y: f32,
    /// Whether the menu opens at the cursor instead of at `x` and `y`.
    pub at_cursor: bool,
    /// Width.
    #[asset(default = 220.0)]
    pub width: f32,
    /// Height of each entry.
    #[asset(default = 26.0)]
    pub row_height: f32,
    /// How the menu draws.
    pub style: WidgetStyle,
    /// How a screen-owned menu maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) the menu opens over. Unset, it joins the world's HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime top-left corner of the open menu. Not authored.
    #[serde(skip)]
    pub at: Option<[f32; 2]>,
    /// Runtime index of the entry under the cursor. Not authored.
    #[serde(skip)]
    pub hovered: Option<usize>,
}

impl Menu {
    // How many entries the menu shows, from `entries` or the list `items`.
    pub(crate) fn entry_count(&self, state: Option<&crate::components::UiState>) -> usize {
        if self.items.is_empty() {
            self.entries.len()
        } else {
            state.map_or(0, |s| s.list(&self.items).len())
        }
    }

    // The open menu's top-left corner.
    pub(crate) fn origin(&self) -> [f32; 2] {
        self.at.unwrap_or([self.x, self.y])
    }

    // The menu's rectangle around `count` entries.
    pub(crate) fn rect(&self, count: usize) -> [f32; 4] {
        let [x, y] = self.origin();
        [x, y, self.width, count as f32 * self.row_height]
    }

    // Entry `i`'s rectangle.
    pub(crate) fn row_rect(&self, i: usize) -> [f32; 4] {
        let [x, y] = self.origin();
        [
            x,
            y + i as f32 * self.row_height,
            self.width,
            self.row_height,
        ]
    }

    // The entry under `(x, y)`, if any.
    pub(crate) fn row_at(&self, x: f32, y: f32, count: usize) -> Option<usize> {
        (0..count).find(|&i| super::point_in(x, y, self.row_rect(i)))
    }

    // Place the menu at `at`, moved in so all `count` entries fit in `area`.
    pub(crate) fn open_at(&mut self, at: [f32; 2], count: usize, area: [f32; 2]) {
        let h = count as f32 * self.row_height;
        self.at = Some([
            at[0].min(area[0] - self.width).max(0.0),
            at[1].min(area[1] - h).max(0.0),
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_menu_opening_near_an_edge_moves_in_to_fit() {
        let mut m = Menu {
            width: 200.0,
            row_height: 20.0,
            ..Default::default()
        };
        m.open_at([1200.0, 700.0], 3, [1280.0, 720.0]);
        assert_eq!(m.origin(), [1080.0, 660.0]);
        assert_eq!(m.row_at(1100.0, 685.0, 3), Some(1));
        m.open_at([10.0, 10.0], 3, [1280.0, 720.0]);
        assert_eq!(m.origin(), [10.0, 10.0]);
    }
}
