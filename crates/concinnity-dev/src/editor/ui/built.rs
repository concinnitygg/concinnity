//! What building the editor's panels left in a world: each built panel's root
//! layout and every element it holds, carried as a world resource so the hook
//! finds them in whichever world it is driving.

use concinnity_engine::ecs::FontHandle;
use concinnity_engine::ecs::asset_id::AssetId;

use crate::editor::panels::registry::{PanelKey, PerPanel};

// One built panel.
#[derive(Debug, Clone, Default)]
pub(crate) struct BuiltPanel {
    // The floating root layout.
    pub(crate) root: AssetId,
    // The screen it is built on, and the face its text draws in.
    pub(crate) screen: Option<AssetId>,
    pub(crate) font: Option<FontHandle>,
    // The surface it draws on, its title bar, and the column of its body.
    pub(crate) surface: AssetId,
    pub(crate) bar: AssetId,
    pub(crate) body: AssetId,
    // Every element and layout the panel holds, the root included.
    pub(crate) elements: Vec<AssetId>,
    // The elements the panel changes as it runs, in an order the panel
    // defines.
    pub(crate) slots: Vec<AssetId>,
}

// The panels built into a world, by registry key.
#[derive(Debug, Default)]
pub(crate) struct BuiltUi {
    pub(crate) panels: PerPanel<Option<BuiltPanel>>,
    // The confirmation dialog.
    pub(crate) dialog: Option<crate::editor::modal::BuiltDialog>,
    // The top bar.
    pub(crate) top_bar: Option<crate::editor::hud::TopBar>,
    // The toast stack the notifications show on.
    pub(crate) toasts: Option<AssetId>,
    // The Display and create menus.
    pub(crate) display_menu: Option<AssetId>,
    pub(crate) create_menu: Option<AssetId>,
    // The canvas the viewport's marks draw on, under every panel.
    pub(crate) marks: Option<crate::editor::viewport::marks::BuiltMarks>,
    // The in-engine mouse cursor.
    pub(crate) cursor: Option<AssetId>,
    // Set by the build, until the hook first brings the panels in line with
    // its model.
    pub(crate) fresh: bool,
}

impl BuiltUi {
    pub(crate) fn panel(&self, key: PanelKey) -> Option<&BuiltPanel> {
        self.panels[key].as_ref()
    }

    // Every built panel.
    pub(crate) fn panels_iter(&self) -> impl Iterator<Item = &BuiltPanel> {
        PanelKey::ALL.into_iter().filter_map(|k| self.panel(k))
    }
}
