//! What building the editor's panels left in a world: each built panel's root
//! layout and every element it holds, carried as a world resource so the hook
//! finds them in whichever world it is driving.

use concinnity_engine::ecs::asset_id::AssetId;

use crate::editor::panels::registry::{PanelKey, PerPanel};

// One built panel.
#[derive(Debug, Clone, Default)]
pub(crate) struct BuiltPanel {
    // The floating root layout.
    pub(crate) root: AssetId,
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
    // Set by the build, until the hook first brings the panels in line with
    // its model.
    pub(crate) fresh: bool,
}

impl BuiltUi {
    pub(crate) fn panel(&self, key: PanelKey) -> Option<&BuiltPanel> {
        self.panels[key].as_ref()
    }
}
