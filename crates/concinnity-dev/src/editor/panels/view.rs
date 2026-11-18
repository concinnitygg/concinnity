//! The editor "View" panel: the hub that toggles the visibility of the other
//! floating editor panels. Its rows come from the panel registry
//! (`registry::VIEW_ROWS`, in registry order), so a new panel gets its toggle
//! here without touching this module. Each row is a checkbox bound to the UI
//! state the hook keeps in step with its panel's shown state; flipping one
//! sends the event that toggles the panel. The top-bar "View" button opens /
//! closes this panel.

use std::sync::LazyLock;

use concinnity_engine::components::{Binding, Checkbox, UiEvent};
use concinnity_engine::ecs::World;

use super::registry::{self, PanelKey};
use crate::editor::ui::{BuiltPanel, PanelSpec, ROW_H, UiBuilder, panel_height, state};

// The number of toggle rows: one per registered panel that opts into a View row.
pub(crate) const fn count() -> usize {
    registry::VIEW_ROWS.len()
}

// The default (and minimum) panel width; the user can widen it past this.
const VIEW_W: f32 = 200.0;

// The event a row's checkbox sends, and the state it shows, followed by the
// row's index.
const TOGGLE: &str = "editor.view.";

// Each row's state name, made once.
static ROW_STATE: LazyLock<Vec<String>> =
    LazyLock::new(|| (0..count()).map(|i| format!("{TOGGLE}{i}")).collect());

// Where the panel sits until the user drags it: the window's top-left, below the
// Preview panel's default anchor so the two do not overlap at launch.
pub(crate) fn default_origin() -> [f32; 2] {
    let preview = super::preview::default_origin();
    [8.0, preview[1] + super::preview::size()[1] + 8.0]
}

// The panel's footprint (tracks the registered toggle count), for the hook's
// drag clamp.
pub(crate) fn size() -> [f32; 2] {
    [VIEW_W, panel_height(ROW_H * count() as f32)]
}

// The panel: a checkbox per toggle row.
pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let mut rows = Vec::with_capacity(count());
    for (i, &(_, caption)) in registry::VIEW_ROWS.iter().enumerate() {
        let name = ROW_STATE.get(i)?;
        rows.push(ui.checkbox(
            Checkbox {
                label: caption.to_string(),
                checked: Some(Binding::State(name.clone())),
                on_change: name.clone(),
                ..Default::default()
            },
            ROW_H,
        )?);
    }
    let spec = PanelSpec {
        key: PanelKey::View,
        title: "View",
        title_state: None,
        size: size(),
        min_size: Some(size()),
    };
    ui.panel(&spec, rows)
}

// Show each row's panel as open or closed.
pub(crate) fn sync(open: impl Iterator<Item = bool>, world: &mut World) {
    for (name, open) in ROW_STATE.iter().zip(open) {
        state::set_bool(world, name, open);
    }
}

// The event row `i`'s checkbox sends.
#[cfg(test)]
pub(crate) fn row_event(i: usize) -> UiEvent {
    UiEvent {
        name: ROW_STATE[i].clone(),
        element: None,
        value: None,
    }
}

// The row a View event toggles, if `event` is one.
pub(crate) fn toggled_row(event: &UiEvent) -> Option<usize> {
    event.name.strip_prefix(TOGGLE)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::{UiState, UiValue};
    use concinnity_engine::ecs::asset_id::AssetId;

    #[test]
    fn a_checkbox_per_view_row_bound_to_its_state() {
        let mut world = World::new();
        let mut ui = UiBuilder::new(&mut world, AssetId(9), None);
        build(&mut ui).unwrap();
        let checks: Vec<&Checkbox> = world.query::<Checkbox>().collect();
        assert_eq!(checks.len(), count());
        assert_eq!(checks[0].label, registry::VIEW_ROWS[0].1);
        assert_eq!(
            checks[1].checked,
            Some(Binding::State(format!("{TOGGLE}1"))),
        );
    }

    #[test]
    fn sync_shows_each_rows_panel_state() {
        let mut world = World::new();
        sync([true, false].into_iter(), &mut world);
        let state = world.resource::<UiState>().unwrap();
        assert_eq!(state.get(&ROW_STATE[0]), Some(&UiValue::Bool(true)));
        assert_eq!(state.get(&ROW_STATE[1]), Some(&UiValue::Bool(false)));
    }

    #[test]
    fn a_rows_event_names_its_row() {
        let event = |name: &str| UiEvent {
            name: name.into(),
            element: None,
            value: None,
        };
        assert_eq!(toggled_row(&event("editor.view.3")), Some(3));
        assert_eq!(toggled_row(&event("editor.close")), None);
    }
}
