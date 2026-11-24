//! The editor "Templates" panel: a floating list of the engine's built-in world
//! templates. Selecting a row opens that template's detail panel, which applies
//! it (skipping any asset whose name already exists, so re-applying is
//! idempotent -- the hook owns that). It is a draggable floating panel toggled
//! from the View panel; the row whose detail is open stays selected.

use concinnity_engine::components::{Binding, List, UiEvent, UiItem, UiValue};
use concinnity_engine::ecs::World;

use super::registry::PanelKey;
use crate::editor::ui::{BuiltPanel, PanelSpec, ROW_H, UiBuilder, panel_height, state};

// The number of template rows (one per built-in template) and the title of row
// `i`, read from the shared templates crate.
pub(crate) const fn count() -> usize {
    concinnity_cook::authoring::template::TEMPLATES.len()
}

// The list the rows are kept in, the row selected, and the event a selection
// sends.
const ROWS: &str = "editor.templates";
const SELECTED: &str = "editor.templates.selected";
const PICK: &str = "editor.templates.pick";

// The default (and minimum) panel width; the user can widen it past this.
const TEMPLATES_W: f32 = 220.0;

// Where the panel sits until the user drags it: below the top bar, centered so it
// clears the Assets panel (right) and the View / Preview panels (left).
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw * 0.5 - TEMPLATES_W * 0.5, crate::editor::hud::body_top()]
}

// The panel's footprint (tracks the template count), for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    [TEMPLATES_W, panel_height(ROW_H * count() as f32)]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let list = ui.list(List {
        items: ROWS.to_string(),
        selected: Some(Binding::State(SELECTED.to_string())),
        row_height: ROW_H,
        on_select: PICK.to_string(),
        ..Default::default()
    })?;
    let spec = PanelSpec {
        key: PanelKey::Templates,
        title: "Templates",
        title_state: None,
        size: size(),
        min_size: Some(size()),
    };
    ui.panel(&spec, vec![list])
}

// Show the template titles, once, and the row whose detail is open.
pub(crate) fn sync(selected: Option<usize>, world: &mut World) {
    state::with_state(world, |s| {
        if s.list(ROWS).is_empty() {
            let rows = concinnity_cook::authoring::template::TEMPLATES
                .iter()
                .map(|t| UiItem::new(t.title))
                .collect();
            s.set_list(ROWS, rows);
        }
        let row = selected.and_then(|i| i32::try_from(i).ok()).unwrap_or(-1);
        s.set(SELECTED, UiValue::Int(row));
    });
}

// The template a Templates event picks, if `event` is one.
pub(crate) fn picked(event: &UiEvent) -> Option<usize> {
    if event.name != PICK {
        return None;
    }
    match event.value {
        Some(UiValue::Int(i)) => usize::try_from(i).ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::UiState;

    #[test]
    fn sync_lists_the_templates_and_selects_the_open_one() {
        let mut world = World::new();
        sync(Some(0), &mut world);
        let s = world.resource::<UiState>().unwrap();
        let titles: Vec<&str> = s.list(ROWS).iter().map(|r| r.text.as_str()).collect();
        let expected: Vec<&str> = concinnity_cook::authoring::template::TEMPLATES
            .iter()
            .map(|t| t.title)
            .collect();
        assert_eq!(titles, expected);
        assert_eq!(s.get(SELECTED), Some(&UiValue::Int(0)));
        sync(None, &mut world);
        let s = world.resource::<UiState>().unwrap();
        assert_eq!(s.get(SELECTED), Some(&UiValue::Int(-1)));
    }

    #[test]
    fn a_selection_picks_its_template() {
        let event = UiEvent {
            name: PICK.into(),
            element: None,
            value: Some(UiValue::Int(1)),
        };
        assert_eq!(picked(&event), Some(1));
    }
}
