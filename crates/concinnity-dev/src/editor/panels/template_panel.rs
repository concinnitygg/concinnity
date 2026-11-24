//! The editor's Template detail panel: a floating preview of one engine-owned
//! world template, opened by selecting a row in the Templates list
//! (`template.rs`). Under its title ("Template `<name>`") a header row carries
//! the template's description and an Apply button, then a scrolling list of
//! the template's assets grouped under their types. Applying layers the
//! template's assets into the world (the hook owns that, idempotently).

use concinnity_engine::components::{List, TextAlign, UiEvent, UiItem, UiValue};
use concinnity_engine::ecs::World;

use super::asset_list::{self, ListRow, MAX_ROWS, ROW_H};
use super::registry::PanelKey;
use crate::editor::theme;
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

// The default (and minimum) width; the user can widen the panel past this.
pub(crate) const TPL_W: f32 = 420.0;
// The header row: the description and the Apply button.
const HEADER_H: f32 = 34.0;
const BTN_W: f32 = 84.0;

// The state the panel shows: its title, the description, the asset rows, and
// which template they are for. The Apply button's event.
const TITLE: &str = "editor.template.title";
const DESCRIPTION: &str = "editor.template.description";
const ROWS: &str = "editor.template.rows";
const SHOWN: &str = "editor.template.shown";
const APPLY: &str = "editor.template.apply";

// The panel's height for a template of `n_rows` grouped rows: the rows shown
// in full up to the default window.
fn height(n_rows: usize) -> f32 {
    panel_height(HEADER_H + n_rows.clamp(1, MAX_ROWS) as f32 * ROW_H)
}

// Where the panel sits until the user drags it: centered, just below the top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [(vw - TPL_W) * 0.5, crate::editor::hud::body_top() + 30.0]
}

// The panel footprint for a template of `n_rows` grouped rows. The hook's drag
// clamp uses this.
pub(crate) fn size(n_rows: usize) -> [f32; 2] {
    [TPL_W, height(n_rows)]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let description = ui.bound_label(DESCRIPTION, theme::LABEL_DIM, TextAlign::Left)?;
    let apply = ui.button("Apply", APPLY, None, [BTN_W, HEADER_H - 6.0])?;
    let header = ui.row(vec![description, apply], HEADER_H)?;
    let rows = ui.list(List {
        items: ROWS.to_string(),
        row_height: ROW_H,
        indent: 14.0,
        ..Default::default()
    })?;
    let spec = PanelSpec {
        key: PanelKey::TemplateDetail,
        title: "Template",
        title_state: Some(TITLE),
        size: size(MAX_ROWS),
        min_size: Some(size(1)),
    };
    ui.panel(&spec, vec![header, rows])
}

// Show template `i`, rebuilt only when it is not the one already shown.
pub(crate) fn sync(i: usize, world: &mut World) {
    let shown = i32::try_from(i).unwrap_or(-1);
    let current = world
        .resource::<concinnity_engine::components::UiState>()
        .and_then(|s| s.get(SHOWN).cloned());
    if current == Some(UiValue::Int(shown)) {
        return;
    }
    let Some(t) = concinnity_cook::authoring::template::TEMPLATES.get(i) else {
        return;
    };
    let entries = crate::authoring::world_template_entries(t);
    let rows = items(&asset_list::grouped_rows(&entries, None));
    state::with_state(world, |s| {
        s.set(TITLE, UiValue::Text(format!("Template {}", t.title)));
        s.set(DESCRIPTION, UiValue::Text(t.description.to_string()));
        s.set_list(ROWS, rows);
        s.set(SHOWN, UiValue::Int(shown));
    });
}

// The grouped rows as list items: a type header, its names indented under it.
fn items(rows: &[ListRow]) -> Vec<UiItem> {
    rows.iter()
        .map(|r| UiItem {
            depth: u16::from(!r.is_header),
            dimmed: r.is_header,
            ..UiItem::new(&r.text)
        })
        .collect()
}

// Whether `event` is the Apply button's: layer the template's assets into the
// world.
pub(crate) fn applies(event: &UiEvent) -> bool {
    event.name == APPLY
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::UiState;

    #[test]
    fn sync_shows_a_template_once_until_another_is_picked() {
        let mut world = World::new();
        sync(0, &mut world);
        let s = world.resource::<UiState>().unwrap();
        let t = &concinnity_cook::authoring::template::TEMPLATES[0];
        assert_eq!(
            s.get(TITLE),
            Some(&UiValue::Text(format!("Template {}", t.title)))
        );
        assert!(!s.list(ROWS).is_empty());
        assert!(s.list(ROWS)[0].dimmed, "a type header leads the list");
        let revision = s.revision();
        sync(0, &mut world);
        assert_eq!(world.resource::<UiState>().unwrap().revision(), revision);
    }

    #[test]
    fn the_apply_button_asks_to_apply() {
        let event = UiEvent {
            name: APPLY.into(),
            element: None,
            value: None,
        };
        assert!(applies(&event));
    }
}
