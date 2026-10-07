//! The editor "Templates" panel: a floating list of the engine's built-in world
//! templates; clicking a row applies that template's assets to the world (skipping
//! any whose name already exists, so re-applying is idempotent -- the hook owns
//! that). It is a draggable floating panel toggled from the View panel. Plain
//! `Sprite` / `TextLabel` components at reserved ids (injected by `inject.rs`),
//! driven each frame by the editor hook, so nothing here reaches the shipped
//! runtime. The title bar, close button, and row draw come from the shared
//! `list_panel`; the rows are label-only (no checkbox) and read from the templates
//! crate.

use concinnity_core::ecs::World;

use super::list_panel::{self, ListIds, Row};
use super::registry::PanelKey;
use crate::editor::hud_ids::{hud_ids, panel_base};
use crate::editor::widget::{self, point_in};

// The number of template rows (one per built-in template) and the title of row
// `i`, read from the shared templates crate.
pub(crate) const fn count() -> usize {
    concinnity_cook::authoring::template::TEMPLATES.len()
}
fn title(i: usize) -> &'static str {
    concinnity_cook::authoring::template::TEMPLATES[i].title
}

// Label-only rows: no checkbox, no value.
hud_ids! {
    base: panel_base(PanelKey::Templates);
    sprites: [pub(crate) PANEL_BG, CLOSE_BG, pub(crate) row_bg[count()]];
    labels: [TITLE_LABEL, CLOSE_LABEL, row_label[count()]];
}

const LIST: ListIds = ListIds {
    panel_bg: PANEL_BG,
    title: TITLE_LABEL,
    close_bg: CLOSE_BG,
    close_label: CLOSE_LABEL,
    row_bg,
    row_label,
    check_box: None,
    value_label: None,
};

// The default (and minimum) panel width; the user can widen it past this.
const TEMPLATES_W: f32 = 220.0;

// A resolved Templates-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TemplatesAction {
    // Apply template row `i`.
    Pick(usize),
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: below the top bar, centered so it
// clears the Assets panel (right) and the View / Preview panels (left).
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw * 0.5 - TEMPLATES_W * 0.5, crate::editor::hud::body_top()]
}

// The panel's footprint (tracks the template count), for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    list_panel::size(TEMPLATES_W, count())
}

// Resolve a click at `(mx, my)` against the panel at origin `o`, size `s`.
// `None` means the click missed the panel. Title-bar presses never reach this:
// the hook intercepts them first to start a drag (the shared routing owns the
// title-bar geometry).
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> Option<TemplatesAction> {
    if let Some(i) = list_panel::hit_row(mx, my, o, s[0], count()) {
        return Some(TemplatesAction::Pick(i));
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(TemplatesAction::Consume)
}

// Position + show the panel at origin `o`, effective size `s`, highlighting the
// hovered row and the `selected` row (the one whose detail panel is open).
pub(crate) fn place(
    world: &mut World,
    o: [f32; 2],
    s: [f32; 2],
    selected: Option<usize>,
    mouse: [f32; 2],
) {
    let rows: Vec<Row> = (0..count())
        .map(|i| Row::label(title(i)).select(selected == Some(i)))
        .collect();
    list_panel::place(world, &LIST, o, s, "Templates", &rows, mouse);
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{Sprite, TextLabel};

    fn injected_world() -> World {
        ids().test_world()
    }

    #[test]
    fn hit_test_picks_a_row_or_swallows() {
        let o = default_origin(1280.0);
        let s = size();
        let r0 = list_panel::row_rect(o, TEMPLATES_W, 0);
        assert_eq!(
            hit_test(r0[0] + 10.0, r0[1] + 10.0, o, s),
            Some(TemplatesAction::Pick(0))
        );
        let t = widget::title_rect(o, TEMPLATES_W);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o, s),
            Some(TemplatesAction::Consume)
        );
        assert_eq!(hit_test(2000.0, 2000.0, o, s), None);
    }

    #[test]
    fn place_labels_rows_from_the_templates_crate() {
        let mut world = injected_world();
        place(&mut world, default_origin(1280.0), size(), None, [0.0, 0.0]);
        let title = world.get_by_id::<TextLabel>(TITLE_LABEL).unwrap();
        assert!(title.visible && title.content == "Templates");
        for i in 0..count() {
            let l = world.get_by_id::<TextLabel>(row_label(i)).unwrap();
            assert_eq!(
                l.content,
                concinnity_cook::authoring::template::TEMPLATES[i].title
            );
        }
    }

    // The row whose detail panel is open stays highlighted even without a hover.
    #[test]
    fn selected_row_is_highlighted() {
        let mut world = injected_world();
        let o = default_origin(1280.0);
        // Idle tint first, then the selected tint differs.
        place(&mut world, o, size(), None, [0.0, 0.0]);
        let idle = world.get_by_id::<Sprite>(row_bg(0)).cloned().unwrap().tint;
        place(&mut world, o, size(), Some(0), [0.0, 0.0]);
        let selected = world.get_by_id::<Sprite>(row_bg(0)).unwrap().tint;
        assert_ne!(idle, selected);
    }
}
