//! The editor "View" panel: the hub that toggles the visibility of the other
//! floating editor panels. Its rows come from the panel registry
//! (`registry::VIEW_ROWS`, in registry order), so a new panel gets its toggle
//! here without touching this module. Like the rest of the editor HUD it is
//! plain `Sprite` / `TextLabel` components at reserved ids (injected by
//! `inject.rs`), driven each frame by the editor hook -- nothing here reaches
//! the shipped runtime. Each row is a checkbox that reflects, and toggles, one
//! panel's shown state; the top-bar "View" button opens / closes this panel.
//! The title bar, close button, and row draw come from the shared `list_panel`.

use concinnity_core::ecs::World;

use super::list_panel::{self, ListIds, Row};
use super::registry::{self, PanelKey};
use crate::editor::hud_ids::{hud_ids, panel_base};
use crate::editor::widget::{self, point_in};

hud_ids! {
    base: panel_base(PanelKey::View);
    sprites: [
        pub(crate) PANEL_BG,
        CLOSE_BG,
        pub(crate) row_bg[count()],
        pub(crate) check_box[count()],
    ];
    labels: [TITLE_LABEL, CLOSE_LABEL, row_label[count()]];
}

const LIST: ListIds = ListIds {
    panel_bg: PANEL_BG,
    title: TITLE_LABEL,
    close_bg: CLOSE_BG,
    close_label: CLOSE_LABEL,
    row_bg,
    row_label,
    check_box: Some(check_box),
    value_label: None,
};

// The number of toggle rows: one per registered panel that opts into a View row.
pub(crate) const fn count() -> usize {
    registry::VIEW_ROWS.len()
}

// The default (and minimum) panel width; the user can widen it past this.
const VIEW_W: f32 = 200.0;

// A resolved View-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewAction {
    // Toggle the panel behind row `i` (an index into the registry's view
    // toggles, in registry order).
    Toggle(usize),
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: the window's top-left, below the
// Preview panel's default anchor so the two do not overlap at launch.
pub(crate) fn default_origin() -> [f32; 2] {
    let preview = super::preview::default_origin();
    [8.0, preview[1] + list_panel::size(VIEW_W, 1)[1] + 8.0]
}

// The panel's footprint (tracks the registered toggle count), for the hook's
// drag clamp.
pub(crate) fn size() -> [f32; 2] {
    list_panel::size(VIEW_W, count())
}

// Resolve a click at `(mx, my)` against the panel at origin `o`, size `s`.
// `None` means the click missed the panel. Title-bar presses never reach this:
// the hook intercepts them first to start a drag (the shared routing owns the
// title-bar geometry).
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> Option<ViewAction> {
    if let Some(i) = list_panel::hit_row(mx, my, o, s[0], count()) {
        return Some(ViewAction::Toggle(i));
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(ViewAction::Consume)
}

// Position + show the panel at origin `o`, effective size `s`, with the given
// toggle rows (built by the hook from the registry).
pub(crate) fn place(world: &mut World, o: [f32; 2], s: [f32; 2], rows: &[Row], mouse: [f32; 2]) {
    list_panel::place(world, &LIST, o, s, "View", rows, mouse);
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{Sprite, TextLabel};

    fn injected_world() -> World {
        ids().test_world()
    }

    fn rows(states: &[(&str, bool)]) -> Vec<Row> {
        states
            .iter()
            .map(|&(caption, on)| Row::checkbox(caption, on))
            .collect()
    }

    #[test]
    fn hit_test_resolves_a_row_or_swallows() {
        let o = default_origin();
        let s = size();
        let r0 = list_panel::row_rect(o, VIEW_W, 0);
        assert_eq!(
            hit_test(r0[0] + 10.0, r0[1] + 10.0, o, s),
            Some(ViewAction::Toggle(0))
        );
        let t = widget::title_rect(o, VIEW_W);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o, s),
            Some(ViewAction::Consume)
        );
        assert_eq!(hit_test(2000.0, 2000.0, o, s), None);
        // Widening the panel keeps the same rows and swallows body clicks.
        let wide = [VIEW_W + 120.0, s[1]];
        let rw = list_panel::row_rect(o, wide[0], 0);
        assert_eq!(
            hit_test(rw[0] + 10.0, rw[1] + 10.0, o, wide),
            Some(ViewAction::Toggle(0))
        );
        assert_eq!(
            hit_test(o[0] + wide[0] - 4.0, rw[1] + 10.0, o, wide),
            Some(ViewAction::Toggle(0)),
            "the widened right side is still part of the row"
        );
        // Growing the panel taller swallows clicks in the padding below the rows.
        let tall = [VIEW_W, s[1] + 100.0];
        assert_eq!(
            hit_test(o[0] + 5.0, o[1] + s[1] + 40.0, o, tall),
            Some(ViewAction::Consume),
            "the padding below the last row is still part of the panel"
        );
    }

    // One row of elements per View row, and none spare.
    #[test]
    fn injects_one_row_per_view_row() {
        assert_eq!(count(), registry::view_toggles().count());
        assert_eq!(ids().sprites.len(), 2 + 2 * count());
        assert_eq!(ids().labels.len(), 2 + count());
    }

    #[test]
    fn place_shows_heading_and_row_captions() {
        let mut world = injected_world();
        place(
            &mut world,
            default_origin(),
            size(),
            &rows(&[("Assets", false), ("Preview", true), ("Templates", false)]),
            [0.0, 0.0],
        );
        let title = world.get_by_id::<TextLabel>(TITLE_LABEL).unwrap();
        assert!(title.visible && title.content == "View");
        let first = world.get_by_id::<TextLabel>(row_label(0)).unwrap();
        assert_eq!(first.content, "Assets");
    }

    #[test]
    fn checkbox_tints_track_the_row_state() {
        let mut world = injected_world();
        let o = default_origin();
        place(
            &mut world,
            o,
            size(),
            &rows(&[("Assets", false)]),
            [0.0, 0.0],
        );
        let off = world
            .get_by_id::<Sprite>(check_box(0))
            .cloned()
            .unwrap()
            .tint;
        place(
            &mut world,
            o,
            size(),
            &rows(&[("Assets", true)]),
            [0.0, 0.0],
        );
        let on = world.get_by_id::<Sprite>(check_box(0)).unwrap().tint;
        assert_ne!(off, on, "the checkbox tint flips with the panel state");
    }
}
