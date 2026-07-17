// src/editor/view.rs
//
// The editor "View" panel: the hub that toggles the visibility of the other
// floating editor panels. Its rows come from the panel registry (every panel
// with a `view_row` caption, in registry order), so a new panel gets its toggle
// here without touching this module. Like the rest of the editor HUD it is
// plain `Sprite` / `TextLabel` components at reserved ids (injected by
// `inject.rs`), driven each frame by the editor hook -- nothing here reaches
// the shipped runtime. Each row is a checkbox that reflects, and toggles, one
// panel's shown state; the top-bar "View" button opens / closes this panel.
// The title bar, close button, and row draw come from the shared `list_panel`.

use super::list_panel::{self, Row};
use super::registry::{self, PanelKey};
use super::widget::point_in;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::View);
// Named ids the cross-module tests reference; the shipping paths derive every id
// from `BASE` through `list_panel`.
#[cfg(test)]
pub(crate) const PANEL_BG: AssetId = list_panel::panel_bg(BASE);
#[cfg(test)]
pub(crate) fn row_bg(i: usize) -> AssetId {
    list_panel::row_bg(BASE, i)
}
#[cfg(test)]
pub(crate) fn check_box(i: usize) -> AssetId {
    list_panel::check_box(BASE, i)
}

// The number of toggle rows: one per registered panel that opts into a View row.
pub(crate) fn count() -> usize {
    registry::view_toggle_count()
}

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

// The panel outer rect (title bar + the toggle rows).
pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    list_panel::panel_rect(o, VIEW_W, count())
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means the
// click missed the panel. Title-bar presses never reach this: the hook intercepts
// them first to start a drag (the shared routing owns the title-bar geometry).
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2]) -> Option<ViewAction> {
    if let Some(i) = list_panel::hit_row(mx, my, o, VIEW_W, count()) {
        return Some(ViewAction::Toggle(i));
    }
    point_in(mx, my, panel_rect(o)).then_some(ViewAction::Consume)
}

// Position + show the panel at origin `o` with the given toggle rows (built by
// the hook from the registry).
pub(crate) fn apply(world: &mut World, o: [f32; 2], rows: &[Row], mouse: [f32; 2]) {
    list_panel::apply(world, BASE, o, VIEW_W, "View", rows, mouse);
}

// Hide every panel element (the F1-hidden pass, or when the panel is toggled off).
pub(crate) fn hide_all(world: &mut World) {
    list_panel::hide_all(world, &all_sprite_ids(), &all_label_ids());
}

// Every panel sprite / label id, for injection and the hidden pass.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    list_panel::all_sprite_ids(BASE, count(), true)
}
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    list_panel::all_label_ids(BASE, count())
}

#[cfg(test)]
mod tests {
    use super::list_panel::{row_label, title_label};
    use super::*;
    use crate::assets::{Sprite, TextLabel};

    fn injected_world() -> World {
        let mut world = World::new_empty();
        for id in all_sprite_ids() {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in all_label_ids() {
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
        }
        world
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
        let r0 = list_panel::row_rect(o, VIEW_W, 0);
        assert_eq!(
            hit_test(r0[0] + 10.0, r0[1] + 10.0, o),
            Some(ViewAction::Toggle(0))
        );
        let t = list_panel::title_rect(o, VIEW_W);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o),
            Some(ViewAction::Consume)
        );
        assert_eq!(hit_test(2000.0, 2000.0, o), None);
    }

    #[test]
    fn count_tracks_the_registry() {
        assert_eq!(count(), registry::view_toggle_count());
        assert!(count() >= 3, "Assets / Preview / Templates are registered");
    }

    #[test]
    fn apply_shows_heading_and_row_captions() {
        let mut world = injected_world();
        apply(
            &mut world,
            default_origin(),
            &rows(&[("Assets", false), ("Preview", true), ("Templates", false)]),
            [0.0, 0.0],
        );
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == title_label(BASE))
            .unwrap();
        assert!(title.visible && title.content == "View");
        let first = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(BASE, 0))
            .unwrap();
        assert_eq!(first.content, "Assets");
    }

    #[test]
    fn checkbox_tints_track_the_row_state() {
        let mut world = injected_world();
        let o = default_origin();
        apply(&mut world, o, &rows(&[("Assets", false)]), [0.0, 0.0]);
        let off = world
            .query::<Sprite>()
            .find(|s| s.asset_id == check_box(0))
            .cloned()
            .unwrap()
            .tint;
        apply(&mut world, o, &rows(&[("Assets", true)]), [0.0, 0.0]);
        let on = world
            .query::<Sprite>()
            .find(|s| s.asset_id == check_box(0))
            .unwrap()
            .tint;
        assert_ne!(off, on, "the checkbox tint flips with the panel state");
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        apply(
            &mut world,
            default_origin(),
            &rows(&[("Assets", true), ("Preview", true), ("Templates", true)]),
            [0.0, 0.0],
        );
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
