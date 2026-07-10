// src/editor/view.rs
//
// The editor "View" panel: the hub that toggles the visibility of the other
// floating editor panels (Assets, Preview, Templates). Like the rest of the
// editor HUD it is plain `Sprite` / `TextLabel` components at reserved ids
// (injected by `inject.rs`), driven each frame by the editor hook -- nothing here
// reaches the shipped runtime. Each row is a checkbox that reflects, and toggles,
// one panel's shown state; the top-bar "View" button opens / closes this panel.
// The title bar, close button, and row draw come from the shared `list_panel`.

use super::list_panel::{self, Row};
use super::widget::point_in;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved id base, past the Preview panel's family (`preview.rs`, `ID_BASE +
// 0x400`). Keep the ranges disjoint.
const BASE: u32 = 0x3000_0000 + 0x500;
// Named ids the cross-module tests reference; the shipping paths derive every id
// from `BASE` through `list_panel`.
#[cfg(test)]
pub(crate) const TITLE_BG: AssetId = list_panel::title_bg(BASE);
#[cfg(test)]
pub(crate) fn row_bg(i: usize) -> AssetId {
    list_panel::row_bg(BASE, i)
}
#[cfg(test)]
pub(crate) fn check_box(i: usize) -> AssetId {
    list_panel::check_box(BASE, i)
}

// The toggle rows, in order: their captions and which panel each controls (the
// hook maps `Toggle(i)` back through this order).
pub(crate) const ROWS: usize = 3;
const ROW_LABELS: [&str; ROWS] = ["Assets", "Preview", "Templates"];

const VIEW_W: f32 = 200.0;

// Which panels are currently shown, so each row's checkbox reflects its state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ViewState {
    pub assets: bool,
    pub preview: bool,
    pub templates: bool,
}

impl ViewState {
    // Whether the panel controlled by row `i` is shown (matches `ROW_LABELS`).
    fn on(&self, i: usize) -> bool {
        match i {
            0 => self.assets,
            1 => self.preview,
            2 => self.templates,
            _ => false,
        }
    }
}

// A resolved View-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewAction {
    // Toggle the panel controlled by row `i` (0 Assets, 1 Preview, 2 Templates).
    Toggle(usize),
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: the window's top-left, below the
// Preview panel's default anchor so the two do not overlap at launch.
pub(crate) fn default_origin() -> [f32; 2] {
    [8.0, 8.0 + list_panel::size(VIEW_W, 1)[1] + 8.0]
}

// The panel's fixed footprint, for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    list_panel::size(VIEW_W, ROWS)
}

// The panel outer rect (title bar + the toggle rows).
pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    list_panel::panel_rect(o, VIEW_W, ROWS)
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2]) -> [f32; 4] {
    list_panel::title_rect(o, VIEW_W)
}

// The "X" close button in the title bar's top-right corner.
pub(crate) fn close_rect(o: [f32; 2]) -> [f32; 4] {
    list_panel::close_rect(o, VIEW_W)
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means the
// click missed the panel. Title-bar presses never reach this: the hook intercepts
// them first to start a drag.
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2]) -> Option<ViewAction> {
    if let Some(i) = list_panel::hit_row(mx, my, o, VIEW_W, ROWS) {
        return Some(ViewAction::Toggle(i));
    }
    point_in(mx, my, panel_rect(o)).then_some(ViewAction::Consume)
}

// Position + show the panel at origin `o`, colouring each row's checkbox by its
// panel's shown state (green when shown) and highlighting the hovered row.
pub(crate) fn apply(world: &mut World, o: [f32; 2], state: ViewState, mouse: [f32; 2]) {
    let rows: Vec<Row> = ROW_LABELS
        .iter()
        .enumerate()
        .map(|(i, caption)| Row::checkbox(*caption, state.on(i)))
        .collect();
    list_panel::apply(world, BASE, o, VIEW_W, "View", &rows, mouse);
}

// Hide every panel element (the F1-hidden pass, or when the panel is toggled off).
pub(crate) fn hide_all(world: &mut World) {
    list_panel::hide_all(world, &all_sprite_ids(), &all_label_ids());
}

// Every panel sprite / label id, for injection and the hidden pass.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    list_panel::all_sprite_ids(BASE, ROWS, true)
}
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    list_panel::all_label_ids(BASE, ROWS)
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

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .unwrap()
    }

    fn state(assets: bool, preview: bool, templates: bool) -> ViewState {
        ViewState {
            assets,
            preview,
            templates,
        }
    }

    #[test]
    fn hit_test_resolves_each_row_and_swallows_the_rest() {
        let o = default_origin();
        for i in 0..ROWS {
            let r = list_panel::row_rect(o, VIEW_W, i);
            assert_eq!(
                hit_test(r[0] + 10.0, r[1] + 10.0, o),
                Some(ViewAction::Toggle(i))
            );
        }
        let t = title_rect(o);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o),
            Some(ViewAction::Consume)
        );
        assert_eq!(hit_test(2000.0, 2000.0, o), None, "a miss falls through");
    }

    #[test]
    fn apply_shows_rows_and_reflects_state() {
        let mut world = injected_world();
        let o = default_origin();
        apply(&mut world, o, state(true, false, true), [0.0, 0.0]);
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == title_label(BASE))
            .unwrap();
        assert!(title.visible && title.content == "View");
        for (i, want) in ROW_LABELS.iter().enumerate() {
            let l = world
                .query::<TextLabel>()
                .find(|l| l.asset_id == row_label(BASE, i))
                .unwrap();
            assert_eq!(&l.content, want);
        }
        // Checkboxes reflect the state: Assets on, Preview off, Templates on. The
        // on / off tints differ, so a state flip changes the box.
        let on = sprite(&world, check_box(0)).tint;
        let off = sprite(&world, check_box(1)).tint;
        assert_ne!(on, off);
        assert_eq!(sprite(&world, check_box(2)).tint, on);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        apply(
            &mut world,
            default_origin(),
            state(true, true, true),
            [0.0, 0.0],
        );
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
