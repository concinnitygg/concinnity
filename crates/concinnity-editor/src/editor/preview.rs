// src/editor/preview.rs
//
// The editor "Preview" panel: a small floating panel holding the controls that
// affect how the running world is previewed: the capture-mouse checkbox (hand
// the cursor to the world / play mode) and the fly-camera checkbox (navigate
// the frozen world; also the F key). Escape leaves either. Like the rest of
// the editor HUD it is plain `Sprite` / `TextLabel` components at reserved ids
// (injected by `inject.rs`), driven each frame by the editor hook. The title
// bar, close button, and row draw come from the shared `list_panel`; this
// module only names the ids, width, and its row actions.

use super::list_panel::{self, Row};
use super::registry::{self, PanelKey};
use super::widget::point_in;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Preview);
// Named ids the cross-module tests reference (injection ordering / visibility);
// the shipping paths derive every id from `BASE` through `list_panel`.
#[cfg(test)]
pub(crate) const PANEL_BG: AssetId = list_panel::panel_bg(BASE);
#[cfg(test)]
pub(crate) const ROW_BG: AssetId = list_panel::row_bg(BASE, 0);
#[cfg(test)]
pub(crate) const CHECK_BOX: AssetId = list_panel::check_box(BASE, 0);

const PREVIEW_W: f32 = 200.0;
// The capture row and the fly row.
const ROWS: usize = 2;

// Where the panel sits until the user drags it: the window's top-left, below
// the top bar (clear of its buttons and the Assets panel's default anchor).
pub(crate) fn default_origin() -> [f32; 2] {
    [8.0, super::hud::body_top()]
}

// The panel's fixed footprint, for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    list_panel::size(PREVIEW_W, ROWS)
}

// The panel outer rect (title bar + the capture row).
pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    list_panel::panel_rect(o, PREVIEW_W, ROWS)
}

// A resolved Preview-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreviewAction {
    // The capture checkbox: hand the cursor to / take it from the world.
    ToggleCapture,
    // The fly checkbox: toggle the edit-mode fly camera.
    ToggleFly,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means the
// click missed the panel. Title-bar presses never reach this: the hook intercepts
// them first to start a drag (the shared routing owns the title-bar geometry).
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2]) -> Option<PreviewAction> {
    match list_panel::hit_row(mx, my, o, PREVIEW_W, ROWS) {
        Some(0) => return Some(PreviewAction::ToggleCapture),
        Some(_) => return Some(PreviewAction::ToggleFly),
        None => {}
    }
    point_in(mx, my, panel_rect(o)).then_some(PreviewAction::Consume)
}

// Position + show the panel at origin `o`, colouring each checkbox by state.
pub(crate) fn apply(world: &mut World, o: [f32; 2], capture: bool, fly: bool, mouse: [f32; 2]) {
    let rows = [
        Row::checkbox("Capture input", capture),
        Row::checkbox("Fly camera (F)", fly),
    ];
    list_panel::apply(world, BASE, o, size(), "Preview", &rows, mouse);
}

// Hide every panel element (the F1-hidden pass).
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
    use super::super::widget;
    use super::list_panel::title_label;
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

    #[test]
    fn hit_test_resolves_the_capture_row_and_swallows_the_rest() {
        let o = default_origin();
        let row = list_panel::row_rect(o, PREVIEW_W, 0);
        assert_eq!(
            hit_test(row[0] + 10.0, row[1] + 10.0, o),
            Some(PreviewAction::ToggleCapture)
        );
        let fly_row = list_panel::row_rect(o, PREVIEW_W, 1);
        assert_eq!(
            hit_test(fly_row[0] + 10.0, fly_row[1] + 10.0, o),
            Some(PreviewAction::ToggleFly)
        );
        let t = widget::title_rect(o, PREVIEW_W);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o),
            Some(PreviewAction::Consume),
            "a title click that was not a drag is swallowed"
        );
        assert_eq!(hit_test(900.0, 500.0, o), None, "a miss falls through");
    }

    #[test]
    fn apply_shows_heading_and_capture_state() {
        let mut world = injected_world();
        let o = default_origin();
        apply(&mut world, o, false, false, [0.0, 0.0]);
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == title_label(BASE))
            .unwrap();
        assert!(title.visible && title.content == "Preview");
        let off = world
            .query::<Sprite>()
            .find(|s| s.asset_id == CHECK_BOX)
            .cloned()
            .unwrap();
        apply(&mut world, o, true, false, [0.0, 0.0]);
        let on = world
            .query::<Sprite>()
            .find(|s| s.asset_id == CHECK_BOX)
            .unwrap();
        assert_ne!(off.tint, on.tint, "the checkbox greens while capturing");
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        apply(&mut world, default_origin(), true, true, [0.0, 0.0]);
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
