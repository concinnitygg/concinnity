// src/editor/preview.rs
//
// The editor "Preview" panel: a small floating panel holding the controls that
// affect how the running world is previewed. For now its only row is the
// capture-mouse checkbox (hand the cursor to the world / play mode; Escape hands
// it back). Like the rest of the editor HUD it is plain `Sprite` / `TextLabel`
// components at reserved ids (injected by `inject.rs`), driven each frame by the
// editor hook. It shares the Assets panel's title-bar style (`widget.rs`);
// dragging the title bar moves it, and the hook owns the position.

use super::widget::{self, place_sprite, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved ids, past the Assets panel's families (see `hud.rs` / `panel.rs`;
// the panel's ids stay below `ID_BASE + 0x400`).
const PREVIEW: u32 = 0x3000_0000 + 0x400;
pub(crate) const TITLE_BG: AssetId = AssetId(PREVIEW);
pub(crate) const TITLE_LABEL: AssetId = AssetId(PREVIEW + 1);
pub(crate) const ROW_BG: AssetId = AssetId(PREVIEW + 2);
pub(crate) const CHECK_BOX: AssetId = AssetId(PREVIEW + 3);
pub(crate) const CHECK_LABEL: AssetId = AssetId(PREVIEW + 4);

// Geometry, in window pixels. Every rect derives from the panel origin `o` (the
// title bar's top-left), like the Assets panel.
const PREVIEW_W: f32 = 200.0;
const ROW_H: f32 = 34.0;
const BOX_SIZE: f32 = 20.0;

const ROW_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.92];
const BOX_TINT_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const BOX_TINT_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];

// Where the panel sits until the user drags it: the window's top-left corner
// (clear of the top-right button bar and the Assets panel's default anchor).
pub(crate) fn default_origin() -> [f32; 2] {
    [8.0, 8.0]
}

// The panel's fixed footprint, for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    [PREVIEW_W, widget::TITLE_H + ROW_H]
}

// The panel outer rect (title bar + the capture row).
pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    let s = size();
    [o[0], o[1], s[0], s[1]]
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], PREVIEW_W, widget::TITLE_H]
}

// The capture-checkbox row below the title bar.
fn capture_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1] + widget::TITLE_H, PREVIEW_W, ROW_H]
}

// A resolved Preview-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreviewAction {
    // The capture checkbox: hand the cursor to / take it from the world.
    ToggleCapture,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means
// the click missed the panel. Title-bar presses never reach this: the hook
// intercepts them first to start a drag.
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2]) -> Option<PreviewAction> {
    if point_in(mx, my, capture_rect(o)) {
        return Some(PreviewAction::ToggleCapture);
    }
    point_in(mx, my, panel_rect(o)).then_some(PreviewAction::Consume)
}

// Position + show the panel at origin `o`, colouring the capture box by state
// (green while the world holds the cursor).
pub(crate) fn apply(world: &mut World, o: [f32; 2], capture: bool) {
    widget::place_title(world, TITLE_BG, TITLE_LABEL, title_rect(o), "Preview");
    let row = capture_rect(o);
    place_sprite(world, ROW_BG, row, ROW_TINT, true);
    let box_tint = if capture { BOX_TINT_ON } else { BOX_TINT_OFF };
    place_sprite(
        world,
        CHECK_BOX,
        [
            row[0] + 8.0,
            row[1] + (ROW_H - BOX_SIZE) * 0.5,
            BOX_SIZE,
            BOX_SIZE,
        ],
        box_tint,
        true,
    );
    if let Some(l) = widget::label_mut(world, CHECK_LABEL) {
        l.x = row[0] + 36.0;
        l.y = row[1] + ROW_H * 0.5 - 10.0;
        l.align = TextAlign::Left;
        l.color = LABEL;
        l.visible = true;
        l.content = "Capture mouse".to_string();
    }
}

// Hide every panel element (the F1-hidden pass).
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
}

// Every panel sprite / label id, for injection and the hidden pass (same
// draw-order contract as the Assets panel's lists).
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    vec![TITLE_BG, ROW_BG, CHECK_BOX]
}
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    vec![TITLE_LABEL, CHECK_LABEL]
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn rects_stack_title_over_the_capture_row() {
        let o = [40.0, 60.0];
        assert_eq!(title_rect(o), [40.0, 60.0, PREVIEW_W, widget::TITLE_H]);
        assert_eq!(
            capture_rect(o)[1],
            60.0 + widget::TITLE_H,
            "the capture row sits under the title bar"
        );
        let p = panel_rect(o);
        assert_eq!(p[3], widget::TITLE_H + ROW_H, "panel is exactly both rows");
    }

    #[test]
    fn hit_test_resolves_capture_row_and_swallows_the_rest() {
        let o = default_origin();
        let row = capture_rect(o);
        assert_eq!(
            hit_test(row[0] + 10.0, row[1] + 10.0, o),
            Some(PreviewAction::ToggleCapture)
        );
        let t = title_rect(o);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o),
            Some(PreviewAction::Consume),
            "a title click that was not a drag is swallowed"
        );
        assert_eq!(hit_test(900.0, 500.0, o), None, "a miss falls through");
    }

    #[test]
    fn apply_shows_heading_checkbox_and_capture_state() {
        let mut world = injected_world();
        let o = default_origin();
        apply(&mut world, o, false);
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == TITLE_LABEL)
            .unwrap();
        assert!(title.visible);
        assert_eq!(title.content, "Preview");
        let label = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == CHECK_LABEL)
            .unwrap();
        assert_eq!(label.content, "Capture mouse");
        assert_eq!(sprite(&world, CHECK_BOX).tint, BOX_TINT_OFF);
        apply(&mut world, o, true);
        assert_eq!(
            sprite(&world, CHECK_BOX).tint,
            BOX_TINT_ON,
            "green while the world holds the cursor"
        );
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        apply(&mut world, default_origin(), true);
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
