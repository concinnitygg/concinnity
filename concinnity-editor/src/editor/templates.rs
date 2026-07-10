// src/editor/templates.rs
//
// The editor "Templates" panel: a floating list of the engine's built-in world
// templates; clicking a row applies that template's assets to the world (skipping
// any whose name already exists, so re-applying is idempotent -- the hook owns
// that). This used to be a top-bar dropdown; it is now a draggable floating panel
// like the others, toggled from the View panel. Plain `Sprite` / `TextLabel`
// components at reserved ids (injected by `inject.rs`), driven each frame by the
// editor hook, so nothing here reaches the shipped runtime.

use super::widget::{self, place_sprite, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved ids, past the View panel's family (`view.rs`, `ID_BASE + 0x500`). The
// row families sit at `+0x10` / `+0x30`; the built-in template list is a handful
// of entries, well short of that stride.
const TEMPLATES: u32 = 0x3000_0000 + 0x600;
pub(crate) const TITLE_BG: AssetId = AssetId(TEMPLATES);
pub(crate) const TITLE_LABEL: AssetId = AssetId(TEMPLATES + 1);
pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(TEMPLATES + 0x10 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(TEMPLATES + 0x30 + i as u32)
}

// The number of template rows (one per built-in template) and the title of row
// `i`, read from the shared templates crate.
pub(crate) fn count() -> usize {
    concinnity_templates::TEMPLATES.len()
}
fn title(i: usize) -> &'static str {
    concinnity_templates::TEMPLATES[i].title
}

// Geometry, in window pixels. Every rect derives from the panel origin `o` (the
// title bar's top-left).
const TEMPLATES_W: f32 = 220.0;
const ROW_H: f32 = 34.0;
const PAD: f32 = 8.0;

const ROW_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.92];
const ROW_TINT_HOVER: [f32; 4] = [0.22, 0.26, 0.36, 0.98];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];

// A resolved Templates-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TemplatesAction {
    // Apply template row `i`.
    Pick(usize),
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: below the top bar, centred so it
// clears the Assets panel (right) and the View / Preview panels (left).
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw * 0.5 - TEMPLATES_W * 0.5, super::hud::body_top()]
}

// The panel's footprint (tracks the template count), for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    [TEMPLATES_W, widget::TITLE_H + count() as f32 * ROW_H]
}

// The panel outer rect (title bar + the template rows).
pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    let s = size();
    [o[0], o[1], s[0], s[1]]
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], TEMPLATES_W, widget::TITLE_H]
}

// Template row `i`, stacked below the title bar.
fn row_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    [
        o[0],
        o[1] + widget::TITLE_H + i as f32 * ROW_H,
        TEMPLATES_W,
        ROW_H,
    ]
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means the
// click missed the panel. Title-bar presses never reach this: the hook intercepts
// them first to start a drag.
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2]) -> Option<TemplatesAction> {
    for i in 0..count() {
        if point_in(mx, my, row_rect(o, i)) {
            return Some(TemplatesAction::Pick(i));
        }
    }
    point_in(mx, my, panel_rect(o)).then_some(TemplatesAction::Consume)
}

// Position + show the panel at origin `o`, highlighting the hovered row.
pub(crate) fn apply(world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
    widget::place_title(world, TITLE_BG, TITLE_LABEL, title_rect(o), "Templates");
    for i in 0..count() {
        let row = row_rect(o, i);
        let hovered = point_in(mouse[0], mouse[1], row);
        let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
        place_sprite(world, row_bg(i), row, tint, true);
        if let Some(l) = widget::label_mut(world, row_label(i)) {
            l.x = row[0] + PAD;
            l.y = row[1] + ROW_H * 0.5 - 10.0;
            l.align = TextAlign::Left;
            l.color = LABEL;
            l.visible = true;
            l.content = title(i).to_string();
        }
    }
}

// Hide every panel element (the F1-hidden pass, or when the panel is toggled off).
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
}

// Every panel sprite / label id, for injection and the hidden pass.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_BG];
    ids.extend((0..count()).map(row_bg));
    ids
}
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL];
    ids.extend((0..count()).map(row_label));
    ids
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

    #[test]
    fn rows_stack_below_the_title_bar() {
        let o = [40.0, 60.0];
        assert_eq!(title_rect(o), [40.0, 60.0, TEMPLATES_W, widget::TITLE_H]);
        assert_eq!(row_rect(o, 0)[1], 60.0 + widget::TITLE_H);
        let p = panel_rect(o);
        assert_eq!(p[3], widget::TITLE_H + count() as f32 * ROW_H);
    }

    #[test]
    fn hit_test_picks_a_row_or_swallows() {
        let o = default_origin(1280.0);
        let r0 = row_rect(o, 0);
        assert_eq!(
            hit_test(r0[0] + 10.0, r0[1] + 10.0, o),
            Some(TemplatesAction::Pick(0))
        );
        let t = title_rect(o);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o),
            Some(TemplatesAction::Consume)
        );
        assert_eq!(hit_test(2000.0, 2000.0, o), None);
    }

    #[test]
    fn apply_labels_rows_from_the_templates_crate() {
        let mut world = injected_world();
        apply(&mut world, default_origin(1280.0), [0.0, 0.0]);
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == TITLE_LABEL)
            .unwrap();
        assert!(title.visible && title.content == "Templates");
        for i in 0..count() {
            let l = world
                .query::<TextLabel>()
                .find(|l| l.asset_id == row_label(i))
                .unwrap();
            assert_eq!(l.content, concinnity_templates::TEMPLATES[i].title);
        }
    }

    #[test]
    fn hovered_row_is_highlighted() {
        let mut world = injected_world();
        let o = default_origin(1280.0);
        let r0 = row_rect(o, 0);
        apply(&mut world, o, [r0[0] + 10.0, r0[1] + 10.0]);
        let bg = world
            .query::<Sprite>()
            .find(|s| s.asset_id == row_bg(0))
            .unwrap();
        assert_eq!(bg.tint, ROW_TINT_HOVER);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        apply(&mut world, default_origin(1280.0), [0.0, 0.0]);
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
