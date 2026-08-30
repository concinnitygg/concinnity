// src/editor/templates.rs
//
// The editor "Templates" panel: a floating list of the engine's built-in world
// templates; clicking a row applies that template's assets to the world (skipping
// any whose name already exists, so re-applying is idempotent -- the hook owns
// that). It is a draggable floating panel toggled from the View panel. Plain
// `Sprite` / `TextLabel` components at reserved ids (injected by `inject.rs`),
// driven each frame by the editor hook, so nothing here reaches the shipped
// runtime. The title bar, close button, and row draw come from the shared
// `list_panel`; the rows are label-only (no checkbox) and read from the templates
// crate.

use super::list_panel::{self, Row};
use super::registry::{self, PanelKey};
use super::widget::{self, point_in};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Templates);
// Named ids the cross-module tests reference; the shipping paths derive every id
// from `BASE` through `list_panel`.
#[cfg(test)]
pub(crate) const PANEL_BG: AssetId = list_panel::panel_bg(BASE);
#[cfg(test)]
pub(crate) fn row_bg(i: usize) -> AssetId {
    list_panel::row_bg(BASE, i)
}

// The number of template rows (one per built-in template) and the title of row
// `i`, read from the shared templates crate.
pub(crate) fn count() -> usize {
    concinnity_cook::authoring::template::TEMPLATES.len()
}
fn title(i: usize) -> &'static str {
    concinnity_cook::authoring::template::TEMPLATES[i].title
}

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

// Where the panel sits until the user drags it: below the top bar, centred so it
// clears the Assets panel (right) and the View / Preview panels (left).
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw * 0.5 - TEMPLATES_W * 0.5, super::hud::body_top()]
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
pub(crate) fn apply(
    world: &mut World,
    o: [f32; 2],
    s: [f32; 2],
    selected: Option<usize>,
    mouse: [f32; 2],
) {
    let rows: Vec<Row> = (0..count())
        .map(|i| Row::label(title(i)).select(selected == Some(i)))
        .collect();
    list_panel::apply(world, BASE, o, s, "Templates", &rows, mouse);
}

// Hide every panel element (the F1-hidden pass, or when the panel is toggled off).
pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[]);
}

// Every panel sprite / label id, for injection and the hidden pass (label-only
// rows carry no checkbox).
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    list_panel::all_sprite_ids(BASE, count(), false)
}
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    list_panel::all_label_ids(BASE, count(), false)
}

#[cfg(test)]
mod tests {
    use super::list_panel::{row_label, title_label};
    use super::*;
    use crate::components::{Sprite, TextLabel};

    fn injected_world() -> World {
        let mut world = World::new();
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
    fn apply_labels_rows_from_the_templates_crate() {
        let mut world = injected_world();
        apply(&mut world, default_origin(1280.0), size(), None, [0.0, 0.0]);
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == title_label(BASE))
            .unwrap();
        assert!(title.visible && title.content == "Templates");
        for i in 0..count() {
            let l = world
                .query::<TextLabel>()
                .find(|l| l.asset_id == row_label(BASE, i))
                .unwrap();
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
        apply(&mut world, o, size(), None, [0.0, 0.0]);
        let idle = world
            .query::<Sprite>()
            .find(|s| s.asset_id == row_bg(0))
            .cloned()
            .unwrap()
            .tint;
        apply(&mut world, o, size(), Some(0), [0.0, 0.0]);
        let selected = world
            .query::<Sprite>()
            .find(|s| s.asset_id == row_bg(0))
            .unwrap()
            .tint;
        assert_ne!(idle, selected);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        apply(&mut world, default_origin(1280.0), size(), None, [0.0, 0.0]);
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
