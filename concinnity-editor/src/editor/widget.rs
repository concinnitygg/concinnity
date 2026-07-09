// src/editor/widget.rs
//
// Small shared helpers for the editor HUD's injected overlay elements. The whole
// HUD (top bar in `hud.rs`, Assets panel in `panel.rs`) is built from plain
// Sprite / TextLabel / TextInput components at reserved ids, driven each frame by
// the editor hook. Both modules repeatedly need to look up one element by its
// reserved id and mutate it; these keep that lookup -- and the identical
// place-a-sprite / point-in-rect logic -- in one place instead of two copies.

use crate::assets::{Sprite, TextAlign, TextInput, TextLabel};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Shared chrome for the floating editor panels (Assets, Preview): each has a
// draggable title bar of this height across its top.
pub(crate) const TITLE_H: f32 = 30.0;
const TITLE_TINT: [f32; 4] = [0.15, 0.17, 0.23, 1.0];
const TITLE_LABEL_COLOR: [f32; 3] = [0.92, 0.93, 0.96];

// Draw a panel's title bar: the background strip and its left-aligned heading.
pub(crate) fn place_title(
    world: &mut World,
    bg: AssetId,
    label: AssetId,
    rect: [f32; 4],
    heading: &str,
) {
    place_sprite(world, bg, rect, TITLE_TINT, true);
    if let Some(l) = label_mut(world, label) {
        l.x = rect[0] + 8.0;
        l.y = rect[1] + rect[3] * 0.5 - 10.0;
        l.align = TextAlign::Left;
        l.color = TITLE_LABEL_COLOR;
        l.visible = true;
        l.content = heading.to_string();
    }
}

// Clamp a dragged panel origin so the whole `size` panel stays on screen: a hard
// stop at every window edge. A panel larger than the window pins to the top-left.
pub(crate) fn clamp_origin(pos: [f32; 2], size: [f32; 2], viewport: [f32; 2]) -> [f32; 2] {
    [
        pos[0].clamp(0.0, (viewport[0] - size[0]).max(0.0)),
        pos[1].clamp(0.0, (viewport[1] - size[1]).max(0.0)),
    ]
}

// Find the reserved-id Sprite / TextLabel / TextInput to mutate (or read), or
// `None` if it was not injected -- a caller then simply no-ops, which is how a
// hidden / absent element is handled.
pub(crate) fn sprite_mut(world: &mut World, id: AssetId) -> Option<&mut Sprite> {
    world.query_mut::<Sprite>().find(|s| s.asset_id == id)
}
pub(crate) fn label_mut(world: &mut World, id: AssetId) -> Option<&mut TextLabel> {
    world.query_mut::<TextLabel>().find(|l| l.asset_id == id)
}
pub(crate) fn input_mut(world: &mut World, id: AssetId) -> Option<&mut TextInput> {
    world.query_mut::<TextInput>().find(|t| t.asset_id == id)
}
pub(crate) fn input(world: &World, id: AssetId) -> Option<&TextInput> {
    world.query::<TextInput>().find(|t| t.asset_id == id)
}

// Move + resize the Sprite with `id` to `rect` ([x, y, w, h]), set its tint +
// visibility. A no-op if the sprite is absent.
pub(crate) fn place_sprite(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    tint: [f32; 4],
    visible: bool,
) {
    if let Some(s) = sprite_mut(world, id) {
        s.x = rect[0];
        s.y = rect[1];
        s.width = rect[2];
        s.height = rect[3];
        s.tint = tint;
        s.visible = visible;
    }
}

pub(crate) fn set_sprite_visible(world: &mut World, id: AssetId, visible: bool) {
    if let Some(s) = sprite_mut(world, id) {
        s.visible = visible;
    }
}
pub(crate) fn set_label_visible(world: &mut World, id: AssetId, visible: bool) {
    if let Some(l) = label_mut(world, id) {
        l.visible = visible;
    }
}

// Whether `(x, y)` lies inside `rect` ([x, y, w, h], top-left origin). The pure
// geometry lives in the shared templates crate; re-exported here so the HUD and
// panel keep using `widget::point_in`.
pub(crate) use concinnity_templates::ui::point_in;

#[cfg(test)]
mod tests {
    use super::*;

    fn world_with(ids: &[AssetId]) -> World {
        let mut world = World::new_empty();
        for &id in ids {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
            world.add_component(TextInput {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    #[test]
    fn accessors_find_by_id_or_return_none() {
        let mut world = world_with(&[AssetId(1)]);
        assert!(sprite_mut(&mut world, AssetId(1)).is_some());
        assert!(label_mut(&mut world, AssetId(1)).is_some());
        assert!(input_mut(&mut world, AssetId(1)).is_some());
        assert!(input(&world, AssetId(1)).is_some());
        // An id that was never injected yields None (callers then no-op).
        assert!(sprite_mut(&mut world, AssetId(999)).is_none());
        assert!(input(&world, AssetId(999)).is_none());
    }

    #[test]
    fn place_sprite_sets_geometry_and_is_a_noop_when_absent() {
        let mut world = world_with(&[AssetId(1)]);
        place_sprite(
            &mut world,
            AssetId(1),
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 0.0, 0.0, 1.0],
            true,
        );
        let s = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(1))
            .unwrap();
        assert_eq!((s.x, s.y, s.width, s.height), (5.0, 6.0, 7.0, 8.0));
        assert_eq!(s.tint, [1.0, 0.0, 0.0, 1.0]);
        assert!(s.visible);
        // A missing id changes nothing and does not panic.
        place_sprite(&mut world, AssetId(2), [0.0; 4], [0.0; 4], true);
        assert_eq!(world.query::<Sprite>().count(), 1);
    }

    #[test]
    fn visibility_setters_toggle_the_right_element() {
        let mut world = world_with(&[AssetId(1)]);
        set_sprite_visible(&mut world, AssetId(1), false);
        set_label_visible(&mut world, AssetId(1), false);
        assert!(!world.query::<Sprite>().next().unwrap().visible);
        assert!(!world.query::<TextLabel>().next().unwrap().visible);
    }

    #[test]
    fn place_title_positions_the_strip_and_heading() {
        let mut world = world_with(&[AssetId(1), AssetId(2)]);
        place_title(
            &mut world,
            AssetId(1),
            AssetId(2),
            [40.0, 60.0, 320.0, TITLE_H],
            "Assets",
        );
        let bg = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(1))
            .unwrap();
        assert!(bg.visible);
        assert_eq!(
            (bg.x, bg.y, bg.width, bg.height),
            (40.0, 60.0, 320.0, TITLE_H)
        );
        let l = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(2))
            .unwrap();
        assert!(l.visible);
        assert_eq!(l.content, "Assets");
        assert_eq!(l.align, TextAlign::Left);
        assert_eq!(l.x, 48.0, "heading is inset from the strip's left edge");
    }

    // The clamp hard-stops a panel at every window edge: it can never be dragged
    // even partially off screen, and an oversized panel pins to the top-left.
    #[test]
    fn clamp_origin_keeps_the_whole_panel_on_screen() {
        let size = [320.0, 400.0];
        let vp = [1280.0, 720.0];
        assert_eq!(
            clamp_origin([100.0, 50.0], size, vp),
            [100.0, 50.0],
            "an in-bounds origin is untouched"
        );
        assert_eq!(clamp_origin([-40.0, -9000.0], size, vp), [0.0, 0.0]);
        assert_eq!(
            clamp_origin([2000.0, 700.0], size, vp),
            [vp[0] - size[0], vp[1] - size[1]],
            "stops with the panel's far edges on the window's"
        );
        // Taller than the window: pinned to the top rather than pushed off.
        assert_eq!(clamp_origin([10.0, 300.0], [320.0, 900.0], vp), [10.0, 0.0]);
    }
}
