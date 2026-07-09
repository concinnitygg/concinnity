// src/editor/widget.rs
//
// Small shared helpers for the editor HUD's injected overlay elements. The whole
// HUD (top bar in `hud.rs`, Assets panel in `panel.rs`) is built from plain
// Sprite / TextLabel / TextInput components at reserved ids, driven each frame by
// the editor hook. Both modules repeatedly need to look up one element by its
// reserved id and mutate it; these keep that lookup -- and the identical
// place-a-sprite / point-in-rect logic -- in one place instead of two copies.

use crate::assets::{Sprite, TextInput, TextLabel};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

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
}
