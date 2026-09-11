// The sky mesh is an inside-out cube sized to fit within the camera far plane,
// so it only ever covers the horizon while the camera is inside it. Keeping it
// centered on the camera is what makes that true anywhere in a world, rather
// than only within one cube's reach of wherever it was placed.

use crate::components::Transform;
use crate::ecs::{Entity, PipelineContext};

/// Move every sky prop onto the camera.
///
/// A prop already there is left alone, so a still camera writes nothing and
/// the frame's transform propagation stays on its cached path.
pub(crate) fn center_on_camera(ctx: &mut PipelineContext, props: &[Entity], cam_pos: [f32; 3]) {
    for &entity in props {
        if ctx
            .get::<Transform>(entity)
            .is_some_and(|t| t.position == cam_pos)
        {
            continue;
        }
        match ctx.get_mut::<Transform>(entity) {
            Some(transform) => transform.position = cam_pos,
            None => ctx.insert(
                entity,
                Transform {
                    position: cam_pos,
                    ..Default::default()
                },
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;

    fn world_with_a_prop_at(position: [f32; 3]) -> (World, Entity) {
        let mut world = World::new();
        let entity = world.spawn();
        world.context().insert(
            entity,
            Transform {
                position,
                ..Default::default()
            },
        );
        (world, entity)
    }

    #[test]
    fn the_sky_follows_the_camera() {
        let (mut world, entity) = world_with_a_prop_at([0.0; 3]);
        center_on_camera(&mut world.context(), &[entity], [1.0, 2.0, -300.0]);
        let ctx = world.context();
        let placed = ctx.get::<Transform>(entity).expect("still there");
        assert_eq!(placed.position, [1.0, 2.0, -300.0]);
    }

    // The whole point of the skip: a still camera must not dirty the Transform
    // column, or every frame of a static scene re-resolves the hierarchy.
    #[test]
    fn a_camera_that_has_not_moved_leaves_the_column_untouched() {
        let (mut world, entity) = world_with_a_prop_at([4.0, 0.0, 0.0]);
        let before = world.context().column_ticks::<Transform>().changed;
        center_on_camera(&mut world.context(), &[entity], [4.0, 0.0, 0.0]);
        assert_eq!(world.context().column_ticks::<Transform>().changed, before);
    }

    #[test]
    fn a_prop_with_no_transform_gains_one() {
        let mut world = World::new();
        let entity = world.spawn();
        center_on_camera(&mut world.context(), &[entity], [0.0, 5.0, 0.0]);
        let ctx = world.context();
        assert_eq!(
            ctx.get::<Transform>(entity).expect("inserted").position,
            [0.0, 5.0, 0.0]
        );
    }
}
