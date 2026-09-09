// Where a behavior finds an entity.
//
// Most entities answer from their `Transform`. A camera has none: its pose
// lives on the `Camera3D` component the camera systems write, which is also
// where physics, the audio listener and the editor read it from. Asking each in
// turn is what lets `position`, `distance` and the spatial questions name a
// camera, which is how a behavior reaches the player.

use crate::components::{Camera3D, Transform};
use crate::ecs::{ComponentStorage, Entity};

/// The world-space position `entity` answers with, or `None` when nothing on
/// it says where it is.
pub(crate) fn of(components: &ComponentStorage, entity: Entity) -> Option<[f32; 3]> {
    if let Some(transform) = components.get::<Transform>(entity) {
        return Some(transform.position);
    }
    components
        .get::<Camera3D>(entity)
        .map(|camera| camera.position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::cook::Camera3D as Camera3DArgs;

    #[test]
    fn a_transform_answers_for_its_entity() {
        let mut components = ComponentStorage::default();
        let entity = components.spawn();
        components.insert_typed(
            entity,
            Transform {
                position: [1.0, 2.0, 3.0],
                ..Default::default()
            },
        );
        assert_eq!(of(&components, entity), Some([1.0, 2.0, 3.0]));
    }

    // The camera carries its pose on Camera3D, so a behavior naming it gets a
    // position rather than nothing.
    #[test]
    fn a_camera_answers_from_its_own_pose() {
        let mut components = ComponentStorage::default();
        let entity = components.spawn();
        components.insert_typed(
            entity,
            Camera3D::bake(Camera3DArgs {
                position: [4.0, 5.0, 6.0],
                ..Default::default()
            }),
        );
        assert_eq!(of(&components, entity), Some([4.0, 5.0, 6.0]));
    }

    // What the camera systems wrote is what answers, not what it was authored
    // with.
    #[test]
    fn a_camera_answers_where_it_moved_to() {
        let mut components = ComponentStorage::default();
        let entity = components.spawn();
        components.insert_typed(entity, Camera3D::bake(Camera3DArgs::default()));
        components
            .get_mut::<Camera3D>(entity)
            .expect("the camera is there")
            .position = [0.0, 0.0, -12.0];
        assert_eq!(of(&components, entity), Some([0.0, 0.0, -12.0]));
    }

    // A camera that also carries a Transform answers from it, so the general
    // rule keeps deciding and the camera pose is only the fallback.
    #[test]
    fn a_transform_wins_over_the_camera_pose() {
        let mut components = ComponentStorage::default();
        let entity = components.spawn();
        components.insert_typed(
            entity,
            Camera3D::bake(Camera3DArgs {
                position: [4.0, 5.0, 6.0],
                ..Default::default()
            }),
        );
        components.insert_typed(
            entity,
            Transform {
                position: [7.0, 8.0, 9.0],
                ..Default::default()
            },
        );
        assert_eq!(of(&components, entity), Some([7.0, 8.0, 9.0]));
    }

    #[test]
    fn an_entity_with_neither_has_no_position() {
        let mut components = ComponentStorage::default();
        let entity = components.spawn();
        assert_eq!(of(&components, entity), None);
    }
}
