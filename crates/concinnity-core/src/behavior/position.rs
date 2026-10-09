// Where a behavior finds an entity, and how it moves one.
//
// Most entities answer from their `Transform`. A camera has none: its pose
// lives on the `Camera3D` component the camera systems write, which is also
// where physics, the audio listener and the editor read it from. Asking each in
// turn is what lets `position`, `distance` and the spatial questions name a
// camera, which is how a behavior reaches the player, and what lets
// `set_transform` move one.

use crate::components::{Camera3D, CameraTrack, Transform};
use crate::ecs::{ComponentStorage, Entity, PipelineContext};

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

/// The transform a `set_transform` on `entity` starts from: its own, or a
/// camera's pose with pitch as the `x` rotation and yaw as the `y`.
pub(crate) fn transform_of(components: &ComponentStorage, entity: Entity) -> Option<Transform> {
    if let Some(transform) = components.get::<Transform>(entity) {
        return Some(*transform);
    }
    components.get::<Camera3D>(entity).map(|camera| Transform {
        position: camera.position,
        rotation_deg: [camera.pitch.to_degrees(), camera.yaw.to_degrees(), 0.0],
        ..Transform::default()
    })
}

/// What a `set_transform` moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Moved {
    /// The entity's own transform.
    Transform,
    /// The pose of the camera it carries.
    Camera,
    /// Nothing: the entity has neither.
    Nothing,
}

/// Write `transform` onto `entity` the way [`transform_of`] read it.
pub(crate) fn write(
    components: &mut ComponentStorage,
    entity: Entity,
    transform: Transform,
) -> Moved {
    if let Some(current) = components.get_mut::<Transform>(entity) {
        *current = transform;
        return Moved::Transform;
    }
    let Some(camera) = components.get_mut::<Camera3D>(entity) else {
        return Moved::Nothing;
    };
    camera.set_pose(
        transform.position,
        transform.rotation_deg[1].to_radians(),
        transform.rotation_deg[0].to_radians(),
    );
    Moved::Camera
}

/// What rewrites `camera`'s pose every tick, undoing a write to it from
/// anywhere else: the world's [`CameraTrack`], or the camera's follow
/// controller. `None` for a camera that keeps what is written to it.
pub fn camera_driver(ctx: &PipelineContext, camera: Entity) -> Option<&'static str> {
    if ctx.query::<CameraTrack>().next().is_some() {
        return Some("the world's CameraTrack");
    }
    let follows = ctx
        .get::<Camera3D>(camera)
        .and_then(|c| c.controller.as_ref())
        .is_some_and(|c| c.follow.is_some());
    follows.then_some("its follow controller")
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

    #[test]
    fn a_camera_pose_reads_as_a_transform_and_writes_back() {
        let mut components = ComponentStorage::default();
        let entity = components.spawn();
        components.insert_typed(
            entity,
            Camera3D::bake(Camera3DArgs {
                position: [1.0, 2.0, 3.0],
                yaw: 0.5,
                pitch: -0.25,
                ..Default::default()
            }),
        );
        let mut transform = transform_of(&components, entity).expect("a camera pose");
        assert_eq!(transform.position, [1.0, 2.0, 3.0]);
        assert!((transform.rotation_deg[0] - (-0.25f32).to_degrees()).abs() < 1e-4);
        assert!((transform.rotation_deg[1] - 0.5f32.to_degrees()).abs() < 1e-4);

        transform.position = [9.0, 8.0, 7.0];
        transform.rotation_deg = [10.0, 90.0, 45.0];
        assert_eq!(write(&mut components, entity, transform), Moved::Camera);
        let camera = components.get::<Camera3D>(entity).expect("the camera");
        assert_eq!(camera.position, [9.0, 8.0, 7.0]);
        assert!((camera.yaw - core::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert!((camera.pitch - 10f32.to_radians()).abs() < 1e-5);
        let expected = crate::gfx::camera::view_matrix(camera.position, camera.yaw, camera.pitch);
        assert_eq!(camera.view_matrix, expected);
    }

    #[test]
    fn an_entity_with_neither_pose_is_not_written() {
        let mut components = ComponentStorage::default();
        let entity = components.spawn();
        assert!(transform_of(&components, entity).is_none());
        assert_eq!(
            write(&mut components, entity, Transform::default()),
            Moved::Nothing
        );
    }

    #[test]
    fn a_camera_is_driven_by_a_track_or_a_follow_controller() {
        use crate::components::{CameraController, FollowController};
        use crate::ecs::World;

        let mut world = World::new();
        let free = world.push(Camera3D::bake(Camera3DArgs::default()));
        let follow = world.push(Camera3D::bake(Camera3DArgs {
            controller: Some(CameraController {
                follow: Some(FollowController::default()),
                ..CameraController::default()
            }),
            ..Default::default()
        }));
        assert_eq!(camera_driver(&world.context(), free), None);
        assert!(camera_driver(&world.context(), follow).is_some());
        world.push(CameraTrack::default());
        assert!(camera_driver(&world.context(), free).is_some());
    }
}
