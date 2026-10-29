use super::*;
use crate::components::cook::Camera3D as Camera3DArgs;
use crate::ecs::World;

fn no_builder(_: PlanetShape) -> Option<Box<dyn GroundBuilder>> {
    None
}

fn system(planet: Planet) -> PlanetSystem {
    PlanetSystem::new(planet, inline_ground)
}

// Init publishes everything physics and the renderer read, with ground
// under the camera and the camera's up away from the center.
#[test]
fn init_publishes_the_frame_gravity_and_ground() {
    let mut world = World::new();
    world.push(Camera3D::bake(Camera3DArgs {
        position: [300.0, 10.0, 0.0],
        ..Default::default()
    }));
    let planet = Planet {
        center: [0.0, -50_000.0, 0.0],
        radius: 50_000.0,
        gravity: 9.0,
        ..Default::default()
    };
    let mut system = PlanetSystem::new(planet, no_builder);
    system.init(&mut world.context());
    let ctx = world.context();
    let gravity = ctx.resource::<PlanetGravity>().unwrap();
    assert_eq!(gravity.strength, 9.0);
    assert_eq!(gravity.center, [0.0, -50_000.0, 0.0]);
    let ground = ctx.resource::<PlanetGroundPatch>().unwrap();
    assert_eq!(ground.revision, 0);
    assert_eq!(ground.patch.center[0], 300.0);
    assert_eq!(ctx.resource::<FrameRebases>().unwrap().count(), 0);
    let up = ctx.query::<Camera3D>().next().unwrap().up;
    assert!(up[0] > 0.0 && up[1] > 0.99, "{up:?}");
    assert!(
        ctx.resource::<SkyOrientation>().is_some(),
        "no SkyRotation, so it is ours"
    );
    // 300 m east of the start and 10 m up: on the equator, a third of a
    // degree east, looking north.
    let geo = ctx.resource::<CameraGeo>().unwrap();
    assert!(geo.position.latitude.abs() < 1e-9, "{geo:?}");
    let east = libm::atan2(300.0, 50_010.0).to_degrees();
    assert!((geo.position.longitude - east).abs() < 1e-6, "{geo:?}");
    assert!((geo.position.altitude - 10.9).abs() < 0.01, "{geo:?}");
    assert!(geo.heading.abs() < 1e-3 || (geo.heading - 360.0).abs() < 1e-3);
}

// A camera that strays a kilometer moves the world once the patch for the
// new frame arrives, and stays where it stood on the planet.
#[test]
fn straying_far_moves_the_world_into_a_new_frame() {
    let mut world = World::new();
    world.push(Camera3D::bake(Camera3DArgs::default()));
    let mut system = system(Planet::default());
    system.init(&mut world.context());
    for camera in world.context().query_mut::<Camera3D>() {
        camera.position = [1_200.0, -14.0, 0.0];
    }
    system.step(&mut world.context());
    let before = *world.context().resource::<CameraGeo>().unwrap();
    assert_eq!(
        world.context().resource::<FrameRebases>().unwrap().count(),
        0
    );
    system.step(&mut world.context());
    let ctx = world.context();
    assert_eq!(ctx.resource::<FrameRebases>().unwrap().count(), 1);
    let camera = ctx.query::<Camera3D>().next().unwrap();
    assert!(
        camera.position.iter().all(|c| c.abs() < 1e-2),
        "{:?}",
        camera.position
    );
    let ground = ctx.resource::<PlanetGroundPatch>().unwrap();
    assert_eq!(ground.revision, 1);
    assert!(
        ground.patch.center[0].abs() < 1e-2,
        "built around the camera"
    );
    let gravity = ctx.resource::<PlanetGravity>().unwrap();
    assert!(gravity.center[0].abs() < 1.0 && gravity.center[1] < -49_000.0);
    // 1e-5 degrees is 9 mm here.
    let after = ctx.resource::<CameraGeo>().unwrap();
    assert!((after.position.latitude - before.position.latitude).abs() < 1e-5);
    assert!((after.position.longitude - before.position.longitude).abs() < 1e-5);
    assert!((after.position.altitude - before.position.altitude).abs() < 1e-2);
    assert!(
        (after.heading - before.heading).abs() < 1e-3,
        "{before:?} {after:?}"
    );
}

// With no builder the ground stays where init built it, and a menu freezes
// the system.
#[test]
fn without_a_builder_or_behind_a_menu_nothing_moves() {
    let mut world = World::new();
    world.push(Camera3D::bake(Camera3DArgs::default()));
    let mut fixed = PlanetSystem::new(Planet::default(), no_builder);
    fixed.init(&mut world.context());
    for camera in world.context().query_mut::<Camera3D>() {
        camera.position = [1_200.0, -14.0, 0.0];
    }
    fixed.step(&mut world.context());
    fixed.step(&mut world.context());
    assert_eq!(
        world.context().resource::<FrameRebases>().unwrap().count(),
        0
    );

    let mut paused = system(Planet::default());
    paused.init(&mut world.context());
    world.context().insert_resource(MenuActive(true));
    paused.step(&mut world.context());
    paused.step(&mut world.context());
    assert_eq!(
        world
            .context()
            .resource::<PlanetGroundPatch>()
            .unwrap()
            .revision,
        0
    );
}

// An edit taken up on the next step reshapes the ground under the camera and
// changes the pull; an edit of the pull alone builds no new ground.
#[test]
fn an_edit_reshapes_the_ground_and_the_pull() {
    let mut world = World::new();
    world.push(Camera3D::bake(Camera3DArgs {
        position: [300.0, 10.0, 0.0],
        ..Default::default()
    }));
    let planet = Planet {
        amplitude: 30.0,
        ..Default::default()
    };
    let mut system = system(planet.clone());
    system.init(&mut world.context());
    let before = world.resource::<PlanetGroundPatch>().cloned().unwrap();

    let pulled = Planet {
        gravity: 4.0,
        ..planet
    };
    world.insert_resource(PlanetEdit(pulled.clone()));
    system.step(&mut world.context());
    assert_eq!(world.resource::<PlanetGravity>().unwrap().strength, 4.0);
    assert_eq!(world.resource::<PlanetGroundPatch>().unwrap(), &before);
    assert!(world.resource::<PlanetEdit>().is_none(), "taken up once");

    let reseeded = Planet { seed: 9, ..pulled };
    world.insert_resource(PlanetEdit(reseeded.clone()));
    system.step(&mut world.context());
    let frame = world.resource::<PlanetFrame>().unwrap();
    assert_eq!(frame.shape, reseeded.shape());
    let after = world.resource::<PlanetGroundPatch>().unwrap();
    assert_eq!(after.revision, before.revision + 1);
    assert_ne!(after.patch, before.patch);
}

// New edits are taken up on the next step and published with the area they
// changed; the patch is built again only when that area reaches it, and an
// edit of the generated ground keeps them.
#[test]
fn new_edits_rebuild_the_patch_they_reach() {
    use crate::planet::{CubeFace, EditArea, EditPage, EditPages, PAGE_SAMPLES, TileId};
    let mut world = World::new();
    world.push(Camera3D::bake(Camera3DArgs {
        position: [300.0, 10.0, 0.0],
        ..Default::default()
    }));
    world.push(Planet::default());
    let mut system = system(Planet::default());
    system.init(&mut world.context());
    let before = world.resource::<PlanetGroundPatch>().cloned().unwrap();
    let raised = |face: u8| {
        let page = EditPage::from_samples(&[3.0; PAGE_SAMPLES], &[1.0; PAGE_SAMPLES]).unwrap();
        Arc::new(EditPages::new([(TileId::root(CubeFace(face)), page)]))
    };

    let far = raised(3);
    world.insert_resource(PlanetEditsChange::new(
        Some(Arc::clone(&far)),
        EditArea::around([0.0, -1.0, 0.0], 100.0, 50_000.0),
    ));
    system.step(&mut world.context());
    system.step(&mut world.context());
    let edited = *world.resource::<PlanetEdited>().unwrap();
    assert_eq!(edited.revision, 1);
    let shape = &world.resource::<PlanetFrame>().unwrap().shape;
    assert!(shape.edits.as_ref().is_some_and(|e| Arc::ptr_eq(e, &far)));
    assert_eq!(
        world.resource::<PlanetGroundPatch>().unwrap(),
        &before,
        "out of reach"
    );
    let component = world.query::<Planet>().next().unwrap();
    assert!(
        component
            .edit_pages
            .as_ref()
            .is_some_and(|e| Arc::ptr_eq(e, &far))
    );

    let near = raised(2);
    let change = world.resource_mut::<PlanetEditsChange>().unwrap();
    assert!(!change.is_pending(), "taken, its slot left for the next");
    change.record(
        Some(Arc::clone(&far)),
        EditArea::around([0.0, -1.0, 0.0], 10.0, 50_000.0),
    );
    change.record(
        Some(Arc::clone(&near)),
        EditArea::around([0.006, 1.0, 0.0], 50.0, 50_000.0),
    );
    system.step(&mut world.context());
    system.step(&mut world.context());
    assert_eq!(world.resource::<PlanetEdited>().unwrap().revision, 2);
    assert!(
        world.resource::<PlanetEdited>().unwrap().area.angle > 1.0,
        "the two areas joined"
    );
    let after = world.resource::<PlanetGroundPatch>().cloned().unwrap();
    assert_eq!(after.revision, before.revision + 1);
    assert!((after.patch.center[1] - before.patch.center[1] - 3.0).abs() < 0.01);

    world.insert_resource(PlanetEdit(Planet {
        seed: 9,
        ..Planet::default()
    }));
    system.step(&mut world.context());
    let shape = &world.resource::<PlanetFrame>().unwrap().shape;
    assert_eq!(shape.seed, 9);
    assert!(shape.edits.as_ref().is_some_and(|e| Arc::ptr_eq(e, &near)));
}
