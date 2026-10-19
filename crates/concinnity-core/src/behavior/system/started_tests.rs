// Behaviors in a world started against the headless table, where the
// load-time decomposition drains the authored Prop column and physics and the
// camera run beside the behavior system.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::components::{
    Behavior, BehaviorExpr, BehaviorNode, BehaviorQuery, BehaviorSource, Camera3D, Prop,
    PropInstance, Transform,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{HEADLESS_SYSTEMS, MeshHandle, Ref, World};

// The core tests drive a bare component storage, where decomposition never
// runs, so they cannot show that a scope survives it. This goes through
// `World::start`, where the authored Prop column is drained, and covers both
// halves of the ModelRenderer-xor-MeshRenderer split a Prop decomposes into.
#[test]
fn a_prop_scoped_behavior_fires_once_started() {
    let mut world = World::new();
    world.push_identified(
        AssetId(1),
        Prop {
            mesh: Some(MeshHandle::new(10)),
            scale: [1.0; 3],
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(2),
        Prop {
            model: Some(Ref::new(AssetId(20))),
            position: [10.0, 0.0, 0.0],
            scale: [1.0; 3],
            ..Default::default()
        },
    );
    world.add_component(Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        body: vec![BehaviorNode::SetTransform {
            entity: BehaviorExpr::SelfEntity,
            position: Some(BehaviorExpr::Add(
                Box::new(BehaviorExpr::Position(Box::new(BehaviorExpr::SelfEntity))),
                Box::new(BehaviorExpr::Vec3([0.0, 1.0, 0.0])),
            )),
            rotation_deg: None,
            scale: None,
            cut: false,
        }],
        ..Default::default()
    });

    world.start(HEADLESS_SYSTEMS).unwrap();
    assert_eq!(
        world.query::<Prop>().count(),
        0,
        "decomposition drained the authored column",
    );
    world.step();

    let mut lifted: Vec<f32> = world
        .join2::<PropInstance, Transform>()
        .map(|(_, _, t)| t.position[1])
        .collect();
    lifted.sort_by(f32::total_cmp);
    assert_eq!(
        lifted,
        vec![1.0, 1.0],
        "both the mesh- and the model-backed prop ran the behavior",
    );
}

// A `set_transform` on a prop the simulation owns is a teleport, not a write
// physics overwrites the same tick: the driver adopts the pose the behavior
// left and simulates on from there.
#[test]
fn a_behavior_moves_a_prop_the_simulation_owns() {
    use crate::components::{PropBody, PropCollider};

    let mut world = World::new();
    world.push_identified(
        AssetId(1),
        Prop {
            mesh: Some(MeshHandle::new(10)),
            position: [0.0, 5.0, 0.0],
            scale: [1.0; 3],
            collider: Some(PropCollider {
                shape: crate::components::PropColliderShape::Ball,
                radius: 0.5,
                half_extents: [0.5; 3],
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    world.add_component(PropBody {
        prop_name: Some(Ref::new(AssetId(1))),
        mass: 1.0,
        ..Default::default()
    });
    world.add_component(Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        body: vec![BehaviorNode::SetTransform {
            entity: BehaviorExpr::SelfEntity,
            position: Some(BehaviorExpr::Add(
                Box::new(BehaviorExpr::Position(Box::new(BehaviorExpr::SelfEntity))),
                Box::new(BehaviorExpr::Vec3([0.0, 4.0, 0.0])),
            )),
            rotation_deg: None,
            scale: None,
            cut: false,
        }],
        ..Default::default()
    });

    world.start(HEADLESS_SYSTEMS).unwrap();
    world.step();

    let y = world
        .join2::<PropInstance, Transform>()
        .map(|(_, _, t)| t.position[1])
        .next()
        .expect("the prop kept its transform");
    assert!(
        (y - 9.0).abs() < 0.05,
        "the body was moved to where the behavior put it (y = {y})",
    );
}

// The documented "act when the player is near" shape, as
// `private/tests/behavior_valid.jsonl` writes it: props scoped, the camera
// queried, its entity bound, and a distance gate deciding. The camera carries
// no Transform, so the gate answers only if a camera says where it is.
#[test]
fn a_distance_gate_on_the_queried_camera_decides_by_where_the_camera_is() {
    use crate::components::cook::Camera3D as Camera3DArgs;

    let mut world = World::new();
    world.push_identified(
        AssetId(1),
        Prop {
            mesh: Some(MeshHandle::new(10)),
            scale: [1.0; 3],
            ..Default::default()
        },
    );
    // Uncontrolled, so nothing but this test moves it.
    world.add_component(Camera3D::bake(Camera3DArgs {
        position: [0.0, 0.0, 100.0],
        controller: None,
        ..Default::default()
    }));
    world.add_component(Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        queries: vec![BehaviorQuery {
            name: "player".into(),
            has: vec!["Camera3D".into()],
        }],
        body: vec![
            BehaviorNode::Let {
                name: "target".into(),
                value: BehaviorExpr::First("player".into()),
            },
            BehaviorNode::If {
                cond: BehaviorExpr::Lt(
                    Box::new(BehaviorExpr::Distance(
                        Box::new(BehaviorExpr::SelfEntity),
                        Box::new(BehaviorExpr::Bind("target".into())),
                    )),
                    Box::new(BehaviorExpr::Float(20.0)),
                ),
                // A fixed step rather than the fixture's `dt`-scaled one, so
                // the assertion does not depend on how long a tick took.
                then: vec![BehaviorNode::SetTransform {
                    entity: BehaviorExpr::SelfEntity,
                    position: Some(BehaviorExpr::Add(
                        Box::new(BehaviorExpr::Position(Box::new(BehaviorExpr::SelfEntity))),
                        Box::new(BehaviorExpr::Mul(
                            Box::new(BehaviorExpr::Normalize(Box::new(BehaviorExpr::Sub(
                                Box::new(BehaviorExpr::Position(Box::new(BehaviorExpr::Bind(
                                    "target".into(),
                                )))),
                                Box::new(BehaviorExpr::Position(Box::new(
                                    BehaviorExpr::SelfEntity,
                                ))),
                            )))),
                            Box::new(BehaviorExpr::Float(2.0)),
                        )),
                    )),
                    rotation_deg: None,
                    scale: None,
                    cut: false,
                }],
                otherwise: Vec::new(),
            },
        ],
        ..Default::default()
    });

    world.start(HEADLESS_SYSTEMS).unwrap();
    let chased = |world: &mut World| {
        world
            .join2::<PropInstance, Transform>()
            .map(|(_, _, t)| t.position[2])
            .next()
            .expect("the prop kept its transform")
    };

    world.step();
    assert_eq!(chased(&mut world), 0.0, "the far camera left the gate shut");

    for camera in world.context().query_mut::<Camera3D>() {
        camera.position = [0.0, 0.0, 5.0];
    }
    world.step();
    assert_eq!(
        chased(&mut world),
        2.0,
        "the near camera opened the gate and the prop stepped toward it",
    );
}
