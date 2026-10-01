//! A pen of falling bodies under jointed chains, with sensors over it: the
//! world's dynamic body population.
//!
//! The bodies hang asleep over the pen until the camera arrives. A volume
//! around the station senses the camera crossing into it, and a behavior
//! listening for that wakes every body at once, so the drop happens in front
//! of the camera rather than seconds before it got there.
//!
//! A motor-driven paddle sweeps the floor of the pen, so the pile never
//! settles: left alone it would fall asleep within a minute and the broad phase
//! and the contact solver would measure nothing after that. The solver steps
//! every awake body wherever the camera is, so from the drop on, what this
//! station puts in the report is the physics row of the CPU breakdown in every
//! segment that follows; the pen, the paddle and the chains are what its own
//! segment draws.

use concinnity::components::{
    Behavior, BehaviorExpr, BehaviorNode, BehaviorQuery, BehaviorSource, PhysicsConfig,
    PhysicsJoint, PhysicsJointKind, ProceduralMesh, Prop, PropBody, PropCollider,
    PropColliderShape, TriggerFilter, TriggerVolume,
};
use concinnity::cook::WorldBuilder;

use crate::palette;
use crate::stations::spread;

/// The stretch of path this station is measured under.
pub(crate) const SEGMENT: &str = "physics";

/// The radius the camera circles this station at.
pub(crate) const RADIUS: f32 = 20.0;

// How far past the camera's circle the arrival volume reaches, so the camera
// stays inside it all the way round once it has crossed in.
const ARRIVAL_MARGIN: f32 = 1.0;

// The volume the camera sets off on arrival, and the query the release wakes.
const ARRIVAL: &str = "physics_arrival";
const DYNAMIC_BODIES: &str = "bodies";

// The falling bodies: how they are stacked, how big they are, and how high they
// start.
const BODIES: [usize; 3] = [6, 5, 6];
const BODY_SPACING: f32 = 1.15;
const BODY_RADIUS: f32 = 0.42;
const BODY_DROP_HEIGHT: f32 = 11.0;
// Bouncy enough to keep the pile working for the length of the run.
const BODY_RESTITUTION: f32 = 0.66;

// The pen the bodies land in: its inside half-width and the walls' thickness.
const PEN_HALF_WIDTH: f32 = 7.0;
const PEN_WALL_HALF_HEIGHT: f32 = 2.0;
const PEN_WALL_THICKNESS: f32 = 0.4;

// The jointed chains hanging over the pen, which the falling bodies knock
// about. Each link is a body, and each joint is a constraint the solver has to
// keep satisfied every step.
const CHAINS: usize = 6;
const CHAIN_LINKS: usize = 5;
const LINK_HALF_EXTENT: f32 = 0.35;
const LINK_DROP: f32 = 1.1;
const CHAIN_ANCHOR_HEIGHT: f32 = 9.0;

// The sensors over the pen, which test every dynamic body against their region
// each step.
const SENSORS: usize = 4;

// The paddle turning on a vertical hinge at the middle of the pen: nearly as
// long as the pen is wide, taller than a ball, and held just clear of the floor
// so only the balls touch it.
const PADDLE_HALF_EXTENTS: [f32; 3] = [PEN_HALF_WIDTH - 1.5, 0.55, 0.2];
const PADDLE_CLEARANCE: f32 = 0.05;
const PADDLE_MASS: f32 = 40.0;
const PADDLE_DEGREES_PER_SECOND: f32 = 40.0;
const PADDLE_MAX_FORCE: f32 = 4000.0;

/// Declare the pen, the bodies, the chains, and the sensors.
pub(crate) fn declare(world: &mut WorldBuilder, center: [f32; 3]) {
    world.add(
        "physics_config",
        PhysicsConfig {
            floor_y: 0.0,
            ..Default::default()
        },
    );

    world.add(
        "physics_ball_mesh",
        ProceduralMesh {
            generator: "sphere".to_string(),
            radius: Some(BODY_RADIUS),
            rings: Some(12),
            segments: Some(16),
            ..Default::default()
        },
    );
    world.add(
        "physics_link_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some([LINK_HALF_EXTENT; 3]),
            ..Default::default()
        },
    );

    pen(world, center);

    for x in 0..BODIES[0] {
        for y in 0..BODIES[1] {
            for z in 0..BODIES[2] {
                let index = (x * BODIES[1] + y) * BODIES[2] + z;
                let name = format!("physics_ball_{index}");
                world
                    .add(
                        name.as_str(),
                        Prop {
                            position: [
                                center[0] + spread(x, BODIES[0], BODY_SPACING),
                                BODY_DROP_HEIGHT + y as f32 * BODY_SPACING,
                                center[2] + spread(z, BODIES[2], BODY_SPACING),
                            ],
                            collider: Some(PropCollider {
                                shape: PropColliderShape::Ball,
                                radius: BODY_RADIUS,
                                ..Default::default()
                            }),
                            ..Default::default()
                        },
                    )
                    .reference("mesh", "physics_ball_mesh")
                    .reference("material", palette::BRICK);
                world
                    .add(
                        format!("physics_ball_body_{index}"),
                        PropBody {
                            mass: 1.0,
                            friction: 0.4,
                            restitution: BODY_RESTITUTION,
                            asleep: true,
                            ..Default::default()
                        },
                    )
                    .reference("prop_name", name.as_str());
            }
        }
    }

    chains(world, center);
    sensors(world, center);
    paddle(world, center);
    release(world, center);
}

// The drop: a volume reaching just past the camera's circle, and a behavior
// that wakes every dynamic body the first time the camera crosses into it.
// The stack is all that is asleep, so waking the chains and the paddle with it
// changes nothing.
fn release(world: &mut WorldBuilder, center: [f32; 3]) {
    world.add(
        ARRIVAL,
        TriggerVolume {
            position: arrival_center(center),
            collider: PropCollider {
                shape: PropColliderShape::Ball,
                radius: RADIUS + ARRIVAL_MARGIN,
                ..Default::default()
            },
            detects: TriggerFilter::Player,
            ..Default::default()
        },
    );
    world
        .add(
            "physics_release",
            Behavior {
                on: BehaviorSource::Enter(None),
                once: true,
                queries: vec![BehaviorQuery {
                    name: DYNAMIC_BODIES.to_string(),
                    has: vec!["PropBody".to_string()],
                }],
                body: vec![BehaviorNode::ForEach {
                    query: DYNAMIC_BODIES.to_string(),
                    bind: "body".to_string(),
                    body: vec![BehaviorNode::Wake {
                        target: BehaviorExpr::Bind("body".to_string()),
                    }],
                }],
                ..Default::default()
            },
        )
        .reference("on.enter", ARRIVAL);
}

// The arrival volume sits on the station at the camera's height, so the
// camera's circle is a ring around its middle.
fn arrival_center(center: [f32; 3]) -> [f32; 3] {
    [center[0], crate::track::HEIGHT, center[2]]
}

// Four static walls, so the bodies stay in the frame the camera looks at
// instead of rolling out of it.
fn pen(world: &mut WorldBuilder, center: [f32; 3]) {
    let half_extents = [PEN_HALF_WIDTH, PEN_WALL_HALF_HEIGHT, PEN_WALL_THICKNESS];
    world.add(
        "physics_wall_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some(half_extents),
            ..Default::default()
        },
    );
    for (index, (offset, turn)) in [
        ([0.0, PEN_HALF_WIDTH], 0.0_f32),
        ([0.0, -PEN_HALF_WIDTH], 0.0),
        ([PEN_HALF_WIDTH, 0.0], 90.0),
        ([-PEN_HALF_WIDTH, 0.0], 90.0),
    ]
    .into_iter()
    .enumerate()
    {
        world
            .add(
                format!("physics_wall_{index}"),
                Prop {
                    position: [
                        center[0] + offset[0],
                        PEN_WALL_HALF_HEIGHT,
                        center[2] + offset[1],
                    ],
                    rotation_deg: [0.0, turn, 0.0],
                    collider: Some(PropCollider {
                        shape: PropColliderShape::Cuboid,
                        half_extents,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .reference("mesh", "physics_wall_mesh")
            .reference("material", palette::STONE);
    }
}

// Chains hung from nothing: the first link joints to a world anchor and each
// one below joints to the link above it.
fn chains(world: &mut WorldBuilder, center: [f32; 3]) {
    for chain in 0..CHAINS {
        let along = spread(chain, CHAINS, PEN_HALF_WIDTH * 0.5);
        for link in 0..CHAIN_LINKS {
            let index = chain * CHAIN_LINKS + link;
            let name = format!("physics_link_{index}");
            world
                .add(
                    name.as_str(),
                    Prop {
                        position: [
                            center[0] + along,
                            CHAIN_ANCHOR_HEIGHT - link as f32 * LINK_DROP,
                            center[2] + along * 0.4,
                        ],
                        collider: Some(PropCollider {
                            shape: PropColliderShape::Cuboid,
                            half_extents: [LINK_HALF_EXTENT; 3],
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                )
                .reference("mesh", "physics_link_mesh")
                .reference("material", palette::METAL);
            world
                .add(
                    format!("physics_link_body_{index}"),
                    PropBody {
                        mass: 1.5,
                        friction: 0.5,
                        restitution: 0.2,
                        ..Default::default()
                    },
                )
                .reference("prop_name", name.as_str());

            let joint = world.add(
                format!("physics_joint_{index}"),
                PhysicsJoint {
                    kind: PhysicsJointKind::Spherical,
                    anchor_a: [0.0, LINK_HALF_EXTENT, 0.0],
                    anchor_b: if link == 0 {
                        [
                            center[0] + along,
                            CHAIN_ANCHOR_HEIGHT + LINK_DROP,
                            center[2] + along * 0.4,
                        ]
                    } else {
                        [0.0, -LINK_HALF_EXTENT, 0.0]
                    },
                    ..Default::default()
                },
            );
            joint.reference("body_a", name.as_str());
            if link > 0 {
                joint.reference("body_b", format!("physics_link_{}", index - 1));
            }
        }
    }
}

// The paddle and the motorized hinge that turns it. The hinge anchors it to
// the world at its own center, so it turns in place without falling.
fn paddle(world: &mut WorldBuilder, center: [f32; 3]) {
    world.add(
        "physics_paddle_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some(PADDLE_HALF_EXTENTS),
            ..Default::default()
        },
    );
    let position = [
        center[0],
        PADDLE_HALF_EXTENTS[1] + PADDLE_CLEARANCE,
        center[2],
    ];
    world
        .add(
            "physics_paddle",
            Prop {
                position,
                collider: Some(PropCollider {
                    shape: PropColliderShape::Cuboid,
                    half_extents: PADDLE_HALF_EXTENTS,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .reference("mesh", "physics_paddle_mesh")
        .reference("material", palette::METAL);
    world
        .add(
            "physics_paddle_body",
            PropBody {
                mass: PADDLE_MASS,
                friction: 0.3,
                restitution: 0.1,
                ..Default::default()
            },
        )
        .reference("prop_name", "physics_paddle");
    world
        .add(
            "physics_paddle_hinge",
            PhysicsJoint {
                kind: PhysicsJointKind::Revolute,
                anchor_b: position,
                axis: [0.0, 1.0, 0.0],
                motor_target_velocity: PADDLE_DEGREES_PER_SECOND,
                motor_max_force: PADDLE_MAX_FORCE,
                ..Default::default()
            },
        )
        .reference("body_a", "physics_paddle");
}

// Sensors over the pen. They collide with nothing; the cost is the overlap
// test they run against every dynamic body each step.
fn sensors(world: &mut WorldBuilder, center: [f32; 3]) {
    for index in 0..SENSORS {
        world.add(
            format!("physics_sensor_{index}"),
            TriggerVolume {
                position: [
                    center[0] + spread(index, SENSORS, PEN_HALF_WIDTH * 0.6),
                    3.0,
                    center[2],
                ],
                collider: PropCollider {
                    shape: PropColliderShape::Cuboid,
                    half_extents: [PEN_HALF_WIDTH * 0.3, 2.5, PEN_HALF_WIDTH],
                    ..Default::default()
                },
                detects: TriggerFilter::Props,
                ..Default::default()
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // How many dynamic bodies the station holds: the dropped balls, the chain
    // links, and the paddle.
    const fn body_count() -> usize {
        BODIES[0] * BODIES[1] * BODIES[2] + CHAINS * CHAIN_LINKS + 1
    }

    #[test]
    fn the_pen_holds_what_its_counts_say() {
        assert_eq!(body_count(), 211);
    }

    // The paddle has to turn inside the walls and pass under nothing but the
    // balls: clear of the floor, and taller than a resting ball so it pushes
    // the pile rather than riding over it.
    #[test]
    fn the_paddle_sweeps_inside_the_pen_and_over_the_floor() {
        let [half_length, half_height, half_thickness] = PADDLE_HALF_EXTENTS;
        let reach = (half_length * half_length + half_thickness * half_thickness).sqrt();
        assert!(
            reach < PEN_HALF_WIDTH - PEN_WALL_THICKNESS,
            "it hits a wall"
        );
        assert!(PADDLE_CLEARANCE > 0.0, "it drags on the floor");
        assert!(2.0 * half_height > 2.0 * BODY_RADIUS, "balls roll over it");
    }

    // Where the camera is `at` seconds into the run, walking the track's legs
    // from the pose the camera is authored at.
    fn camera_at(at: f32) -> [f32; 3] {
        let mut from = [0.0, crate::track::HEIGHT, crate::stations::START_Z];
        let mut began = 0.0;
        for leg in crate::track::travel_legs() {
            let seconds = if leg.speed > 0.0 {
                leg.distance / leg.speed
            } else {
                leg.seconds
            };
            let length = leg.direction.iter().map(|d| d * d).sum::<f32>().sqrt();
            let reach = if length > 0.0 {
                leg.distance / length
            } else {
                0.0
            };
            let along = ((at - began) / seconds).clamp(0.0, 1.0);
            let to = [0, 1, 2].map(|i| from[i] + leg.direction[i] * reach);
            if at < began + seconds {
                return [0, 1, 2].map(|i| from[i] + (to[i] - from[i]) * along);
            }
            from = to;
            began += seconds;
        }
        from
    }

    // The drop is set off by the camera arriving: the first moment its path
    // reaches into the arrival volume is the moment this station's segment
    // opens, give or take the approach grazing the volume on its way in.
    #[test]
    fn the_camera_sets_off_the_drop_as_its_segment_opens() {
        let index = crate::stations::STATIONS
            .iter()
            .position(|s| s.segment == SEGMENT)
            .expect("the station is on the corridor");
        let volume = arrival_center(crate::stations::center(index));
        let inside = |p: [f32; 3]| {
            let d = [0, 1, 2].map(|i| p[i] - volume[i]);
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < RADIUS + ARRIVAL_MARGIN
        };
        let step = 0.01;
        let entered = (0..10_000)
            .map(|i| i as f32 * step)
            .find(|&t| inside(camera_at(t)))
            .expect("the camera reaches the station");
        let (opens, closes) = crate::track::segment_window(SEGMENT);
        assert!(
            (opens - 0.5..=opens + step).contains(&entered),
            "the drop is set off at {entered}s, the segment opens at {opens}s"
        );
        // Once in, the camera stays in for the whole of its circle.
        let mut t = entered;
        while t < closes {
            assert!(inside(camera_at(t)), "the camera leaves at {t}s");
            t += step;
        }
    }

    // The bodies have to fit inside the pen they are dropped over, or the outer
    // ones land on the walls and never reach the pile.
    #[test]
    fn the_dropped_stack_fits_inside_the_pen() {
        let half_span = BODY_SPACING * (BODIES[0] - 1) as f32 * 0.5 + BODY_RADIUS;
        assert!(
            half_span < PEN_HALF_WIDTH,
            "{half_span} into {PEN_HALF_WIDTH}"
        );
    }
}
