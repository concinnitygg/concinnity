//! A pen of falling bodies under jointed chains, with sensors over it: the
//! world's dynamic body population.
//!
//! The bodies are bouncy and the chains never hang still, so the broad phase
//! and the contact solver keep working for the length of the run rather than
//! going quiet in the first second and measuring nothing after that. The solver
//! steps every body wherever the camera is, so what this station puts in the
//! report is the physics row of the CPU breakdown in every segment; the pen and
//! the chains are what its own segment draws.

use concinnity::components::{
    PhysicsConfig, PhysicsJoint, ProceduralMesh, Prop, PropBody, PropCollider, TriggerFilter,
    TriggerVolume,
};
use concinnity::cook::WorldBuilder;

use crate::palette;
use crate::stations::spread;

/// The stretch of path this station is measured under.
pub(crate) const SEGMENT: &str = "physics";

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
                                shape: "ball".to_string(),
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
                            ..Default::default()
                        },
                    )
                    .reference("prop_name", name.as_str());
            }
        }
    }

    chains(world, center);
    sensors(world, center);
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
                        shape: "cuboid".to_string(),
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
                            shape: "cuboid".to_string(),
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
                    kind: "spherical".to_string(),
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
                    shape: "cuboid".to_string(),
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

    // How many dynamic bodies the station drops.
    const fn body_count() -> usize {
        BODIES[0] * BODIES[1] * BODIES[2] + CHAINS * CHAIN_LINKS
    }

    #[test]
    fn the_pen_holds_what_its_counts_say() {
        assert_eq!(body_count(), 210);
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
