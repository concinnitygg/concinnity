//! A colonnade under a field of small local lights: the station that loads
//! forward shading.
//!
//! The lights are many and short-ranged rather than few and wide, so the
//! clusters they fall into differ from one another and a lit pixel sums a
//! different handful depending on where it is. The cost lands in the main pass,
//! where that sum is taken, rather than in the cull that decides it.

use concinnity::components::{PointLight, ProceduralMesh, Prop, RectAreaLight, ReflectionProbe};
use concinnity::cook::WorldBuilder;

use crate::palette;
use crate::stations::spread;

// The chamber the lights hang in.
const FLOOR_HALF_EXTENT: f32 = 16.0;
const WALL_HEIGHT: f32 = 14.0;

// The colonnade the lights play over.
const PILLARS: [usize; 2] = [6, 3];
const PILLAR_SPACING: f32 = 4.4;
const PILLAR_RADIUS: f32 = 0.55;
const PILLAR_HEIGHT: f32 = 9.0;

// The light field: a grid of point lights, each reaching only a little past its
// neighbours so a cluster's list stays short and the count is what costs.
const LIGHTS: [usize; 3] = [5, 3, 4];
const LIGHT_SPACING: [f32; 3] = [5.0, 3.6, 5.6];
const LIGHT_BASE_HEIGHT: f32 = 2.6;
const LIGHT_RANGE: f32 = 7.5;
const LIGHT_INTENSITY: f32 = 14.0;

// The soft sources on the walls, which shade through a different path from the
// point lights beside them.
const PANELS: usize = 4;

/// Declare the chamber, the light field, and the probe over both.
pub(crate) fn declare(world: &mut WorldBuilder, center: [f32; 3]) {
    world.add(
        "lights_floor_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some([FLOOR_HALF_EXTENT, 0.2, FLOOR_HALF_EXTENT]),
            ..Default::default()
        },
    );
    world
        .add(
            "lights_floor",
            Prop {
                position: [center[0], 0.2, center[2]],
                ..Default::default()
            },
        )
        .reference("mesh", "lights_floor_mesh")
        .reference("material", palette::TILE);

    world.add(
        "lights_wall_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some([FLOOR_HALF_EXTENT, WALL_HEIGHT * 0.5, 0.4]),
            ..Default::default()
        },
    );
    world
        .add(
            "lights_wall",
            Prop {
                position: [
                    center[0] + FLOOR_HALF_EXTENT * 0.6,
                    WALL_HEIGHT * 0.5,
                    center[2],
                ],
                rotation_deg: [0.0, 90.0, 0.0],
                ..Default::default()
            },
        )
        .reference("mesh", "lights_wall_mesh")
        .reference("material", palette::BRICK);

    world.add(
        "lights_pillar_mesh",
        ProceduralMesh {
            generator: "cylinder".to_string(),
            radius: Some(PILLAR_RADIUS),
            height: Some(PILLAR_HEIGHT),
            segments: Some(24),
            ..Default::default()
        },
    );
    for x in 0..PILLARS[0] {
        for z in 0..PILLARS[1] {
            let index = x * PILLARS[1] + z;
            world
                .add(
                    format!("lights_pillar_{index}"),
                    Prop {
                        position: [
                            center[0] + spread(x, PILLARS[0], PILLAR_SPACING),
                            PILLAR_HEIGHT * 0.5,
                            center[2] + spread(z, PILLARS[1], PILLAR_SPACING),
                        ],
                        ..Default::default()
                    },
                )
                .reference("mesh", "lights_pillar_mesh")
                .reference("material", palette::STONE);
        }
    }

    for x in 0..LIGHTS[0] {
        for y in 0..LIGHTS[1] {
            for z in 0..LIGHTS[2] {
                let index = (x * LIGHTS[1] + y) * LIGHTS[2] + z;
                world.add(
                    format!("lights_point_{index}"),
                    PointLight {
                        position: [
                            center[0] + spread(x, LIGHTS[0], LIGHT_SPACING[0]),
                            LIGHT_BASE_HEIGHT + y as f32 * LIGHT_SPACING[1],
                            center[2] + spread(z, LIGHTS[2], LIGHT_SPACING[2]),
                        ],
                        color: bulb_colour(index),
                        intensity: LIGHT_INTENSITY,
                        range: LIGHT_RANGE,
                    },
                );
            }
        }
    }

    // The emissive strips the panels stand in front of, so the soft sources
    // read as coming from somewhere.
    world.add(
        "lights_strip_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some([2.4, 0.9, 0.08]),
            ..Default::default()
        },
    );
    for index in 0..PANELS {
        let along = spread(index, PANELS, FLOOR_HALF_EXTENT * 0.55);
        let at = [
            center[0] + FLOOR_HALF_EXTENT * 0.55,
            WALL_HEIGHT * 0.45,
            center[2] + along,
        ];
        world
            .add(
                format!("lights_strip_{index}"),
                Prop {
                    position: [at[0] + 0.2, at[1], at[2]],
                    rotation_deg: [0.0, 90.0, 0.0],
                    ..Default::default()
                },
            )
            .reference("mesh", "lights_strip_mesh")
            .reference("material", palette::GLOW);
        world.add(
            format!("lights_panel_{index}"),
            RectAreaLight {
                center: at,
                normal: [-1.0, 0.0, 0.0],
                half_size: [2.4, 0.9],
                color: [1.0, 0.86, 0.70],
                intensity: 26.0,
                range: 22.0,
                two_sided: false,
            },
        );
    }

    world.add(
        "lights_probe",
        ReflectionProbe {
            position: [center[0], WALL_HEIGHT * 0.35, center[2]],
            half_extents: [FLOOR_HALF_EXTENT, WALL_HEIGHT * 0.5, FLOOR_HALF_EXTENT],
        },
    );
}

// A repeating spread of warm and cool bulbs, so neighbouring clusters never
// hold identical lists.
fn bulb_colour(index: usize) -> [f32; 3] {
    const WHEEL: [[f32; 3]; 5] = [
        [1.00, 0.62, 0.36],
        [0.42, 0.72, 1.00],
        [0.96, 0.94, 0.72],
        [0.58, 1.00, 0.68],
        [0.94, 0.50, 0.72],
    ];
    WHEEL[index % WHEEL.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    // How many point lights the field holds.
    const fn point_light_count() -> usize {
        LIGHTS[0] * LIGHTS[1] * LIGHTS[2]
    }

    #[test]
    fn the_field_holds_what_its_dimensions_say() {
        assert_eq!(point_light_count(), 60);
    }

    // Neighbouring lights have to overlap, or the clusters between them hold
    // nothing and the culling path is never asked a hard question.
    #[test]
    fn the_bulbs_reach_past_their_neighbours() {
        for step in LIGHT_SPACING {
            assert!(LIGHT_RANGE > step, "a bulb stops short of the next one");
        }
    }
}
