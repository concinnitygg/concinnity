//! A grove on broken ground under a rank of casting spot lights: the densest
//! stand of shadow casters in the corridor.
//!
//! Every caster here is rasterised once for the frame, once into each of the
//! sun's cascades, and once into each spot's slice. The cascades follow the
//! camera, so whatever that costs is spent while the camera is here. The ground
//! under the grove is a displaced grid rather than a plane, which is the
//! station's vertex load and the surface every shadow lands on.

use concinnity::components::{ProceduralMesh, Prop, SpotLight};
use concinnity::cook::WorldBuilder;

use crate::palette;
use crate::stations::spread;

// The ground under the grove: a dense displaced grid, which is also the
// station's vertex load.
const TERRAIN_HALF_EXTENT: f32 = 22.0;
const TERRAIN_SUBDIVISIONS: u32 = 130;
const TERRAIN_AMPLITUDE: f32 = 1.6;

// The grove: how many trees across and deep, and how far apart.
const GROVE: [usize; 2] = [13, 8];
const GROVE_SPACING: f32 = 2.7;

const TRUNK_RADIUS: f32 = 0.28;
const TRUNK_HEIGHT: f32 = 5.5;
const CANOPY_RADIUS: f32 = 1.9;

// A crown is an ellipsoid rather than a ball, so no two throw the same shadow.
const CANOPY_SCALE: [f32; 3] = [1.35, 0.68, 0.82];

// The lights over it. Each casts, so each rasterises the grove into a slice of
// the spot shadow array every frame. The array holds sixteen slices; a light
// past the last one is lit but throws nothing, so the count stays under it.
const SPOTS: usize = 8;
const SPOT_HEIGHT: f32 = 13.0;

/// Declare the grove, the ground under it, and the lights over it.
pub(crate) fn declare(world: &mut WorldBuilder, centre: [f32; 3]) {
    world.add(
        "shadows_terrain_mesh",
        ProceduralMesh {
            generator: "terrain".to_string(),
            half_width: TERRAIN_HALF_EXTENT,
            half_depth: TERRAIN_HALF_EXTENT,
            subdivisions: Some(TERRAIN_SUBDIVISIONS),
            amplitude: Some(TERRAIN_AMPLITUDE),
            ..Default::default()
        },
    );
    world
        .add(
            "shadows_terrain",
            Prop {
                position: [centre[0], 0.05, centre[2]],
                ..Default::default()
            },
        )
        .reference("mesh", "shadows_terrain_mesh")
        .reference("material", palette::GRASS);

    world.add(
        "shadows_trunk_mesh",
        ProceduralMesh {
            generator: "cylinder".to_string(),
            radius: Some(TRUNK_RADIUS),
            height: Some(TRUNK_HEIGHT),
            segments: Some(32),
            ..Default::default()
        },
    );
    world.add(
        "shadows_canopy_mesh",
        ProceduralMesh {
            generator: "sphere".to_string(),
            radius: Some(CANOPY_RADIUS),
            rings: Some(24),
            segments: Some(32),
            ..Default::default()
        },
    );

    for x in 0..GROVE[0] {
        for z in 0..GROVE[1] {
            let index = x * GROVE[1] + z;
            let at = [
                centre[0] + spread(x, GROVE[0], GROVE_SPACING),
                centre[2] + spread(z, GROVE[1], GROVE_SPACING),
            ];
            // A little lean per tree, so no two casters throw the same shadow.
            let lean = (index % 7) as f32 - 3.0;
            world
                .add(
                    format!("shadows_trunk_{index}"),
                    Prop {
                        position: [at[0], TRUNK_HEIGHT * 0.5, at[1]],
                        rotation_deg: [lean, index as f32 * 11.0, 0.0],
                        ..Default::default()
                    },
                )
                .reference("mesh", "shadows_trunk_mesh")
                .reference("material", palette::WOOD);
            // The crown hangs off its own trunk, which is what lets one rule
            // below reach every crown and no other placement in the corridor.
            world
                .add(
                    format!("shadows_canopy_{index}"),
                    Prop {
                        position: [0.0, TRUNK_HEIGHT * 0.5 + CANOPY_RADIUS * 0.4, 0.0],
                        scale: CANOPY_SCALE,
                        ..Default::default()
                    },
                )
                .reference("mesh", "shadows_canopy_mesh")
                .reference("material", palette::GRASS)
                .reference("parent", format!("shadows_trunk_{index}"));
        }
    }

    for index in 0..SPOTS {
        let along = spread(index, SPOTS, GROVE_SPACING * 2.6);
        world.add(
            format!("shadows_spot_{index}"),
            SpotLight {
                position: [centre[0] + along, SPOT_HEIGHT, centre[2] + along * 0.3],
                direction: [0.0, -1.0, 0.0],
                color: [1.0, 0.94 - index as f32 * 0.03, 0.80],
                intensity: 140.0,
                range: 34.0,
                inner_angle: 20.0,
                outer_angle: 36.0,
                cast_shadows: true,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // How many props the grove stands up.
    const fn count() -> usize {
        GROVE[0] * GROVE[1] * 2
    }

    #[test]
    fn the_grove_stands_up_a_trunk_and_a_canopy_per_tree() {
        assert_eq!(count(), 208);
    }

    // Two crowns of the same shape at the same height throw the same shadow;
    // an ellipsoid turned by its trunk's own heading does not.
    #[test]
    fn a_crown_is_not_a_sphere() {
        assert!(
            CANOPY_SCALE[0] != CANOPY_SCALE[2],
            "the silhouette is round"
        );
    }
}
