//! A pool at the end of the corridor: a tessellated, animated surface that
//! reflects everything above it, and the last thing the camera looks at.

use concinnity::components::{ProceduralMesh, Prop, WaterSurface, WaterWave};
use concinnity::cook::WorldBuilder;

use crate::palette;
use crate::stations::spread;

// The pool's extent and how finely it is tessellated. The subdivision count is
// the station's vertex load; the reflection is its fill. It reaches almost to
// the circle the camera goes round, so the surface fills the frame from every
// point on it.
const EXTENT: [f32; 2] = [30.0, 26.0];
const SUBDIVISIONS: u32 = 180;

// The colonnade standing in the pool, which is what the surface has to reflect.
const COLUMNS: usize = 9;
const COLUMN_SPACING: [f32; 2] = [11.0, 11.0];
const COLUMN_RADIUS: f32 = 0.9;
const COLUMN_HEIGHT: f32 = 12.0;

/// Declare the pool and what stands in it.
pub(crate) fn declare(world: &mut WorldBuilder, centre: [f32; 3]) {
    world.add(
        "water_column_mesh",
        ProceduralMesh {
            generator: "cylinder".to_string(),
            radius: Some(COLUMN_RADIUS),
            height: Some(COLUMN_HEIGHT),
            segments: Some(28),
            ..Default::default()
        },
    );
    for index in 0..COLUMNS {
        let across = spread(index % 3, 3, COLUMN_SPACING[0]);
        let along = spread(index / 3, 3, COLUMN_SPACING[1]);
        world
            .add(
                format!("water_column_{index}"),
                Prop {
                    position: [centre[0] + across, COLUMN_HEIGHT * 0.5, centre[2] + along],
                    ..Default::default()
                },
            )
            .reference("mesh", "water_column_mesh")
            .reference("material", palette::STONE);
    }

    world.add(
        "water_pool",
        WaterSurface {
            centre,
            extent: EXTENT,
            subdivisions: SUBDIVISIONS,
            waves: vec![
                WaterWave {
                    amplitude: 0.09,
                    wavelength: 4.2,
                    speed: 0.5,
                    direction: [1.0, 0.25],
                    steepness: 0.2,
                },
                WaterWave {
                    amplitude: 0.05,
                    wavelength: 2.3,
                    speed: 0.35,
                    direction: [-0.4, 1.0],
                    steepness: 0.2,
                },
            ],
            deep_colour: [0.01, 0.04, 0.07],
            shallow_colour: [0.05, 0.16, 0.20],
            depth_falloff_metres: 2.0,
            foam_width_metres: 0.6,
            foam_intensity: 0.4,
            fresnel_power: 1.2,
            roughness: 0.08,
            refraction_strength: 0.03,
            visible: true,
            ..Default::default()
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // The columns have to stand inside the pool, or nothing is reflected and
    // the station measures a flat mirror of the sky.
    #[test]
    fn the_colonnade_stands_in_the_water() {
        assert!(
            COLUMN_SPACING[0] < EXTENT[0],
            "the columns fall off the sides"
        );
        assert!(
            COLUMN_SPACING[1] < EXTENT[1],
            "the columns fall off the ends"
        );
    }
}
