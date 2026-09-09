//! The corridor itself: the floor every station stands on, the sun, and the
//! sky that turns over both.

use concinnity::components::{DirectionalLight, ProceduralMesh, Prop, SkyRotation};
use concinnity::cook::{EnvironmentMap, WorldBuilder};

use crate::palette;

// How far past the ends of the path the floor reaches, and how far to either
// side of the centre line, so its edge never enters frame.
const MARGIN: f32 = 45.0;
const HALF_WIDTH: f32 = 70.0;

// How fast the celestial sphere turns. Slow enough that the sun crosses a
// useful arc over a run without the shadows visibly sweeping within a segment.
const SKY_DEGREES_PER_SECOND: f32 = 1.6;

/// Declare the floor, the key light, and the sky.
pub(crate) fn declare(world: &mut WorldBuilder) {
    let end = crate::stations::corridor_end();
    world.add(
        "ground_mesh",
        ProceduralMesh {
            generator: "plane".to_string(),
            half_width: HALF_WIDTH,
            half_depth: (crate::stations::START_Z - end) * 0.5 + MARGIN,
            ..Default::default()
        },
    );
    world
        .add(
            "ground",
            Prop {
                position: [0.0, 0.0, (crate::stations::START_Z + end) * 0.5],
                ..Default::default()
            },
        )
        .reference("mesh", "ground_mesh")
        .reference("material", palette::GROUND);

    world.add(
        "sun",
        DirectionalLight {
            direction: [0.35, 0.80, 0.45],
            color: [1.0, 0.96, 0.88],
            intensity: 3.2,
        },
    );

    // The scene's ambient fill, convolved once at world compile. Small faces on
    // purpose: this is the lighting, not the subject, and the convolution is
    // the slowest thing in the compile.
    world.add(
        "sky",
        EnvironmentMap {
            generator: "sky".to_string(),
            prefilter_face_size: 256,
            irradiance_face_size: 32,
            prefilter_samples: 64,
            ..Default::default()
        },
    );

    // The sky advances on the fixed simulation clock, like the camera track, so
    // two runs light the same station from the same angle.
    world.add(
        "sky_spin",
        SkyRotation {
            axis: [1.0, 0.0, 0.15],
            degrees_per_second: SKY_DEGREES_PER_SECOND,
            ..Default::default()
        },
    );
}
