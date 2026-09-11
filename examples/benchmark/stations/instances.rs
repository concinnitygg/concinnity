//! A lattice of instanced spheres: one declaration, one mesh, and several
//! hundred copies of it tested for visibility and shaded every frame.

use concinnity::components::{InstanceTransform, InstancedProp, ProceduralMesh};
use concinnity::cook::WorldBuilder;

use crate::palette;
use crate::stations::spread;

// How many spheres along each axis, and how far apart.
const LATTICE: [usize; 3] = [10, 4, 10];
const SPACING: f32 = 1.7;
const BASE_HEIGHT: f32 = 1.4;
const RADIUS: f32 = 0.55;

// Far enough that the whole lattice is drawn from every point on the circle the
// camera goes round, and dropped once the camera has moved on.
const CULL_DISTANCE: f32 = 90.0;

/// How many instances the lattice holds.
pub(crate) const fn count() -> usize {
    LATTICE[0] * LATTICE[1] * LATTICE[2]
}

/// Declare the lattice.
pub(crate) fn declare(world: &mut WorldBuilder, center: [f32; 3]) {
    world.add(
        "instances_mesh",
        ProceduralMesh {
            generator: "sphere".to_string(),
            radius: Some(RADIUS),
            rings: Some(12),
            segments: Some(16),
            ..Default::default()
        },
    );

    let mut instances = Vec::with_capacity(count());
    for x in 0..LATTICE[0] {
        for y in 0..LATTICE[1] {
            for z in 0..LATTICE[2] {
                instances.push(InstanceTransform {
                    position: [
                        center[0] + spread(x, LATTICE[0], SPACING),
                        BASE_HEIGHT + y as f32 * SPACING,
                        center[2] + spread(z, LATTICE[2], SPACING),
                    ],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                });
            }
        }
    }
    world
        .add(
            "instances_lattice",
            InstancedProp {
                instances,
                cull_distance: CULL_DISTANCE,
                ..Default::default()
            },
        )
        .reference("mesh", "instances_mesh")
        .reference("material", palette::METAL);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lattice_holds_what_its_dimensions_say() {
        assert_eq!(count(), 400);
    }
}
