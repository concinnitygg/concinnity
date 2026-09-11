//! A rack of glass over a marked floor: the station that holds the world's
//! see-through surfaces, its refraction and its decals.
//!
//! The rack's slabs overlap along the view, so a pixel at the back of it is
//! shaded once per slab in front of it, and the floor under them is covered in
//! projected marks. It is the lightest station in the corridor, which is what
//! makes it the one to read the others against.
//!
//! The refracting pane beside the rack is a single one. A pane is a reflector
//! plane, and a reflector plane re-renders the scene from its own side of the
//! glass wherever the camera is, so a second one would cost every other segment
//! as much as this one.

use concinnity::components::{Decal, GlassPanel, ProceduralMesh, Prop};
use concinnity::cook::WorldBuilder;

use crate::palette;
use crate::stations::spread;

// The floor the panels stand on and the decals are projected onto.
const FLOOR_HALF_EXTENT: f32 = 14.0;

// The refracting pane: how big it is and how high it stands.
const PANEL_HALF_SIZE: [f32; 2] = [5.5, 4.0];
const PANEL_HEIGHT: f32 = 4.2;

// The marks on the floor under them.
const DECALS: [usize; 2] = [7, 7];
const DECAL_SPACING: f32 = 3.4;
const DECAL_SIZE: [f32; 3] = [2.6, 1.4, 2.6];

// The blocks behind the glass, which are what a refracted pixel resolves to.
const BLOCKS: usize = 10;

// The rack of see-through slabs the camera looks through end to end. They
// overlap along the view, so a pixel at the back of the rack is shaded once per
// slab in front of it and the transparent pass is what the station costs.
const SLABS: usize = 16;
const SLAB_SPACING: f32 = 0.55;
const SLAB_HALF_EXTENTS: [f32; 3] = [4.5, 3.2, 0.06];
const SLAB_HEIGHT: f32 = 3.6;

/// Declare the floor, the panels, the marks, and what shows through.
pub(crate) fn declare(world: &mut WorldBuilder, center: [f32; 3]) {
    world.add(
        "glass_floor_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some([FLOOR_HALF_EXTENT, 0.2, FLOOR_HALF_EXTENT]),
            ..Default::default()
        },
    );
    world
        .add(
            "glass_floor",
            Prop {
                position: [center[0], 0.2, center[2]],
                ..Default::default()
            },
        )
        .reference("mesh", "glass_floor_mesh")
        .reference("material", palette::TILE);

    world.add(
        "glass_block_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some([1.1, 2.6, 1.1]),
            ..Default::default()
        },
    );
    for index in 0..BLOCKS {
        world
            .add(
                format!("glass_block_{index}"),
                Prop {
                    position: [
                        center[0] + spread(index % 5, 5, 4.2),
                        2.8,
                        center[2] + spread(index / 5, 2, 6.0) - 8.0,
                    ],
                    rotation_deg: [0.0, index as f32 * 17.0, 0.0],
                    ..Default::default()
                },
            )
            .reference("mesh", "glass_block_mesh")
            .reference("material", palette::BRICK);
    }

    // The pane faces across the corridor, so the camera looks through it rather
    // than along its edge.
    world.add(
        "glass_pane",
        GlassPanel {
            center: [center[0], PANEL_HEIGHT, center[2] - 5.0],
            normal: [1.0, 0.0, 0.15],
            half_size: PANEL_HALF_SIZE,
            tint: [0.72, 0.86, 0.92],
            opacity: 0.28,
            refraction_strength: 0.14,
            fresnel_power: 3.0,
            visible: true,
            ..Default::default()
        },
    );

    world.add(
        "glass_slab_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some(SLAB_HALF_EXTENTS),
            ..Default::default()
        },
    );
    for index in 0..SLABS {
        world
            .add(
                format!("glass_slab_{index}"),
                Prop {
                    position: [
                        center[0] + 3.0,
                        SLAB_HEIGHT,
                        center[2] + spread(index, SLABS, SLAB_SPACING),
                    ],
                    rotation_deg: [0.0, 90.0, 0.0],
                    ..Default::default()
                },
            )
            .reference("mesh", "glass_slab_mesh")
            .reference("material", palette::PANE);
    }

    for x in 0..DECALS[0] {
        for z in 0..DECALS[1] {
            let index = x * DECALS[1] + z;
            world
                .add(
                    format!("glass_mark_{index}"),
                    Decal {
                        position: [
                            center[0] + spread(x, DECALS[0], DECAL_SPACING),
                            0.6,
                            center[2] + spread(z, DECALS[1], DECAL_SPACING),
                        ],
                        rotation_deg: [0.0, index as f32 * 23.0, 0.0],
                        size: DECAL_SIZE,
                        tint: [0.35, 0.40, 0.46, 0.75],
                        visible: true,
                        ..Default::default()
                    },
                )
                .reference("texture", palette::SCUFF);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The slabs are what makes the transparent pass expensive, and they only do
    // that where they overlap: a rack spread wider than one slab is a row of
    // separate windows, each shading its pixels once.
    #[test]
    fn the_rack_overlaps_itself_along_the_view() {
        assert!(SLAB_HALF_EXTENTS[0] * 2.0 > SLAB_SPACING * SLABS as f32);
    }

    #[test]
    fn the_marks_cover_the_floor_the_camera_sees() {
        let span = DECAL_SPACING * (DECALS[0] - 1) as f32;
        assert!(span < FLOOR_HALF_EXTENT * 2.0, "marks fall off the floor");
    }
}
