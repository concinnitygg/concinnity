//! The textures and materials the stations draw from, declared once so a
//! surface anywhere in the corridor samples the same maps.
//!
//! Every texture here comes from a built-in generator, so the world stands up
//! with nothing under `assets/`.

use concinnity::cook::{Material, Texture, WorldBuilder};

/// The corridor floor: rough, unlit-looking concrete.
pub(crate) const GROUND: &str = "mat_ground";
/// Dressed stone, the default for structure.
pub(crate) const STONE: &str = "mat_stone";
/// Painted brick, for anything that should read warm.
pub(crate) const BRICK: &str = "mat_brick";
/// Polished metal, the most reflective surface in the world.
pub(crate) const METAL: &str = "mat_metal";
/// Glazed tile: smooth, bright, and cheap to light.
pub(crate) const TILE: &str = "mat_tile";
/// Varnished wood.
pub(crate) const WOOD: &str = "mat_wood";
/// Ground cover, for the parts of the corridor that are not floor.
pub(crate) const GRASS: &str = "mat_grass";
/// A surface that emits rather than reflects, patterned by its emissive map.
pub(crate) const GLOW: &str = "mat_glow";
/// Glass: what is behind it shows through, so it draws in the transparent pass
/// and the pixels under it are shaded more than once.
pub(crate) const PANE: &str = "mat_pane";

/// The texture a particle sprite samples.
pub(crate) const SOOT: &str = "tex_plaster";
/// The texture a decal projects.
pub(crate) const SCUFF: &str = "tex_checker";

// Color maps are generated at this edge length. Large enough that a surface
// filling the frame is not obviously a small tile, small enough that the eight
// of them generate in well under the opening hold.
const MAP_RESOLUTION: u32 = 512;
// The two maps read as patterns rather than as surfaces, and are sampled at a
// distance, so they cost half the edge.
const PATTERN_RESOLUTION: u32 = 256;

/// Declare every shared texture and material.
pub(crate) fn declare(world: &mut WorldBuilder) {
    for (name, generator) in [
        ("tex_concrete", "concrete"),
        ("tex_stone", "stone"),
        ("tex_brick", "brick"),
        ("tex_metal", "metal"),
        ("tex_tile", "tile"),
        ("tex_wood", "wood"),
        ("tex_grass", "grass"),
    ] {
        world.add(name, color_map(generator, MAP_RESOLUTION));
    }
    world.add(SOOT, color_map("plaster", PATTERN_RESOLUTION));
    world.add(SCUFF, color_map("checker", PATTERN_RESOLUTION));

    // The concrete map is bound a second time as occlusion-roughness-metallic,
    // so the floor's gloss varies across it the way a worn surface does and the
    // main pass samples two maps rather than one.
    world
        .add(
            GROUND,
            Material {
                roughness: 0.9,
                metallic: 0.0,
                tint: [0.55, 0.56, 0.58],
                ..Default::default()
            },
        )
        .reference("albedo", "tex_concrete")
        .reference("orm_map", "tex_concrete");

    for (name, texture, roughness, metallic, tint) in [
        (STONE, "tex_stone", 0.78, 0.0, [0.62, 0.63, 0.66]),
        (BRICK, "tex_brick", 0.72, 0.0, [0.85, 0.55, 0.45]),
        (METAL, "tex_metal", 0.17, 0.9, [0.82, 0.84, 0.88]),
        (TILE, "tex_tile", 0.24, 0.05, [0.75, 0.80, 0.84]),
        (WOOD, "tex_wood", 0.58, 0.0, [0.72, 0.55, 0.36]),
        (GRASS, "tex_grass", 0.95, 0.0, [0.48, 0.62, 0.36]),
    ] {
        world
            .add(
                name,
                Material {
                    roughness,
                    metallic,
                    tint,
                    ..Default::default()
                },
            )
            .reference("albedo", texture);
    }

    world
        .add(
            PANE,
            Material {
                roughness: 0.08,
                metallic: 0.0,
                tint: [0.70, 0.86, 0.90],
                opacity: 0.34,
                see_through: true,
                ..Default::default()
            },
        )
        .reference("albedo", "tex_tile");

    world
        .add(
            GLOW,
            Material {
                roughness: 0.4,
                tint: [0.1, 0.1, 0.12],
                emissive_factor: [3.0, 2.4, 1.6],
                ..Default::default()
            },
        )
        .reference("emissive_map", SCUFF);
}

fn color_map(generator: &str, resolution: u32) -> Texture {
    Texture {
        generator: generator.to_string(),
        resolution,
        ..Default::default()
    }
}
