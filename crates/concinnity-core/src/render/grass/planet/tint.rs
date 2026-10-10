//! A planet's far field: past where the blades end, every tile's ground blends
//! toward the color its grass shows from afar, thinned by the same meadow the
//! blades are. Each tile vertex carries that color premultiplied by how much
//! grass covers it, exactly as a terrain's grid corners do (see
//! [`crate::render::grass::tint`]).

use alloc::vec::Vec;

use super::PlanetGrassLayer;
use super::density::meadow_density;
use crate::planet::dvec;
use crate::planet::{PlanetShape, TileMesh};
use crate::render::grass::tint::{FAR_FIELD_FULL_COVER_DENSITY, far_field_color, luminance};

/// The color a planet's ground shows from afar, wherever on it.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanetFarField {
    // Each layer's far color, how much of the ground it covers where it grows
    // fully, and its meadow seed.
    layers: Vec<([f32; 3], f32, u32)>,
    scale: f32,
    /// The luminance a fully covered vertex's color has.
    pub luma: f32,
}

impl PlanetFarField {
    /// The far field of `layers`, whose meadow noise has `scale` lattice
    /// cells per unit of direction; `None` when no layer covers anything.
    pub fn new(layers: &[PlanetGrassLayer], scale: f32) -> Option<Self> {
        let layers: Vec<_> = layers
            .iter()
            .map(|l| {
                let cover = (l.density / FAR_FIELD_FULL_COVER_DENSITY).min(1.0);
                (far_field_color(&l.look), cover, l.seed)
            })
            .filter(|(_, cover, _)| *cover > 0.0)
            .collect();
        let weight: f32 = layers.iter().map(|(_, c, _)| c).sum();
        if weight <= 0.0 {
            return None;
        }
        let luma = layers
            .iter()
            .map(|(color, c, _)| c * luminance(*color))
            .sum::<f32>()
            / weight;
        Some(Self {
            layers,
            scale,
            luma,
        })
    }

    /// The premultiplied far color in unit direction `dir` from the planet's
    /// center.
    pub fn color_at(&self, dir: [f32; 3]) -> [f32; 3] {
        let mut bare = 1.0f32;
        let mut sum = [0.0f32; 3];
        let mut total = 0.0f32;
        for (color, cover, seed) in &self.layers {
            let c = cover * meadow_density(dir, self.scale, *seed);
            bare *= 1.0 - c;
            total += c;
            for (s, x) in sum.iter_mut().zip(color) {
                *s += c * x;
            }
        }
        if total <= 0.0 {
            return [0.0; 3];
        }
        let covered = 1.0 - bare;
        sum.map(|x| x / total * covered)
    }

    /// Color every vertex of `mesh`, a tile of `shape`, with the far field
    /// over it.
    pub fn tint(&self, shape: &PlanetShape, mesh: &mut TileMesh) {
        for v in &mut mesh.vertices {
            let at = dvec::add(mesh.origin, dvec::from_f32(v.pos));
            v.color = self.color_at(dvec::to_f32(shape.up_at(at)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Grass;
    use crate::planet::{CubeFace, TileId, tile_mesh};
    use crate::render::grass::GrassBladeLook;
    use crate::render::grass::tint::luminance;

    fn layer(density: f32, seed: u32) -> PlanetGrassLayer {
        PlanetGrassLayer {
            look: GrassBladeLook::of(&Grass::default()),
            density,
            cells: 30,
            cell_size: 0.1,
            seed,
        }
    }

    #[test]
    fn bare_meadow_shows_no_color_and_full_meadow_the_reference() {
        let far = PlanetFarField::new(&[layer(100.0, 5)], 2000.0).unwrap();
        let mut saw_bare = false;
        let mut saw_full = false;
        for i in 0..2000 {
            let a = i as f32 * 0.0011;
            let dir = crate::math::vec3::normalize_or(
                [a.sin(), 1.0, a.cos() * 0.3],
                1e-9,
                [0.0, 1.0, 0.0],
            );
            let c = far.color_at(dir);
            let d = meadow_density(dir, 2000.0, 5);
            let coverage = luminance(c) / far.luma;
            assert!((coverage - d).abs() < 1e-4, "{coverage} {d}");
            saw_bare |= d < 0.01;
            saw_full |= d > 0.99;
        }
        assert!(saw_bare && saw_full);
        assert_eq!(PlanetFarField::new(&[layer(0.0, 1)], 2000.0), None);
    }

    #[test]
    fn a_tinted_tile_carries_the_field_at_each_vertex() {
        let shape = PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 6,
            seed: 1,
        };
        let far = PlanetFarField::new(&[layer(80.0, 9), layer(20.0, 3)], 2000.0).unwrap();
        let tile = TileId {
            face: CubeFace(2),
            level: 10,
            x: 500,
            y: 520,
        };
        let mut mesh = tile_mesh(&shape, tile);
        far.tint(&shape, &mut mesh);
        let v = mesh.vertices[40];
        let at = dvec::add(mesh.origin, dvec::from_f32(v.pos));
        assert_eq!(v.color, far.color_at(dvec::to_f32(shape.up_at(at))));
    }
}
