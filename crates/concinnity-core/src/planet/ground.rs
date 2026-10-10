// The patch of a planet's ground around one point, as a height grid in the
// simulated frame: what bodies near that point collide with.

use alloc::vec::Vec;

use super::dvec;
use super::frame::LocalFrame;
use super::shape::PlanetShape;
use crate::terrain::TerrainGrid;

/// A square of ground in the simulated frame: `grid`'s heights are meters
/// above `center`'s height, its corners spread around `center` along the
/// frame's `X` and `Z`.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundPatch {
    /// The patch's center in the simulated frame; its height is the grid's
    /// base.
    pub center: [f32; 3],
    /// The heights, one per corner.
    pub grid: TerrainGrid,
}

// Refinements of each corner's height. Each lands the corner on the sphere
// through the last estimate; the ground's height barely changes over the
// step, so the third is exact to well under a millimeter.
const REFINEMENTS: usize = 3;

/// The ground of `shape` in `frame` around local point `around`: `cells` cells
/// along each side of a square `2 * half_width` across.
///
/// Each corner's height is where the frame's vertical line through it meets
/// the ground. The grid triangulates the ground differently from the render
/// tiles, which differ from it by the ground's curvature within one cell.
pub fn ground_patch(
    shape: &PlanetShape,
    frame: &LocalFrame,
    around: [f32; 3],
    cells: u32,
    half_width: f32,
) -> Option<GroundPatch> {
    let side = cells as usize + 1;
    let center_local = frame.to_local_f64(shape.center);
    let x0 = f64::from(around[0]) - f64::from(half_width);
    let z0 = f64::from(around[2]) - f64::from(half_width);
    let step = 2.0 * f64::from(half_width) / f64::from(cells);
    let mut heights = Vec::with_capacity(side * side);
    for row in 0..side {
        for col in 0..side {
            let x = x0 + col as f64 * step;
            let z = z0 + row as f64 * step;
            heights.push(ground_y(shape, frame, center_local, x, z)?);
        }
    }
    let base = heights.iter().copied().fold(f64::INFINITY, f64::min);
    let grid = TerrainGrid::new(
        cells,
        [half_width; 2],
        heights.iter().map(|h| (h - base) as f32).collect(),
    )
    .ok()?;
    Some(GroundPatch {
        center: [around[0], base as f32, around[2]],
        grid,
    })
}

// The local height of the ground under local column (x, z), or `None` when
// the column misses the planet.
fn ground_y(
    shape: &PlanetShape,
    frame: &LocalFrame,
    center: dvec::DVec3,
    x: f64,
    z: f64,
) -> Option<f64> {
    let (dx, dz) = (x - center[0], z - center[2]);
    let mut radius = shape.radius + shape.amplitude * 0.5;
    let mut y = 0.0;
    for _ in 0..REFINEMENTS {
        let under = radius * radius - dx * dx - dz * dz;
        if under <= 0.0 {
            return None;
        }
        y = center[1] + libm::sqrt(under);
        let world = frame.to_world([x as f32, y as f32, z as f32]);
        let dir = shape.up_at(world);
        radius = shape.radius + shape.height(dir);
    }
    Some(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 30.0,
            feature_size: 1_500.0,
            octaves: 6,
            seed: 9,
        }
    }

    // Every corner of the patch lies on the ground: its distance from the
    // center is the radius plus the height the shape gives that direction.
    #[test]
    fn every_corner_lies_on_the_ground() {
        let s = shape();
        let frame = LocalFrame::AUTHORED;
        let patch = ground_patch(&s, &frame, [120.0, 0.0, -40.0], 16, 32.0).unwrap();
        let side = patch.grid.side();
        let [cx, cy, cz] = patch.center;
        for row in 0..side {
            for col in 0..side {
                let x = cx - 32.0 + col as f32 * 4.0;
                let z = cz - 32.0 + row as f32 * 4.0;
                let y = cy + patch.grid.heights()[row * side + col];
                let world = frame.to_world([x, y, z]);
                let r = dvec::length(dvec::sub(world, s.center));
                let expected = s.radius + s.height(s.up_at(world));
                assert!((r - expected).abs() < 0.01, "{r} vs {expected}");
            }
        }
    }

    // A kilometer from the frame's origin the ground has dropped by the
    // curvature, about d^2 / 2r.
    #[test]
    fn the_patch_follows_the_curvature() {
        let flat = PlanetShape {
            amplitude: 0.0,
            ..shape()
        };
        let frame = LocalFrame::AUTHORED;
        let patch = ground_patch(&flat, &frame, [1_000.0, 0.0, 0.0], 4, 8.0).unwrap();
        let drop = 1_000.0_f32 * 1_000.0 / (2.0 * 50_000.0);
        let mid = patch.center[1] + patch.grid.heights()[2 * 5 + 2];
        assert!((mid + drop).abs() < 0.05, "{mid}");
    }

    #[test]
    fn a_patch_off_the_planet_is_refused() {
        let s = shape();
        assert!(ground_patch(&s, &LocalFrame::AUTHORED, [60_000.0, 0.0, 0.0], 4, 8.0).is_none());
    }
}
