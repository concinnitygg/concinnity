//! Where the grass kernel looks for blades: the world-aligned tile grid around
//! the camera, and the jittered cell grid inside each tile that holds one blade
//! per cell.
//!
//! Cells are counted from the world origin, so a blade's place depends only on
//! its cell and not on which tiles the camera currently covers.

use crate::math::{ceil, floor, round, sqrt};

/// Edge of one grass tile, in meters: the unit the kernel frustum- and
/// occlusion-culls.
pub const GRASS_TILE_SIZE: f32 = 4.0;

/// Distance from the camera past which no blade is drawn, in meters.
pub const GRASS_DRAW_DISTANCE: f32 = 90.0;

/// Highest density the grid places, in blades per square meter.
pub const MAX_GRASS_DENSITY: f32 = 1000.0;

/// Most blades the visible-blade buffer holds, whatever the field asks for.
pub const MAX_GRASS_BLADES: u32 = 1 << 21;

/// Threads per kernel group: each thread places one candidate blade.
pub const GRASS_GROUP_SIZE: u32 = 64;

/// The cell grid inside every tile: `cells_per_side` squared cells, one blade
/// each.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassGrid {
    /// Cells along each tile edge.
    pub cells_per_side: u32,
}

impl GrassGrid {
    /// The grid closest to `density` blades per square meter, or `None` when
    /// the density places no blade at all.
    pub fn for_density(density: f32) -> Option<Self> {
        if density.is_nan() || density <= 0.0 {
            return None;
        }
        let density = density.min(MAX_GRASS_DENSITY);
        let cells = round(GRASS_TILE_SIZE * sqrt(density)).max(1.0);
        Some(Self {
            cells_per_side: cells as u32,
        })
    }

    /// Edge of one cell, in meters.
    pub fn cell_size(&self) -> f32 {
        GRASS_TILE_SIZE / self.cells_per_side as f32
    }

    /// Candidate blades per tile.
    pub fn blades_per_tile(&self) -> u32 {
        self.cells_per_side * self.cells_per_side
    }

    /// The density the grid actually places, in blades per square meter.
    pub fn density(&self) -> f32 {
        self.blades_per_tile() as f32 / (GRASS_TILE_SIZE * GRASS_TILE_SIZE)
    }

    /// Kernel groups across one tile's candidates.
    pub fn groups_per_tile(&self) -> u32 {
        self.blades_per_tile().div_ceil(GRASS_GROUP_SIZE)
    }
}

/// An axis-aligned rectangle on the ground plane, `[x, z]` corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundRect {
    /// Minimum corner.
    pub min: [f32; 2],
    /// Maximum corner.
    pub max: [f32; 2],
}

impl GroundRect {
    /// The rectangle centered on `center` with half-extents `extent`.
    pub fn centered(center: [f32; 2], extent: [f32; 2]) -> Self {
        Self {
            min: [center[0] - extent[0], center[1] - extent[1]],
            max: [center[0] + extent[0], center[1] + extent[1]],
        }
    }

    /// Area in square meters; 0 for an empty rectangle.
    pub fn area(&self) -> f32 {
        (self.max[0] - self.min[0]).max(0.0) * (self.max[1] - self.min[1]).max(0.0)
    }

    fn intersect(&self, other: &Self) -> Self {
        Self {
            min: [self.min[0].max(other.min[0]), self.min[1].max(other.min[1])],
            max: [self.max[0].min(other.max[0]), self.max[1].min(other.max[1])],
        }
    }
}

/// A block of tiles: the tile at `origin` (in tile coordinates) and `count`
/// tiles along `+x` and `+z` from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRange {
    /// Tile coordinates of the first tile.
    pub origin: [i32; 2],
    /// Tiles along each axis; zero when nothing is in range.
    pub count: [u32; 2],
}

impl TileRange {
    /// No tiles.
    pub const EMPTY: Self = Self {
        origin: [0, 0],
        count: [0, 0],
    };

    /// Number of tiles in the block.
    pub fn len(&self) -> u32 {
        self.count[0] * self.count[1]
    }

    /// True when the block holds no tile.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The tiles that can hold a drawn blade: those overlapping both `ground` and
/// the square of half-width `distance` around the camera's `cam_xz`.
pub fn tiles_in_reach(ground: &GroundRect, cam_xz: [f32; 2], distance: f32) -> TileRange {
    let reach = GroundRect::centered(cam_xz, [distance, distance]);
    let area = ground.intersect(&reach);
    if area.area() <= 0.0 {
        return TileRange::EMPTY;
    }
    let lo = |v: f32| floor(v / GRASS_TILE_SIZE) as i32;
    // The last tile is the one holding the far edge, exclusive of a tile that
    // only touches it.
    let hi = |v: f32| ceil(v / GRASS_TILE_SIZE) as i32 - 1;
    let origin = [lo(area.min[0]), lo(area.min[1])];
    let last = [hi(area.max[0]), hi(area.max[1])];
    TileRange {
        origin,
        count: [
            (last[0] - origin[0] + 1).max(0) as u32,
            (last[1] - origin[1] + 1).max(0) as u32,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_rounds_density_to_whole_cells_per_tile() {
        let grid = GrassGrid::for_density(100.0).unwrap();
        assert_eq!(grid.cells_per_side, 40);
        assert_eq!(grid.blades_per_tile(), 1600);
        assert_eq!(grid.density(), 100.0);
        assert_eq!(grid.cell_size(), 0.1);
        assert_eq!(grid.groups_per_tile(), 25);
    }

    #[test]
    fn a_sparse_field_keeps_one_blade_per_tile() {
        let grid = GrassGrid::for_density(0.001).unwrap();
        assert_eq!(grid.cells_per_side, 1);
        assert_eq!(grid.groups_per_tile(), 1);
    }

    #[test]
    fn no_density_places_nothing_and_density_is_capped() {
        assert_eq!(GrassGrid::for_density(0.0), None);
        assert_eq!(GrassGrid::for_density(-3.0), None);
        assert_eq!(GrassGrid::for_density(f32::NAN), None);
        let capped = GrassGrid::for_density(1.0e9).unwrap();
        assert_eq!(capped, GrassGrid::for_density(MAX_GRASS_DENSITY).unwrap());
    }

    #[test]
    fn reach_clips_the_ground_to_the_camera_square() {
        let ground = GroundRect::centered([0.0, 0.0], [100.0, 100.0]);
        let range = tiles_in_reach(&ground, [0.0, 0.0], 8.0);
        // [-8, 8) on both axes is tiles -2..=1.
        assert_eq!(range.origin, [-2, -2]);
        assert_eq!(range.count, [4, 4]);
        assert_eq!(range.len(), 16);
    }

    #[test]
    fn reach_keeps_a_ground_smaller_than_the_camera_square() {
        let ground = GroundRect::centered([2.0, 2.0], [1.0, 1.0]);
        let range = tiles_in_reach(&ground, [0.0, 0.0], 60.0);
        assert_eq!(range.origin, [0, 0]);
        assert_eq!(range.count, [1, 1]);
    }

    #[test]
    fn a_ground_straddling_a_tile_edge_takes_both_tiles() {
        let ground = GroundRect::centered([4.0, 0.5], [0.5, 0.25]);
        let range = tiles_in_reach(&ground, [4.0, 0.5], 60.0);
        assert_eq!(range.origin, [0, 0]);
        assert_eq!(range.count, [2, 1]);
    }

    #[test]
    fn a_camera_out_of_reach_sees_no_tiles() {
        let ground = GroundRect::centered([0.0, 0.0], [10.0, 10.0]);
        let range = tiles_in_reach(&ground, [500.0, 0.0], 60.0);
        assert!(range.is_empty());
        assert_eq!(range, TileRange::EMPTY);
    }
}
