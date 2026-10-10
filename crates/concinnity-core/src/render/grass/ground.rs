//! The ground a grass layer roots in, as the grass kernel reads it: a
//! terrain's heights and a layer's density mask, packed into the two buffers
//! every layer shares, and the CPU twin of the kernel's lookups into them.
//!
//! [`GrassGround::surface_at`] and [`GrassMask::density_at`] are line-for-line
//! the `grass_ground` and `grass_mask_density` functions in `grass.hlsl`, so
//! tests can hold the kernel's placement to the terrain's own surface.

use alloc::vec::Vec;

use super::tiles::GroundRect;
use crate::math::{floor, sqrt};
use crate::terrain::{DensityMask, TerrainGrid};

/// The heights and mask texels every layer's ground and mask index into.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GrassGroundBuffers {
    /// Every terrain's heights, each terrain's run row-major.
    pub heights: Vec<f32>,
    /// Every mask's texels, four to a word, low byte first.
    pub mask_words: Vec<u32>,
    // Texels written to `mask_words`.
    mask_texels: u32,
}

impl GrassGroundBuffers {
    /// Append `grid`'s heights, returning the index of the first.
    pub fn push_heights(&mut self, grid: &TerrainGrid) -> u32 {
        let offset = self.heights.len() as u32;
        self.heights.extend_from_slice(grid.heights());
        offset
    }

    /// Append `mask`'s texels, returning where the layer finds them.
    pub fn push_mask(&mut self, mask: &DensityMask) -> GrassMask {
        let offset = self.mask_texels;
        for &texel in mask.texels() {
            let i = self.mask_texels as usize;
            if i.is_multiple_of(4) {
                self.mask_words.push(0);
            }
            self.mask_words[i / 4] |= u32::from(texel) << (8 * (i % 4));
            self.mask_texels += 1;
        }
        GrassMask {
            offset,
            width: mask.width(),
            height: mask.height(),
        }
    }

    /// The mask words as the kernel binds them: never empty, since every
    /// backend needs a buffer to bind even when no layer is masked.
    pub fn bound_mask_words(&self) -> &[u32] {
        if self.mask_words.is_empty() {
            &[0]
        } else {
            &self.mask_words
        }
    }

    /// The texel at index `i` of the mask buffer, in [0, 255].
    pub fn mask_texel(&self, i: u32) -> u32 {
        (self.mask_words[(i >> 2) as usize] >> (8 * (i & 3))) & 0xff
    }
}

/// Where a layer's density mask sits in the mask buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrassMask {
    /// Index of the first texel.
    pub offset: u32,
    /// Texels across.
    pub width: u32,
    /// Texels down.
    pub height: u32,
}

impl GrassMask {
    /// The packed size the kernel reads: width low, height high.
    pub fn packed_size(mask: Option<&Self>) -> u32 {
        mask.map_or(0, |m| m.width | (m.height << 16))
    }

    /// The density at normalized (`s`, `t`) across the terrain, in [0, 1],
    /// as the kernel's `grass_mask_density` reads it.
    pub fn density_at(&self, buffers: &GrassGroundBuffers, s: f32, t: f32) -> f32 {
        let (w, h) = (self.width, self.height);
        let fx = s.clamp(0.0, 1.0) * (w - 1) as f32;
        let fy = t.clamp(0.0, 1.0) * (h - 1) as f32;
        let (x0, y0) = (floor(fx) as u32, floor(fy) as u32);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (sx, sy) = (fx - x0 as f32, fy - y0 as f32);
        let at = |x: u32, y: u32| buffers.mask_texel(self.offset + y * w + x) as f32 / 255.0;
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * sx;
        let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * sx;
        top + (bottom - top) * sy
    }
}

/// A terrain's surface as a layer's kernel reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassGround {
    /// The terrain's footprint.
    pub rect: GroundRect,
    /// World height of the terrain's base.
    pub base_y: f32,
    /// Lowest and highest ground, world heights.
    pub y_range: [f32; 2],
    /// Grid cells along each side.
    pub resolution: u32,
    /// Index of the terrain's first height in the heights buffer.
    pub heights_offset: u32,
}

/// The ground under a point: its world height and its unit normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundSample {
    /// World height.
    pub y: f32,
    /// Unit normal.
    pub normal: [f32; 3],
}

impl GrassGround {
    /// The ground of the terrain centered on `center` with height `grid`,
    /// whose heights start at `heights_offset`.
    pub fn new(center: [f32; 3], grid: &TerrainGrid, heights_offset: u32) -> Self {
        let [lo, hi] = grid.height_range();
        Self {
            rect: GroundRect::centered([center[0], center[2]], grid.extent()),
            base_y: center[1],
            y_range: [center[1] + lo, center[1] + hi],
            resolution: grid.resolution(),
            heights_offset,
        }
    }

    /// The ground at world `xz`, as the kernel's `grass_ground` finds it: the
    /// height on the grid triangle under the point and the normal blended
    /// across its cell. `xz` must lie on the terrain.
    pub fn surface_at(&self, buffers: &GrassGroundBuffers, xz: [f32; 2]) -> GroundSample {
        let n = self.resolution as f32;
        let size = [
            self.rect.max[0] - self.rect.min[0],
            self.rect.max[1] - self.rect.min[1],
        ];
        let gx = ((xz[0] - self.rect.min[0]) / size[0] * n).clamp(0.0, n);
        let gz = ((xz[1] - self.rect.min[1]) / size[1] * n).clamp(0.0, n);
        let col = floor(gx).min(n - 1.0);
        let row = floor(gz).min(n - 1.0);
        let (fx, fz) = (gx - col, gz - row);
        let side = self.resolution + 1;
        let at = |c: f32, r: f32| {
            buffers.heights[(self.heights_offset + r as u32 * side + c as u32) as usize]
        };
        let h00 = at(col, row);
        let h10 = at(col + 1.0, row);
        let h01 = at(col, row + 1.0);
        let h11 = at(col + 1.0, row + 1.0);
        let h = if fx + fz <= 1.0 {
            h00 + (h10 - h00) * fx + (h01 - h00) * fz
        } else {
            h11 + (h01 - h11) * (1.0 - fx) + (h10 - h11) * (1.0 - fz)
        };
        let dx = ((h10 - h00) * (1.0 - fz) + (h11 - h01) * fz) * n / size[0];
        let dz = ((h01 - h00) * (1.0 - fx) + (h11 - h10) * fx) * n / size[1];
        let len = sqrt(dx * dx + 1.0 + dz * dz);
        GroundSample {
            y: self.base_y + h,
            normal: [-dx / len, 1.0 / len, -dz / len],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{noise_heights, terrain_chunks};
    use alloc::vec;

    const CENTER: [f32; 3] = [-7.0, 2.5, 11.0];

    fn grid() -> TerrainGrid {
        TerrainGrid::new(20, [15.0, 9.0], noise_heights(20, 3.5, 4)).unwrap()
    }

    // A ground sharing its buffers with a terrain packed ahead of it, so the
    // offset is exercised.
    fn ground() -> (GrassGround, GrassGroundBuffers, TerrainGrid) {
        let mut buffers = GrassGroundBuffers::default();
        let other = TerrainGrid::new(4, [1.0, 1.0], vec![9.0; 25]).unwrap();
        buffers.push_heights(&other);
        let g = grid();
        let offset = buffers.push_heights(&g);
        assert_eq!(offset, 25);
        (GrassGround::new(CENTER, &g, offset), buffers, g)
    }

    fn points() -> impl Iterator<Item = [f32; 2]> {
        (0..60).map(|i| {
            let x = -14.9 + (i * 13 % 60) as f32 * 0.4966;
            let z = -8.95 + (i * 29 % 60) as f32 * 0.2983;
            [x, z]
        })
    }

    // The kernel roots every blade on the grid's own surface: the height it
    // reads is the height the terrain grid (and so its collider) reports.
    #[test]
    fn grass_roots_on_the_grid_surface() {
        let (ground, buffers, grid) = ground();
        for [x, z] in points() {
            let sample = ground.surface_at(&buffers, [CENTER[0] + x, CENTER[2] + z]);
            let expected = CENTER[1] + grid.height_at([x, z]).unwrap();
            assert!(
                (sample.y - expected).abs() < 1e-4,
                "({x}, {z}): grass {} vs grid {expected}",
                sample.y
            );
            let slope = grid.slope_at([x, z]).unwrap();
            let n = sample.normal;
            assert!((-n[0] / n[1] - slope[0]).abs() < 1e-3, "{n:?} vs {slope:?}");
            assert!((-n[2] / n[1] - slope[1]).abs() < 1e-3, "{n:?} vs {slope:?}");
        }
    }

    // The same agreement held against the render mesh itself: every chunk
    // vertex sits where the grass kernel reads the ground.
    #[test]
    fn grass_roots_on_the_render_mesh() {
        let (ground, buffers, grid) = ground();
        for chunk in terrain_chunks(&grid) {
            for v in &chunk.vertices {
                let world = [CENTER[0] + v.pos[0], CENTER[2] + v.pos[2]];
                let sample = ground.surface_at(&buffers, world);
                assert!((sample.y - (CENTER[1] + v.pos[1])).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn the_ground_spans_the_terrain_and_its_heights() {
        let (ground, _, grid) = ground();
        assert_eq!(ground.rect.min, [CENTER[0] - 15.0, CENTER[2] - 9.0]);
        assert_eq!(ground.rect.max, [CENTER[0] + 15.0, CENTER[2] + 9.0]);
        let [lo, hi] = grid.height_range();
        assert_eq!(ground.y_range, [CENTER[1] + lo, CENTER[1] + hi]);
        assert_eq!(ground.resolution, 20);
    }

    // Masks pack four texels to a word back to back, and each reads back as
    // the mask's own filtered density.
    #[test]
    fn masks_pack_and_sample_like_the_mask_itself() {
        let mut buffers = GrassGroundBuffers::default();
        let a = DensityMask::new(3, 1, vec![1, 2, 3]).unwrap();
        let b = DensityMask::new(3, 3, vec![0, 40, 80, 120, 160, 200, 240, 255, 7]).unwrap();
        let ma = buffers.push_mask(&a);
        let mb = buffers.push_mask(&b);
        assert_eq!((ma.offset, mb.offset), (0, 3));
        assert_eq!(buffers.mask_words.len(), 3);
        assert_eq!(buffers.mask_texel(3), 0);
        assert_eq!(buffers.mask_texel(8), 200);
        for &(s, t) in &[(0.0, 0.0), (0.3, 0.7), (1.0, 0.5), (0.81, 0.12)] {
            let gpu = mb.density_at(&buffers, s, t);
            assert!((gpu - b.sample(s, t)).abs() < 1e-6, "({s}, {t})");
        }
        assert_eq!(GrassMask::packed_size(Some(&mb)), 3 | (3 << 16));
        assert_eq!(GrassMask::packed_size(None), 0);
        assert_eq!(buffers.bound_mask_words(), &buffers.mask_words[..]);
        assert_eq!(GrassGroundBuffers::default().bound_mask_words(), &[0]);
    }
}
