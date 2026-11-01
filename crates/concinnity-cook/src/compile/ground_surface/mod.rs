// Sand and dirt drawn into seamless one-meter tiles: a relief height field and
// the albedo lying over it, encoded into a texture's albedo (height in alpha)
// or tangent-space normal map.

mod dirt;
mod field;
mod sand;

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::components::{DirtLook, GroundMap, SandLook};
use rayon::prelude::*;

/// Meters a ground tile spans; one texture repeat.
pub(crate) const TILE_METERS: f32 = 1.0;

// Fewest texels across a ground tile.
const MIN_SIZE: u32 = 64;

// How far a cavity darkens the albedo at its deepest, and the cavity's width
// in meters.
const CAVITY_DARKEN: f32 = 0.35;
const CAVITY_RADIUS_M: f32 = 0.006;

/// A ground tile before encoding: relief in meters and linear albedo per
/// texel, `size` by `size`, row-major with v down the rows.
pub(super) struct Surface {
    size: usize,
    height: Vec<f32>,
    albedo: Vec<[f32; 3]>,
}

impl Surface {
    // A surface whose texels `texel(u, v)` returns, as (height, albedo).
    fn sampled(size: u32, texel: impl Fn(f32, f32) -> (f32, [f32; 3]) + Sync) -> Self {
        let size = size.max(MIN_SIZE) as usize;
        let step = 1.0 / size as f32;
        let texels: Vec<(f32, [f32; 3])> = (0..size * size)
            .into_par_iter()
            .map(|i| {
                texel(
                    ((i % size) as f32 + 0.5) * step,
                    ((i / size) as f32 + 0.5) * step,
                )
            })
            .collect();
        let (height, albedo) = texels.into_iter().unzip();
        Self {
            size,
            height,
            albedo,
        }
    }

    // Scale the albedo so its mean is `color`.
    fn match_mean(&mut self, color: [f32; 3]) {
        let n = self.albedo.len().max(1) as f32;
        let mean: [f32; 3] =
            core::array::from_fn(|c| self.albedo.iter().map(|a| a[c]).sum::<f32>() / n);
        let scale: [f32; 3] = core::array::from_fn(|c| color[c] / mean[c].max(1e-6));
        for a in &mut self.albedo {
            *a = core::array::from_fn(|c| a[c] * scale[c]);
        }
    }

    // Darken the albedo where the relief dips below its surroundings.
    fn darken_cavities(&mut self) {
        let radius = ((CAVITY_RADIUS_M / TILE_METERS) * self.size as f32)
            .round()
            .max(1.0);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a small positive texel count"
        )]
        let radius = radius as usize;
        let around = field::box_blur(&self.height, self.size, radius);
        let depth = self
            .height
            .iter()
            .zip(&around)
            .map(|(h, a)| (a - h).max(0.0))
            .fold(0.0_f32, f32::max)
            .max(1e-6);
        for ((a, h), around) in self.albedo.iter_mut().zip(&self.height).zip(&around) {
            let cavity = ((around - h).max(0.0) / depth).sqrt();
            let k = 1.0 - CAVITY_DARKEN * cavity;
            *a = a.map(|c| c * k);
        }
    }

    // RGBA8 albedo divided by `peak`, the relief's height in alpha.
    fn encode_albedo(&self, peak: [f32; 3]) -> Vec<u8> {
        let (lo, hi) = self.height_range();
        let span = (hi - lo).max(1e-9);
        let mut pixels = Vec::with_capacity(self.albedo.len() * 4);
        for (a, h) in self.albedo.iter().zip(&self.height) {
            for c in 0..3 {
                pixels.push(unorm8(a[c] / peak[c].max(1e-6)));
            }
            pixels.push(unorm8((h - lo) / span));
        }
        pixels
    }

    // RGBA8 tangent-space normal of the relief, x along u and y along v.
    fn encode_normal(&self) -> Vec<u8> {
        let n = self.size;
        let texel_m = TILE_METERS / n as f32;
        let h = |x: usize, y: usize| self.height[(y % n) * n + (x % n)];
        let mut pixels = Vec::with_capacity(n * n * 4);
        for y in 0..n {
            for x in 0..n {
                let dx = (h(x + 1, y) - h(x + n - 1, y)) / (2.0 * texel_m);
                let dy = (h(x, y + 1) - h(x, y + n - 1)) / (2.0 * texel_m);
                let len = (dx * dx + dy * dy + 1.0).sqrt();
                for c in [-dx / len, -dy / len, 1.0 / len] {
                    pixels.push(unorm8(c * 0.5 + 0.5));
                }
                pixels.push(255);
            }
        }
        pixels
    }

    fn height_range(&self) -> (f32, f32) {
        self.height
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)))
    }

    // The tile encoded as `map`, with the albedo divided by `peak`.
    fn encode(&self, map: GroundMap, peak: [f32; 3]) -> (u32, u32, Vec<u8>) {
        let side = u32::saturating_from(self.size);
        let pixels = match map {
            GroundMap::Albedo => self.encode_albedo(peak),
            GroundMap::Normal => self.encode_normal(),
        };
        (side, side, pixels)
    }
}

fn unorm8(x: f32) -> u8 {
    u8::saturating_from((x.clamp(0.0, 1.0) * 255.0).round())
}

/// The brightest albedo a sand's tile holds, per channel: its albedo map
/// stores albedo over this, and its material's tint restores it.
pub(crate) fn sand_peak(look: &SandLook) -> [f32; 3] {
    peak(look.color, look.shell_color)
}

/// The brightest albedo a dirt's tile holds, per channel: its albedo map
/// stores albedo over this, and its material's tint restores it.
pub(crate) fn dirt_peak(look: &DirtLook) -> [f32; 3] {
    peak(look.color, look.stone_color)
}

// Headroom over the body's mean and over its brightest inclusions.
fn peak(body: [f32; 3], inclusion: [f32; 3]) -> [f32; 3] {
    core::array::from_fn(|c| (body[c] * 1.8).max(inclusion[c] * 1.3).clamp(1e-3, 1.0))
}

/// `map` of the sand `look`, `size` texels across one tile.
pub(crate) fn sand_texture(look: &SandLook, map: GroundMap, size: u32) -> (u32, u32, Vec<u8>) {
    sand::surface(look, size).encode(map, sand_peak(look))
}

/// `map` of the dirt `look`, `size` texels across one tile.
pub(crate) fn dirt_texture(look: &DirtLook, map: GroundMap, size: u32) -> (u32, u32, Vec<u8>) {
    dirt::surface(look, size).encode(map, dirt_peak(look))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel_mean(pixels: &[u8], c: usize) -> f32 {
        let n = pixels.len() / 4;
        pixels.chunks_exact(4).map(|p| f32::from(p[c])).sum::<f32>() / n as f32 / 255.0
    }

    #[test]
    fn a_sands_albedo_averages_to_its_color() {
        let look = SandLook {
            shells: 0.0,
            ..SandLook::default()
        };
        let (w, h, px) = sand_texture(&look, GroundMap::Albedo, 64);
        assert_eq!((w, h, px.len()), (64, 64, 64 * 64 * 4));
        let peak = sand_peak(&look);
        for (c, (peak, color)) in peak.iter().zip(look.color).enumerate() {
            let mean = channel_mean(&px, c) * peak;
            assert!((mean - color).abs() < 0.02 * color + 0.004, "{c}: {mean}");
        }
    }

    #[test]
    fn a_dirts_albedo_stays_under_its_peak_and_spans_its_heights() {
        let look = DirtLook::default();
        let (_, _, px) = dirt_texture(&look, GroundMap::Albedo, 64);
        let heights: Vec<u8> = px.chunks_exact(4).map(|p| p[3]).collect();
        assert_eq!(heights.iter().min(), Some(&0));
        assert_eq!(heights.iter().max(), Some(&255));
        let saturated = px.chunks_exact(4).filter(|p| p[..3].contains(&255)).count();
        assert!(saturated * 50 < heights.len(), "{saturated} texels clip");
    }

    #[test]
    fn a_flat_surface_encodes_an_upright_normal() {
        let flat = Surface {
            size: 4,
            height: vec![0.0; 16],
            albedo: vec![[0.5; 3]; 16],
        };
        let px = flat.encode_normal();
        assert!(px.chunks_exact(4).all(|p| p == [128, 128, 255, 255]));
    }

    #[test]
    fn a_slope_up_u_tilts_the_normal_back_along_u() {
        let size = 8;
        let height = (0..size * size)
            .map(|i| ((i % size) as f32 * 0.25).sin() * 0.01)
            .collect();
        let surface = Surface {
            size,
            height,
            albedo: vec![[0.5; 3]; size * size],
        };
        let px = surface.encode_normal();
        // At u = 1 texel the height rises along u, so x leans negative.
        assert!(px[4] < 128, "{}", px[4]);
    }

    #[test]
    fn the_tile_repeats_without_a_seam() {
        let look = SandLook::default();
        let surface = sand::surface(&look, 64);
        let n = surface.size;
        let row_step = |y: usize, x0: usize, x1: usize| {
            (surface.height[y * n + x0] - surface.height[y * n + x1]).abs()
        };
        // The step across the wrap is no bigger than steps inside the tile.
        let across: f32 = (0..n).map(|y| row_step(y, n - 1, 0)).sum();
        let inside: f32 = (0..n).map(|y| row_step(y, n / 2 - 1, n / 2)).sum();
        assert!(across < inside * 3.0 + 1e-6, "{across} vs {inside}");
    }

    #[test]
    fn cavities_darken_only_low_texels() {
        let size = 8;
        let mut height = vec![0.0; size * size];
        height[3 * size + 3] = -0.01;
        let mut surface = Surface {
            size,
            height,
            albedo: vec![[0.5; 3]; size * size],
        };
        surface.darken_cavities();
        assert!(surface.albedo[3 * size + 3][0] < 0.5);
        assert_eq!(surface.albedo[0][0], 0.5);
    }
}
