// Where a terrain grid's heights come from: a generated noise field, or the
// red channel of a decoded heightmap image.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::math::floor;
use crate::math::noise::{lattice_value, lcg_hash};

// Octaves of value noise, as (lattice cells across the terrain, weight): broad
// hills, then bumps on them, then fine variation.
const OCTAVES: [(f32, f32); 3] = [(2.0, 1.0), (5.0, 0.4), (13.0, 0.15)];

// Heights of a `resolution`-cell grid, row-major, from `sample(s, t)` at each
// corner's normalized position across the terrain.
fn grid_heights(resolution: u32, sample: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    let side = resolution as usize + 1;
    let n = resolution as f32;
    let mut heights = Vec::with_capacity(side * side);
    for row in 0..side {
        for col in 0..side {
            heights.push(sample(col as f32 / n, row as f32 / n));
        }
    }
    heights
}

/// Generated rolling hills over a `resolution`-cell grid: heights in
/// `[0, amplitude]`, the same for a given `seed` on every build. The hills
/// stretch with the terrain, so changing `resolution` samples the same
/// landscape more or less finely.
pub fn noise_heights(resolution: u32, amplitude: f32, seed: u32) -> Vec<f32> {
    let offset = [lcg_hash(seed), lcg_hash(seed ^ 0x9e37_79b9)];
    grid_heights(resolution, |s, t| noise(s, t, offset) * amplitude)
}

// Value noise in [0, 1] at normalized position (s, t), each octave a smoothly
// interpolated lattice shifted by `offset`.
fn noise(s: f32, t: f32, offset: [u32; 2]) -> f32 {
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for (cells, weight) in OCTAVES {
        let (gs, gt) = (s * cells, t * cells);
        let (x0, y0) = (floor(gs), floor(gt));
        let fade = |f: f32| f * f * (3.0 - 2.0 * f);
        let (fx, fy) = (fade(gs - x0), fade(gt - y0));
        let x = (x0 as u32).wrapping_add(offset[0]);
        let y = (y0 as u32).wrapping_add(offset[1]);
        let v = |dx: u32, dy: u32| lattice_value(x.wrapping_add(dx), y.wrapping_add(dy));
        let top = v(0, 0) + (v(1, 0) - v(0, 0)) * fx;
        let bottom = v(0, 1) + (v(1, 1) - v(0, 1)) * fx;
        sum += (top + (bottom - top) * fy) * weight;
        weight_sum += weight;
    }
    sum / weight_sum
}

/// Heights of a `resolution`-cell grid from a decoded `width` x `height` RGBA8
/// image: the red channel, bilinearly sampled with the image's corner texels
/// on the grid's corners, mapped from [0, 255] onto `[min, max]`.
pub fn heightmap_heights(
    resolution: u32,
    width: u32,
    height: u32,
    rgba: &[u8],
    [min, max]: [f32; 2],
) -> Result<Vec<f32>, String> {
    let red = red_channel(width, height, rgba)?;
    Ok(grid_heights(resolution, |s, t| {
        min + sample_bilinear(&red, width, height, s, t) * (max - min)
    }))
}

/// The red channel of a `width` x `height` RGBA8 image, or an error naming
/// what is wrong with it.
pub(super) fn red_channel(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("the image has no texels".into());
    }
    let texels = width as usize * height as usize;
    if rgba.len() < texels * 4 {
        return Err(format!(
            "a {width}x{height} image needs {} bytes, not {}",
            texels * 4,
            rgba.len()
        ));
    }
    Ok(rgba.chunks_exact(4).take(texels).map(|p| p[0]).collect())
}

/// Bilinearly sample `texels` (`width` x `height`, one byte each) at
/// normalized (`s`, `t`), the corner texels on the corners, as a value in
/// [0, 1].
pub(super) fn sample_bilinear(texels: &[u8], width: u32, height: u32, s: f32, t: f32) -> f32 {
    let fx = s.clamp(0.0, 1.0) * (width - 1) as f32;
    let fy = t.clamp(0.0, 1.0) * (height - 1) as f32;
    let (x0, y0) = (floor(fx) as u32, floor(fy) as u32);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (sx, sy) = (fx - x0 as f32, fy - y0 as f32);
    let at = |x: u32, y: u32| texels[(y * width + x) as usize] as f32 / 255.0;
    let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * sx;
    let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * sx;
    top + (bottom - top) * sy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_deterministic_bounded_and_seeded() {
        let a = noise_heights(16, 3.0, 7);
        assert_eq!(a, noise_heights(16, 3.0, 7));
        assert_eq!(a.len(), 17 * 17);
        assert!(a.iter().all(|h| (0.0..=3.0).contains(h)));
        let lo = a.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(hi - lo > 0.3, "expected hills, got {lo}..{hi}");
        assert_ne!(
            a,
            noise_heights(16, 3.0, 8),
            "another seed, another landscape"
        );
    }

    #[test]
    fn zero_amplitude_is_flat() {
        assert!(noise_heights(8, 0.0, 3).iter().all(|&h| h == 0.0));
    }

    // The field is a function of position across the terrain, so a finer grid
    // passes through the coarser grid's samples.
    #[test]
    fn resolution_resamples_the_same_landscape() {
        let coarse = noise_heights(8, 2.0, 1);
        let fine = noise_heights(16, 2.0, 1);
        for row in 0..9 {
            for col in 0..9 {
                let c = coarse[row * 9 + col];
                let f = fine[(2 * row) * 17 + 2 * col];
                assert!((c - f).abs() < 1e-5, "{col},{row}: {c} vs {f}");
            }
        }
    }

    // A 2x2 image: black on the left column, white on the right.
    fn left_black_right_white() -> [u8; 16] {
        [
            0, 0, 0, 255, 255, 0, 0, 255, //
            0, 0, 0, 255, 255, 0, 0, 255,
        ]
    }

    #[test]
    fn a_heightmap_maps_red_onto_the_elevation_range() {
        let h = heightmap_heights(2, 2, 2, &left_black_right_white(), [-1.0, 3.0]).unwrap();
        assert_eq!(h.len(), 9);
        for row in 0..3 {
            assert_eq!(h[row * 3], -1.0);
            assert_eq!(h[row * 3 + 1], 1.0);
            assert_eq!(h[row * 3 + 2], 3.0);
        }
    }

    #[test]
    fn a_heightmap_that_is_not_an_image_is_refused() {
        assert!(heightmap_heights(4, 0, 0, &[], [0.0, 1.0]).is_err());
        let err = heightmap_heights(4, 8, 8, &[0; 100], [0.0, 1.0]).unwrap_err();
        assert!(err.contains("needs 256 bytes"), "{err}");
    }

    #[test]
    fn a_single_texel_image_is_flat() {
        let h = heightmap_heights(4, 1, 1, &[128, 0, 0, 255], [0.0, 10.0]).unwrap();
        assert!(h.iter().all(|&v| (v - 128.0 / 255.0 * 10.0).abs() < 1e-5));
    }
}
