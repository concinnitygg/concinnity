//! The tiles one simulation evolves. Each cascade is a square of the sea
//! repeating every `length` meters, sampled on `size` x `size` texels, and
//! holds only its own band of wavelengths: from four of its texels up to
//! where the next longer cascade's band ends. The bands meet exactly, so no
//! wavelength is counted twice, and the shortest band ends where the
//! procedural ripples begin.

use super::{MAX_OCEAN_CASCADES, OceanQuality};
use core::f32::consts::{PI, TAU};

/// The longest cascade's side, in peak wavelengths: long enough that its
/// wavenumber grid samples the spectrum's peak finely in heading and length.
pub const TILE_PER_PEAK: f32 = 10.0;

/// The shortest wave a cascade holds, in its texels: half its Nyquist
/// wavelength, so bilinear filtering never blurs what it holds.
pub const SHORTEST_TEXELS: f32 = 4.0;

/// The cascades of one simulation, longest first.
#[derive(Debug, Clone, PartialEq)]
pub struct Cascades {
    lengths: [f32; MAX_OCEAN_CASCADES],
    count: usize,
    size: u32,
}

impl Cascades {
    /// `quality`'s cascades for a sea peaking at `peak_wavelength` meters:
    /// the longest [`TILE_PER_PEAK`] peaks across, each next one
    /// [`Cascades::ratio`] times shorter.
    pub fn new(peak_wavelength: f32, quality: OceanQuality) -> Self {
        let quality = quality.clamped();
        let ratio = Self::ratio(quality.size);
        let mut lengths = [0.0; MAX_OCEAN_CASCADES];
        let mut length = TILE_PER_PEAK * peak_wavelength;
        for slot in lengths.iter_mut().take(quality.cascades as usize) {
            *slot = length;
            length /= ratio;
        }
        Self {
            lengths,
            count: quality.cascades as usize,
            size: quality.size,
        }
    }

    /// How many times shorter each cascade is than the one before it: a
    /// band's lower edge then falls eight of its own wavenumber steps from
    /// zero, far enough from the coarse center of its grid.
    pub fn ratio(size: u32) -> f32 {
        (size as f32 / 32.0).max(2.0)
    }

    /// How many cascades there are.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Texels along each side.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Each cascade's side, in meters, longest first.
    pub fn lengths(&self) -> &[f32] {
        &self.lengths[..self.count]
    }

    /// The cascades' sides padded with zeros, for a shader lane.
    pub fn padded_lengths(&self) -> [f32; MAX_OCEAN_CASCADES] {
        self.lengths
    }

    /// The shortest wave cascade `c` holds, in meters.
    pub fn shortest_wavelength(&self, c: usize) -> f32 {
        SHORTEST_TEXELS * self.lengths[c] / self.size as f32
    }

    /// The wavenumbers, in radians per meter, cascade `c` holds:
    /// `[lowest, highest)`.
    pub fn band(&self, c: usize) -> [f32; 2] {
        let lo = if c == 0 {
            0.0
        } else {
            TAU / self.shortest_wavelength(c - 1)
        };
        [lo, TAU / self.shortest_wavelength(c)]
    }

    /// The wavelength, in meters, the procedural ripples start below: the
    /// shortest wave the last cascade holds.
    pub fn ripple_wavelength(&self) -> f32 {
        self.shortest_wavelength(self.count - 1)
    }

    /// The highest wavenumber cascade `c`'s grid can hold, its Nyquist limit.
    pub fn nyquist(&self, c: usize) -> f32 {
        PI * self.size as f32 / self.lengths[c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cascades(size: u32, count: u32) -> Cascades {
        Cascades::new(
            56.0,
            OceanQuality {
                size,
                cascades: count,
            },
        )
    }

    // Every wavenumber from the longest tile's fundamental to where the
    // ripples start falls in exactly one cascade's band.
    #[test]
    fn the_bands_tile_without_double_counting() {
        for (size, count) in [(256, 3), (128, 2), (128, 3), (64, 4), (256, 1)] {
            let c = cascades(size, count);
            assert_eq!(c.count(), count as usize);
            let lo = TAU / c.lengths()[0];
            let hi = TAU / c.ripple_wavelength();
            for i in 0..2000 {
                let k = lo * (hi / lo).powf(i as f32 / 2000.0);
                let holders = (0..c.count())
                    .filter(|&b| {
                        let [a, z] = c.band(b);
                        k >= a && k < z
                    })
                    .count();
                assert_eq!(holders, 1, "{size}x{count} k {k}");
            }
            for b in 1..c.count() {
                assert_eq!(c.band(b)[0], c.band(b - 1)[1]);
            }
        }
    }

    // Each band fits its own grid: under its Nyquist limit and, past the
    // first, several wavenumber steps from the grid's center.
    #[test]
    fn each_band_fits_its_grid() {
        for (size, count) in [(256, 3), (128, 2), (64, 4)] {
            let c = cascades(size, count);
            for b in 0..c.count() {
                let [lo, hi] = c.band(b);
                assert!(hi <= 0.5 * c.nyquist(b) + 1e-3, "{size} {b}");
                if b > 0 {
                    let step = TAU / c.lengths()[b];
                    assert!(lo >= 6.0 * step, "{size} {b}: {}", lo / step);
                }
            }
        }
    }

    // The full simulation holds the open sea down to a band of decimeter
    // waves, and the small one leaves more to the ripples.
    #[test]
    fn the_ripples_take_over_below_the_last_cascade() {
        let full = cascades(256, 3);
        assert!((full.lengths()[0] - 560.0).abs() < 1e-3);
        assert!((0.05..0.3).contains(&full.ripple_wavelength()));
        let small = cascades(128, 2);
        assert!(small.ripple_wavelength() > full.ripple_wavelength());
        assert_eq!(full.padded_lengths()[3], 0.0);
    }
}
