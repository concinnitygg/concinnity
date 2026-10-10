// A grass layer's density mask: one byte per texel, stretched over its
// terrain's extent.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::source::{red_channel, sample_bilinear};

/// Longest edge a density mask keeps, in texels.
pub const MAX_MASK_SIZE: u32 = 2048;

/// How densely a grass layer grows across its terrain: `width` x `height`
/// bytes, 0 bare and 255 the grass's full density, row-major from the
/// terrain's `-X, -Z` corner with the corner texels on the terrain's corners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensityMask {
    width: u32,
    height: u32,
    texels: Vec<u8>,
}

impl DensityMask {
    /// A mask of `texels`, or an error when it is empty, past
    /// [`MAX_MASK_SIZE`], or the wrong size.
    pub fn new(width: u32, height: u32, texels: Vec<u8>) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("a density mask needs at least one texel".into());
        }
        if width.max(height) > MAX_MASK_SIZE {
            return Err(format!(
                "a {width}x{height} density mask is larger than {MAX_MASK_SIZE} texels on a side"
            ));
        }
        if texels.len() != width as usize * height as usize {
            return Err(format!(
                "a {width}x{height} density mask needs {} texels, not {}",
                width as usize * height as usize,
                texels.len()
            ));
        }
        Ok(Self {
            width,
            height,
            texels,
        })
    }

    /// The red channel of a decoded RGBA8 image as a mask.
    pub fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Result<Self, String> {
        Self::new(width, height, red_channel(width, height, rgba)?)
    }

    /// Texels across.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Texels down.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The texels, row-major.
    pub fn texels(&self) -> &[u8] {
        &self.texels
    }

    /// The density at normalized position (`s`, `t`) across the terrain, in
    /// [0, 1], bilinearly filtered.
    pub fn sample(&self, s: f32, t: f32) -> f32 {
        sample_bilinear(&self.texels, self.width, self.height, s, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_mask_keeps_the_red_channel() {
        let m = DensityMask::from_rgba(2, 1, &[10, 1, 2, 3, 250, 4, 5, 6]).unwrap();
        assert_eq!((m.width(), m.height()), (2, 1));
        assert_eq!(m.texels(), &[10, 250]);
    }

    #[test]
    fn a_mask_samples_between_its_texels() {
        let m = DensityMask::new(2, 2, vec![0, 255, 0, 255]).unwrap();
        assert_eq!(m.sample(0.0, 0.0), 0.0);
        assert_eq!(m.sample(1.0, 1.0), 1.0);
        assert!((m.sample(0.5, 0.3) - 0.5).abs() < 1e-6);
        assert_eq!(m.sample(-4.0, 9.0), 0.0, "clamped to the edge");
    }

    #[test]
    fn a_mask_refuses_shapes_it_cannot_hold() {
        assert!(DensityMask::new(0, 4, vec![]).is_err());
        assert!(DensityMask::new(2, 2, vec![0; 3]).is_err());
        let err = DensityMask::new(MAX_MASK_SIZE + 1, 1, vec![0; 2049]).unwrap_err();
        assert!(err.contains("larger than"), "{err}");
    }
}
