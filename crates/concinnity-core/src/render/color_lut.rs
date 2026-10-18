//! The color grade's 3D lookup table as the composite pass binds it.

/// Edge length of [`IDENTITY_LUT`].
pub const IDENTITY_LUT_SIZE: u32 = 2;

/// A 2x2x2 RGBA8 lookup table holding the corners of the unit RGB cube, red
/// fastest, then green, then blue. Trilinear filtering across the corners
/// reproduces any input, so a world with no `ColorLut` binds this and the grade
/// is a no-op at any strength.
pub const IDENTITY_LUT: [u8; 32] = identity_lut();

const fn identity_lut() -> [u8; 32] {
    let mut data = [255u8; 32];
    let mut i = 0;
    while i < 8 {
        data[i * 4] = (i & 1) as u8 * 255;
        data[i * 4 + 1] = ((i >> 1) & 1) as u8 * 255;
        data[i * 4 + 2] = ((i >> 2) & 1) as u8 * 255;
        i += 1;
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_lut_walks_red_then_green_then_blue() {
        let texel = |i: usize| &IDENTITY_LUT[i * 4..i * 4 + 4];
        assert_eq!(texel(0), [0, 0, 0, 255]);
        assert_eq!(texel(1), [255, 0, 0, 255]);
        assert_eq!(texel(2), [0, 255, 0, 255]);
        assert_eq!(texel(4), [0, 0, 255, 255]);
        assert_eq!(texel(7), [255, 255, 255, 255]);
        let edge = IDENTITY_LUT_SIZE as usize;
        assert_eq!(IDENTITY_LUT.len(), edge * edge * edge * 4);
    }
}
