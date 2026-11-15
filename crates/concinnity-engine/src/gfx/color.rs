//! Color arithmetic shared by the renderer's CPU side.

/// The Rec. 709 luminance of linear RGB `rgb`.
pub fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_has_unit_luminance_and_green_weighs_most() {
        assert!((luminance([1.0; 3]) - 1.0).abs() < 1e-6);
        assert!(luminance([0.0, 1.0, 0.0]) > luminance([1.0, 0.0, 0.0]));
        assert!(luminance([1.0, 0.0, 0.0]) > luminance([0.0, 0.0, 1.0]));
    }
}
