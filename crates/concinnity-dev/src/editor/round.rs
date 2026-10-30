// The precision the editor writes authored values at, so a drag or slider never
// leaves float noise in the world file.

/// `v` rounded to the nearest thousandth: positions, scales and shape sliders.
pub(super) fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

/// `v` rounded to the nearest tenth: rotations in degrees.
pub(super) fn round1(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round3_keeps_three_decimals() {
        assert_eq!(round3(1.234_56), 1.235);
        assert_eq!(round3(-0.000_4), 0.0);
        assert_eq!(round3(2.0), 2.0);
    }

    #[test]
    fn round1_keeps_one_decimal() {
        assert_eq!(round1(45.06), 45.1);
        assert_eq!(round1(-12.34), -12.3);
    }
}
