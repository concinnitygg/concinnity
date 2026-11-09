// Directions from yaw and pitch, and turning between angles. Yaw 0 looks down
// -Z and turns toward -X; pitch lifts toward +Y.

use super::scalar::{atan2, rem_euclid, sin_cos};

/// The unit direction a yaw and pitch (radians) look along.
pub fn forward(yaw: f32, pitch: f32) -> [f32; 3] {
    let (sin_yaw, cos_yaw) = sin_cos(yaw);
    let (sin_pitch, cos_pitch) = sin_cos(pitch);
    [-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch]
}

/// The level unit direction a yaw (radians) faces.
pub fn flat_forward(yaw: f32) -> [f32; 3] {
    let (s, c) = sin_cos(yaw);
    [-s, 0.0, -c]
}

/// The level unit direction to the right of [`flat_forward`].
pub fn flat_right(yaw: f32) -> [f32; 3] {
    let (s, c) = sin_cos(yaw);
    [c, 0.0, -s]
}

/// The yaw whose [`flat_forward`] points along `dir`'s level part.
pub fn heading_of(dir: [f32; 3]) -> f32 {
    atan2(-dir[0], -dir[2])
}

/// The signed turn from `from` to `to` the short way round, in
/// `-full_turn / 2 .. full_turn / 2`, for angles measured in units where one
/// revolution is `full_turn` (`TAU` for radians, 360 for degrees).
pub fn shortest_turn(from: f32, to: f32, full_turn: f32) -> f32 {
    let half = full_turn * 0.5;
    rem_euclid(to - from + half, full_turn) - half
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx::assert_close;
    use core::f32::consts::{FRAC_PI_2, PI, TAU};

    const TOL: f32 = 1e-6;

    #[test]
    fn yaw_zero_looks_down_negative_z_and_turns_toward_negative_x() {
        assert_close(forward(0.0, 0.0), [0.0, 0.0, -1.0], TOL);
        assert_close(forward(FRAC_PI_2, 0.0), [-1.0, 0.0, 0.0], TOL);
        assert_close(forward(0.0, FRAC_PI_2), [0.0, 1.0, 0.0], TOL);
        assert_close(flat_forward(FRAC_PI_2), forward(FRAC_PI_2, 0.0), TOL);
        assert_close(flat_right(0.0), [1.0, 0.0, 0.0], TOL);
    }

    #[test]
    fn heading_inverts_the_flat_forward() {
        for yaw in [-3.0, -1.0, 0.0, 0.5, 2.5] {
            assert!((heading_of(flat_forward(yaw)) - yaw).abs() < 1e-5);
        }
    }

    #[test]
    fn turns_take_the_short_way_in_any_unit() {
        assert!((shortest_turn(0.1, TAU - 0.1, TAU) + 0.2).abs() < 1e-5);
        assert!((shortest_turn(-PI + 0.1, PI - 0.1, TAU) + 0.2).abs() < 1e-5);
        assert_eq!(shortest_turn(350.0, 10.0, 360.0), 20.0);
        assert_eq!(shortest_turn(10.0, 350.0, 360.0), -20.0);
        assert_eq!(shortest_turn(0.0, 180.0, 360.0), -180.0);
    }
}
