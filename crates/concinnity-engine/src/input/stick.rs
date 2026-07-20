// src/input/stick.rs
//
// Pure stick-shaping math: radial deadzone and look response curve. Both
// operate on a whole [x, y] vector so direction is preserved (a per-component
// deadzone would snap diagonals toward the axes).

// Clamp a stick vector to the unit disc. Cheap pads overshoot slightly past
// 1.0 on the diagonals; everything downstream assumes magnitude <= 1.
fn clamp_unit(v: [f32; 2]) -> [f32; 2] {
    let mag = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if mag > 1.0 {
        [v[0] / mag, v[1] / mag]
    } else {
        v
    }
}

// Radial deadzone: deflections at or below `deadzone` read as rest, and the
// remaining band rescales to the full [0, 1] range along the same direction,
// so movement ramps from zero exactly at the deadzone edge with no jump.
pub(crate) fn radial_deadzone(v: [f32; 2], deadzone: f32) -> [f32; 2] {
    let v = clamp_unit(v);
    let dz = deadzone.clamp(0.0, 0.95);
    let mag = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if mag <= dz {
        return [0.0, 0.0];
    }
    let scaled = (mag - dz) / (1.0 - dz);
    [v[0] / mag * scaled, v[1] / mag * scaled]
}

// Response curve: raise the magnitude to `exponent` while keeping direction,
// so small deflections give fine control and full deflection stays full.
// Expects a deadzone-shaped vector (magnitude <= 1); exponent 1 is linear.
pub(crate) fn response_curve(v: [f32; 2], exponent: f32) -> [f32; 2] {
    let mag = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if mag <= f32::EPSILON {
        return [0.0, 0.0];
    }
    let curved = mag.powf(exponent.max(f32::EPSILON));
    [v[0] / mag * curved, v[1] / mag * curved]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mag(v: [f32; 2]) -> f32 {
        (v[0] * v[0] + v[1] * v[1]).sqrt()
    }

    #[test]
    fn deadzone_zeroes_small_deflections() {
        assert_eq!(radial_deadzone([0.1, 0.05], 0.15), [0.0, 0.0]);
        assert_eq!(radial_deadzone([0.0, 0.0], 0.15), [0.0, 0.0]);
        // Exactly at the edge is still rest, so the ramp starts from zero.
        assert_eq!(radial_deadzone([0.15, 0.0], 0.15), [0.0, 0.0]);
    }

    #[test]
    fn deadzone_rescales_to_the_full_range() {
        // Just past the edge ramps from ~0; full deflection stays full.
        let near = radial_deadzone([0.16, 0.0], 0.15);
        assert!(near[0] > 0.0 && near[0] < 0.02, "{near:?}");
        let full = radial_deadzone([1.0, 0.0], 0.15);
        assert!((full[0] - 1.0).abs() < 1e-6, "{full:?}");
        // Midway through the live band lands midway through the output.
        let mid = radial_deadzone([0.575, 0.0], 0.15);
        assert!((mid[0] - 0.5).abs() < 1e-6, "{mid:?}");
    }

    #[test]
    fn deadzone_preserves_direction() {
        let v = radial_deadzone([0.6, 0.6], 0.15);
        assert!((v[0] - v[1]).abs() < 1e-6, "diagonal stays diagonal: {v:?}");
        let v = radial_deadzone([-0.8, 0.4], 0.15);
        assert!(v[0] < 0.0 && v[1] > 0.0, "signs preserved: {v:?}");
        assert!((v[0] / v[1] + 2.0).abs() < 1e-5, "ratio preserved: {v:?}");
    }

    #[test]
    fn deadzone_clamps_overshooting_diagonals() {
        // A cheap pad reporting (1, 1) is clamped to the unit disc first, so
        // the output magnitude never exceeds 1.
        let v = radial_deadzone([1.0, 1.0], 0.15);
        assert!(mag(v) <= 1.0 + 1e-6, "{v:?}");
    }

    #[test]
    fn zero_deadzone_is_identity_inside_the_disc() {
        let v = radial_deadzone([0.3, -0.4], 0.0);
        assert!(
            (v[0] - 0.3).abs() < 1e-6 && (v[1] + 0.4).abs() < 1e-6,
            "{v:?}"
        );
    }

    #[test]
    fn response_curve_softens_small_and_keeps_full() {
        let small = response_curve([0.5, 0.0], 2.0);
        assert!((small[0] - 0.25).abs() < 1e-6, "{small:?}");
        let full = response_curve([0.0, 1.0], 2.0);
        assert!((full[1] - 1.0).abs() < 1e-6, "{full:?}");
        assert_eq!(response_curve([0.0, 0.0], 2.0), [0.0, 0.0]);
    }

    #[test]
    fn response_curve_preserves_direction() {
        let v = response_curve([-0.4, 0.4], 2.0);
        assert!(v[0] < 0.0 && v[1] > 0.0, "{v:?}");
        assert!((v[0] + v[1]).abs() < 1e-6, "diagonal stays diagonal: {v:?}");
    }

    #[test]
    fn response_curve_exponent_one_is_linear() {
        let v = response_curve([0.5, -0.25], 1.0);
        assert!(
            (v[0] - 0.5).abs() < 1e-6 && (v[1] + 0.25).abs() < 1e-6,
            "{v:?}"
        );
    }
}
