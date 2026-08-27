// src/editor/orbit.rs
//
// Pure orbit (tumble) math: a camera position expressed as spherical
// coordinates around a pivot, in the same yaw/pitch convention as the fly
// camera (yaw 0 faces -Z, positive pitch looks up). The drag drive keeps the
// camera's orientation offset from the orbit angles constant, so a camera
// that was looking at the pivot keeps looking at it, and one that was not
// keeps its subject at the same place on screen while circling.

use super::framing;

// Matches the fly camera's look sensitivity so tumbling and flying feel alike.
pub(crate) const ORBIT_SENS: f32 = 0.003;

const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

// Decompose a camera offset from the pivot into (distance, yaw, pitch) such
// that `position_from_spherical` reproduces it: the angles are those of a
// camera at that offset looking straight at the pivot.
pub(crate) fn spherical_from_offset(offset: [f32; 3]) -> (f32, f32, f32) {
    let dist = (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt();
    if dist <= f32::EPSILON {
        return (0.0, 0.0, 0.0);
    }
    let pitch = (-offset[1] / dist).clamp(-1.0, 1.0).asin();
    let yaw = offset[0].atan2(offset[2]);
    (dist, yaw, pitch)
}

// The camera position for orbit angles around a pivot: the point `dist`
// behind the pivot along the (yaw, pitch) view direction.
pub(crate) fn position_from_spherical(
    pivot: [f32; 3],
    dist: f32,
    yaw: f32,
    pitch: f32,
) -> [f32; 3] {
    let f = framing::forward(yaw, pitch);
    [
        pivot[0] - f[0] * dist,
        pivot[1] - f[1] * dist,
        pivot[2] - f[2] * dist,
    ]
}

// One tumble step: apply a mouse delta to the orbit angles, clamping pitch
// short of the poles. Same signs as the fly look, so dragging right turns the
// view right.
pub(crate) fn apply_deltas(yaw: f32, pitch: f32, dx: f32, dy: f32) -> (f32, f32) {
    (
        yaw - dx * ORBIT_SENS,
        (pitch - dy * ORBIT_SENS).clamp(-PITCH_LIMIT, PITCH_LIMIT),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::math::vec3::sub;

    #[test]
    fn spherical_round_trips() {
        let pivot = [1.0, 2.0, 3.0];
        let pos = [4.0, 5.0, -2.0];
        let (dist, yaw, pitch) = spherical_from_offset(sub(pos, pivot));
        let back = position_from_spherical(pivot, dist, yaw, pitch);
        for a in 0..3 {
            assert!((back[a] - pos[a]).abs() < 1e-4, "{back:?} vs {pos:?}");
        }
    }

    #[test]
    fn spherical_angles_look_at_the_pivot() {
        let pivot = [0.0; 3];
        let pos = [3.0, 1.5, 4.0];
        let (dist, yaw, pitch) = spherical_from_offset(sub(pos, pivot));
        // The forward vector at those angles points from the camera to the pivot.
        let f = framing::forward(yaw, pitch);
        for a in 0..3 {
            assert!((pos[a] + f[a] * dist - pivot[a]).abs() < 1e-4);
        }
    }

    #[test]
    fn tumble_preserves_distance() {
        let pivot = [2.0, 0.0, -1.0];
        let (dist, mut yaw, mut pitch) = spherical_from_offset([3.0, 2.0, 1.0]);
        for _ in 0..100 {
            (yaw, pitch) = apply_deltas(yaw, pitch, 17.0, -9.0);
        }
        let pos = position_from_spherical(pivot, dist, yaw, pitch);
        let off = sub(pos, pivot);
        let d = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt();
        assert!((d - dist).abs() < 1e-3);
    }

    #[test]
    fn pitch_clamps_short_of_the_poles() {
        let (_, pitch) = apply_deltas(0.0, 0.0, 0.0, 1e6);
        assert!(pitch > -std::f32::consts::FRAC_PI_2);
        let (_, pitch) = apply_deltas(0.0, 0.0, 0.0, -1e6);
        assert!(pitch < std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn zero_offset_is_harmless() {
        assert_eq!(spherical_from_offset([0.0; 3]), (0.0, 0.0, 0.0));
    }
}
