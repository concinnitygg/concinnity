// src/editor/group_transform.rs
//
// Pure math for gizmo edits over a multi-member selection, anchored at the
// selection centroid. Translate applies one shared world-space delta; rotate
// orbits member positions about the centroid (and spins each member the same
// amount, the standard group-rotate behavior); scale pushes member positions
// away from the centroid along the dragged axis while stretching each member.
// With a single member every pivot operation degenerates to the in-place edit
// the single-selection gizmo always did.

// Mean of the member origins: the gizmo anchor and the rotate / scale pivot.
pub(crate) fn centroid(points: &[[f32; 3]]) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    for p in points {
        for i in 0..3 {
            sum[i] += p[i];
        }
    }
    let n = points.len().max(1) as f32;
    sum.map(|v| v / n)
}

pub(crate) fn translate(start: [f32; 3], axis: [f32; 3], delta: f32) -> [f32; 3] {
    [
        start[0] + axis[0] * delta,
        start[1] + axis[1] * delta,
        start[2] + axis[2] * delta,
    ]
}

// Rotate `start` about the world axis `axis` (0 = X, 1 = Y, 2 = Z) through
// `pivot` by `angle_deg`, right-handed.
pub(crate) fn orbit(start: [f32; 3], pivot: [f32; 3], axis: usize, angle_deg: f32) -> [f32; 3] {
    // The two axes spanning the rotation plane, ordered so +angle is a
    // right-hand turn about the rotation axis.
    let (b, c) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    let db = start[b] - pivot[b];
    let dc = start[c] - pivot[c];
    let mut out = start;
    out[b] = pivot[b] + cos * db - sin * dc;
    out[c] = pivot[c] + sin * db + cos * dc;
    out
}

// Push `start` away from `pivot` along one world axis by `factor`.
pub(crate) fn scale_from(start: [f32; 3], pivot: [f32; 3], axis: usize, factor: f32) -> [f32; 3] {
    let mut out = start;
    out[axis] = pivot[axis] + (start[axis] - pivot[axis]) * factor;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn centroid_is_the_member_mean() {
        assert_eq!(centroid(&[[2.0, 4.0, -6.0]]), [2.0, 4.0, -6.0]);
        assert_eq!(
            centroid(&[[0.0, 0.0, 0.0], [2.0, 4.0, -6.0]]),
            [1.0, 2.0, -3.0]
        );
        assert_eq!(centroid(&[]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn translate_moves_along_the_axis() {
        assert!(close(
            translate([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], -2.5),
            [1.0, -0.5, 3.0]
        ));
    }

    #[test]
    fn orbit_turns_right_handed_about_each_axis() {
        // +90 about Y carries +X into -Z.
        assert!(close(
            orbit([1.0, 0.0, 0.0], [0.0; 3], 1, 90.0),
            [0.0, 0.0, -1.0]
        ));
        // +90 about Z carries +X into +Y.
        assert!(close(
            orbit([1.0, 0.0, 0.0], [0.0; 3], 2, 90.0),
            [0.0, 1.0, 0.0]
        ));
        // +90 about X carries +Y into +Z.
        assert!(close(
            orbit([0.0, 1.0, 0.0], [0.0; 3], 0, 90.0),
            [0.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn orbit_about_an_offset_pivot_keeps_the_radius() {
        let p = orbit([3.0, 5.0, 0.0], [1.0, 5.0, 0.0], 1, 180.0);
        assert!(close(p, [-1.0, 5.0, 0.0]), "{p:?}");
        // A member sitting on the pivot never moves.
        assert!(close(
            orbit([1.0, 5.0, 0.0], [1.0, 5.0, 0.0], 1, 77.0),
            [1.0, 5.0, 0.0]
        ));
    }

    #[test]
    fn scale_from_stretches_only_the_dragged_axis() {
        assert!(close(
            scale_from([3.0, 2.0, 1.0], [1.0, 0.0, 0.0], 0, 2.0),
            [5.0, 2.0, 1.0]
        ));
        // A member on the pivot plane stays put.
        assert!(close(
            scale_from([1.0, 2.0, 1.0], [1.0, 0.0, 0.0], 0, 4.0),
            [1.0, 2.0, 1.0]
        ));
    }
}
