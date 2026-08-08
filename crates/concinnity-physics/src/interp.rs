// concinnity-physics/src/interp.rs
//
// Render interpolation state for the fixed-timestep simulation. Each fixed
// tick pushes the newly simulated pose; each rendered frame samples a blend of
// the previous and current poses by the frame's accumulator alpha, so motion
// stays smooth when the render rate and the tick rate differ. Positions blend
// linearly; rotations blend as quaternions (see `convert` for the one-time
// Euler conversion at the Transform write boundary).

// A position with prev/curr tick snapshots.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PointInterp {
    prev: [f32; 3],
    curr: [f32; 3],
}

impl PointInterp {
    pub fn new(point: [f32; 3]) -> Self {
        Self {
            prev: point,
            curr: point,
        }
    }

    // The authoritative simulated position (the latest tick's).
    pub fn current(&self) -> [f32; 3] {
        self.curr
    }

    // Record a tick's result: the old current becomes the blend origin.
    pub fn push(&mut self, point: [f32; 3]) {
        self.prev = self.curr;
        self.curr = point;
    }

    // Adopt an externally written position with no blend across the jump.
    pub fn snap(&mut self, point: [f32; 3]) {
        self.prev = point;
        self.curr = point;
    }

    pub fn sample(&self, alpha: f32) -> [f32; 3] {
        lerp3(self.prev, self.curr, alpha)
    }
}

// A full pose (position + rotation quaternion) with prev/curr tick snapshots.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoseInterp {
    prev: ([f32; 3], [f32; 4]),
    curr: ([f32; 3], [f32; 4]),
}

impl PoseInterp {
    pub fn new(position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            prev: (position, rotation),
            curr: (position, rotation),
        }
    }

    pub fn push(&mut self, position: [f32; 3], rotation: [f32; 4]) {
        self.prev = self.curr;
        self.curr = (position, rotation);
    }

    pub fn sample(&self, alpha: f32) -> ([f32; 3], [f32; 4]) {
        (
            lerp3(self.prev.0, self.curr.0, alpha),
            slerp(self.prev.1, self.curr.1, alpha),
        )
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

// Spherical interpolation between unit quaternions, taking the shorter arc.
// Falls back to normalized linear blending when the arc is too small for the
// sine ratio to be stable.
fn slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let mut b = b;
    if dot < 0.0 {
        dot = -dot;
        b = [-b[0], -b[1], -b[2], -b[3]];
    }
    let (wa, wb) = if dot > 0.9995 {
        (1.0 - t, t)
    } else {
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        (
            ((1.0 - t) * theta).sin() / sin_theta,
            (t * theta).sin() / sin_theta,
        )
    };
    normalize([
        a[0] * wa + b[0] * wb,
        a[1] * wa + b[1] * wb,
        a[2] * wa + b[2] * wb,
        a[3] * wa + b[3] * wb,
    ])
}

fn normalize(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 1.0e-6 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    fn quat_y(deg: f32) -> [f32; 4] {
        let half = deg.to_radians() * 0.5;
        [0.0, half.sin(), 0.0, half.cos()]
    }

    fn assert_close3(a: [f32; 3], b: [f32; 3]) {
        for axis in 0..3 {
            assert!((a[axis] - b[axis]).abs() < 1.0e-4, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn point_blend_hits_the_endpoints_and_midpoint() {
        let mut p = PointInterp::new([0.0, 0.0, 0.0]);
        p.push([2.0, 4.0, -6.0]);
        assert_close3(p.sample(0.0), [0.0, 0.0, 0.0]);
        assert_close3(p.sample(0.5), [1.0, 2.0, -3.0]);
        assert_close3(p.sample(1.0), [2.0, 4.0, -6.0]);
    }

    #[test]
    fn push_shifts_current_to_the_blend_origin() {
        let mut p = PointInterp::new([0.0; 3]);
        p.push([1.0, 0.0, 0.0]);
        p.push([2.0, 0.0, 0.0]);
        assert_close3(p.sample(0.0), [1.0, 0.0, 0.0]);
        assert_close3(p.sample(1.0), [2.0, 0.0, 0.0]);
    }

    #[test]
    fn snap_jumps_without_blending() {
        let mut p = PointInterp::new([0.0; 3]);
        p.push([1.0, 0.0, 0.0]);
        p.snap([50.0, 0.0, 0.0]);
        assert_close3(p.sample(0.5), [50.0, 0.0, 0.0]);
    }

    #[test]
    fn slerp_halves_a_rotation() {
        let half = slerp(IDENTITY, quat_y(90.0), 0.5);
        let expected = quat_y(45.0);
        for i in 0..4 {
            assert!((half[i] - expected[i]).abs() < 1.0e-4, "{half:?}");
        }
    }

    #[test]
    fn slerp_takes_the_shorter_arc() {
        // The negated quaternion is the same rotation; the blend must not
        // swing the long way around.
        let b = quat_y(90.0);
        let negated = [-b[0], -b[1], -b[2], -b[3]];
        let half = slerp(IDENTITY, negated, 0.5);
        let expected = quat_y(45.0);
        for i in 0..4 {
            assert!(
                (half[i] - expected[i]).abs() < 1.0e-4,
                "{half:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn pose_blend_endpoints_return_the_stored_poses() {
        let mut pose = PoseInterp::new([0.0; 3], IDENTITY);
        pose.push([1.0, 2.0, 3.0], quat_y(90.0));
        let (p0, r0) = pose.sample(0.0);
        assert_close3(p0, [0.0; 3]);
        assert!((r0[3] - 1.0).abs() < 1.0e-4);
        let (p1, r1) = pose.sample(1.0);
        assert_close3(p1, [1.0, 2.0, 3.0]);
        let expected = quat_y(90.0);
        for i in 0..4 {
            assert!((r1[i] - expected[i]).abs() < 1.0e-4);
        }
    }
}
