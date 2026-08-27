// Render interpolation state for the fixed-timestep simulation. Each fixed
// tick pushes the newly simulated pose; each rendered frame samples a blend of
// the previous and current poses by the frame's accumulator alpha, so motion
// stays smooth when the render rate and the tick rate differ. Positions blend
// linearly; rotations blend as quaternions through the crate's own
// `quat_slerp` (see `convert` for the one-time Euler conversion at the
// Transform write boundary).

use crate::gfx::transform::quat_slerp;
use crate::math::vec3::lerp;

// A position with prev/curr tick snapshots.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PointInterp {
    prev: [f32; 3],
    curr: [f32; 3],
}

impl PointInterp {
    pub(crate) fn new(point: [f32; 3]) -> Self {
        Self {
            prev: point,
            curr: point,
        }
    }

    // The authoritative simulated position (the latest tick's).
    pub(crate) fn current(&self) -> [f32; 3] {
        self.curr
    }

    // Record a tick's result: the old current becomes the blend origin.
    pub(crate) fn push(&mut self, point: [f32; 3]) {
        self.prev = self.curr;
        self.curr = point;
    }

    // Adopt an externally written position with no blend across the jump.
    pub(crate) fn snap(&mut self, point: [f32; 3]) {
        self.prev = point;
        self.curr = point;
    }

    pub(crate) fn sample(&self, alpha: f32) -> [f32; 3] {
        lerp(self.prev, self.curr, alpha)
    }
}

// A full pose (position + rotation quaternion) with prev/curr tick snapshots.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoseInterp {
    prev: ([f32; 3], [f32; 4]),
    curr: ([f32; 3], [f32; 4]),
}

impl PoseInterp {
    pub(crate) fn new(position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            prev: (position, rotation),
            curr: (position, rotation),
        }
    }

    pub(crate) fn push(&mut self, position: [f32; 3], rotation: [f32; 4]) {
        self.prev = self.curr;
        self.curr = (position, rotation);
    }

    pub(crate) fn sample(&self, alpha: f32) -> ([f32; 3], [f32; 4]) {
        (
            lerp(self.prev.0, self.curr.0, alpha),
            quat_slerp(self.prev.1, self.curr.1, alpha),
        )
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

    // The pose blend is the shared `quat_slerp`, so what this covers is that a
    // pose reaches it: the halved rotation, and the shorter arc across a
    // negated (identical) quaternion.
    #[test]
    fn pose_rotation_blends_along_the_shorter_arc() {
        let target = quat_y(90.0);
        let negated = [-target[0], -target[1], -target[2], -target[3]];
        let expected = quat_y(45.0);
        for end in [target, negated] {
            let mut pose = PoseInterp::new([0.0; 3], IDENTITY);
            pose.push([0.0; 3], end);
            let (_, half) = pose.sample(0.5);
            for i in 0..4 {
                assert!(
                    (half[i] - expected[i]).abs() < 1.0e-4,
                    "{half:?} != {expected:?}"
                );
            }
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
