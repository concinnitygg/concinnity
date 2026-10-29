//! The camera half of the motion-vector history: the previous frame's
//! unjittered view-projection, which the G-buffer pre-pass reprojects the sky
//! and every surface through, and the clock and camera position that frame's
//! vertex hooks read.
//! The per-object half is the model-history ring (see
//! [`crate::render::temporal::model_history`]).

use crate::transform::{IDENTITY, Mat4, mat4_mul};

/// One frame's camera and clock, as motion reprojects through them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewFrame {
    /// The unjittered view-projection.
    pub vp: Mat4,
    /// The main pass's `elapsed`, in seconds.
    pub elapsed: f32,
    /// The camera's world position.
    pub cam_pos: [f32; 3],
}

/// The previous frame's camera and clock, absent until a frame has recorded
/// one, and the moves of the world since.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ViewHistory {
    prev: Option<ViewFrame>,
    // The rebases since the previous frame, composed; `None` for none.
    carry: Option<Mat4>,
}

impl ViewHistory {
    /// The frame this frame's motion reprojects to. A view with no history yet
    /// reprojects to itself, so its first motion frame reads no camera or
    /// clock motion rather than reprojecting through a placeholder.
    pub(crate) fn prev_or(&self, cur: ViewFrame) -> ViewFrame {
        self.prev.unwrap_or(cur)
    }

    /// The move that carries a transform the previous frame recorded into
    /// this frame: identity unless the world was rebased since.
    pub(crate) fn carry(&self) -> Mat4 {
        self.carry.unwrap_or(IDENTITY)
    }

    /// Record this frame as the next frame's history.
    pub(crate) fn advance(&mut self, cur: ViewFrame) {
        self.prev = Some(cur);
        self.carry = None;
    }

    /// Forget the history, so the next motion frame starts over.
    pub(crate) fn reset(&mut self) {
        self.prev = None;
        self.carry = None;
    }

    /// Carry the history into the frame the world moved to, so a point that
    /// stayed put reprojects to where the previous frame drew it.
    pub(crate) fn rebase(&mut self, rebase: &crate::planet::Rebase) {
        if let Some(prev) = self.prev.as_mut() {
            prev.vp = rebase.reproject(prev.vp);
            prev.cam_pos = rebase.apply_point(prev.cam_pos);
        }
        self.carry = Some(mat4_mul(rebase.matrix(), self.carry()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUR: ViewFrame = ViewFrame {
        vp: [
            [1.2, 0.0, 0.0, 0.0],
            [0.0, 2.1, 0.0, 0.0],
            [0.3, 0.1, -1.0, -1.0],
            [0.0, 0.0, -0.2, 0.0],
        ],
        elapsed: 4.5,
        cam_pos: [1.0, 2.0, 3.0],
    };

    const EARLIER: ViewFrame = ViewFrame {
        vp: IDENTITY,
        elapsed: 4.25,
        cam_pos: [1.0, 2.0, 2.5],
    };

    #[test]
    fn the_first_motion_frame_reprojects_through_its_own_view() {
        assert_eq!(ViewHistory::default().prev_or(CUR), CUR);
    }

    #[test]
    fn an_advanced_history_returns_the_recorded_view_and_clock() {
        let mut history = ViewHistory::default();
        history.advance(EARLIER);
        assert_eq!(history.prev_or(CUR), EARLIER);
    }

    // A rebased history sees a carried point at the clip position the
    // recorded frame saw the point at before it was carried.
    #[test]
    fn a_rebased_history_reprojects_a_still_point_where_it_was() {
        use crate::planet::Rebase;
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([1.0, 0.0, 0.0], 0.02),
            translation: [-900.0, 4.0, 10.0],
        };
        let mut history = ViewHistory::default();
        history.advance(CUR);
        history.rebase(&rebase);
        let prev = history.prev_or(EARLIER);
        let p = [905.0, -3.0, -12.0];
        let clip = |m: Mat4, p: [f32; 3]| -> [f32; 4] {
            core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
        };
        let before = clip(CUR.vp, p);
        let after = clip(prev.vp, rebase.apply_point(p));
        for i in 0..4 {
            assert!((before[i] - after[i]).abs() < 1e-2, "{before:?} {after:?}");
        }
        let cam = rebase.apply_point(CUR.cam_pos);
        assert!((0..3).all(|i| (prev.cam_pos[i] - cam[i]).abs() < 1e-3));
        // Nothing recorded, nothing to carry.
        let mut empty = ViewHistory::default();
        empty.rebase(&rebase);
        assert_eq!(empty.prev_or(CUR), CUR);
    }

    #[test]
    fn a_reset_history_starts_over() {
        let mut history = ViewHistory::default();
        history.advance(EARLIER);
        history.reset();
        assert_eq!(history.prev_or(CUR), CUR);
    }

    // A transform recorded before two rebases, carried by the history's move,
    // places a point where both rebases put it; the next frame recorded
    // after them carries nothing.
    #[test]
    fn the_carry_composes_the_rebases_since_the_previous_frame() {
        use crate::planet::Rebase;
        let first = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.0, 0.0, 1.0], 0.03),
            translation: [-1_000.0, 2.0, 0.0],
        };
        let second = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([1.0, 0.0, 0.0], -0.02),
            translation: [0.0, -5.0, 1_000.0],
        };
        let mut history = ViewHistory::default();
        assert_eq!(history.carry(), IDENTITY);
        history.advance(CUR);
        history.rebase(&first);
        history.rebase(&second);
        let p = [1_003.0, 7.0, -998.0];
        let carry = history.carry();
        let carried: [f32; 3] = core::array::from_fn(|i| {
            carry[0][i] * p[0] + carry[1][i] * p[1] + carry[2][i] * p[2] + carry[3][i]
        });
        let want = second.apply_point(first.apply_point(p));
        assert!(
            (0..3).all(|i| (carried[i] - want[i]).abs() < 1e-2),
            "{carried:?} {want:?}"
        );
        history.advance(EARLIER);
        assert_eq!(history.carry(), IDENTITY);
        history.rebase(&first);
        history.reset();
        assert_eq!(history.carry(), IDENTITY);
    }
}
