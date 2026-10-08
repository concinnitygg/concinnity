//! The camera half of the motion-vector history: the previous frame's
//! unjittered view-projection, which the G-buffer pre-pass reprojects the sky
//! and every surface through, and the clock and camera position that frame's
//! vertex hooks read.
//! The per-object half is the model-history ring (see
//! [`crate::render::model_history`]).

use crate::transform::Mat4;

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
/// one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewHistory {
    prev: Option<ViewFrame>,
}

impl ViewHistory {
    /// The frame this frame's motion reprojects to. A view with no history yet
    /// reprojects to itself, so its first motion frame reads no camera or
    /// clock motion rather than reprojecting through a placeholder.
    pub fn prev_or(&self, cur: ViewFrame) -> ViewFrame {
        self.prev.unwrap_or(cur)
    }

    /// Record this frame as the next frame's history.
    pub fn advance(&mut self, cur: ViewFrame) {
        self.prev = Some(cur);
    }

    /// Forget the history, so the next motion frame starts over.
    pub fn reset(&mut self) {
        self.prev = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::IDENTITY;

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

    #[test]
    fn a_reset_history_starts_over() {
        let mut history = ViewHistory::default();
        history.advance(EARLIER);
        history.reset();
        assert_eq!(history.prev_or(CUR), CUR);
    }
}
