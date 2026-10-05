//! The camera half of the motion-vector history: the previous frame's
//! unjittered view-projection, which the G-buffer pre-pass reprojects the sky
//! and every surface through. The per-object half is the model-history ring
//! (see [`crate::render::model_history`]).

use crate::transform::Mat4;

/// The previous frame's unjittered view-projection, absent until a frame has
/// recorded one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewHistory {
    prev: Option<Mat4>,
}

impl ViewHistory {
    /// The view-projection this frame's motion reprojects through. A view with
    /// no history yet reprojects through its own, so its first motion frame
    /// reads no camera motion rather than reprojecting through a placeholder.
    pub fn prev_or(&self, cur: Mat4) -> Mat4 {
        self.prev.unwrap_or(cur)
    }

    /// Record this frame's view-projection as the next frame's history.
    pub fn advance(&mut self, cur: Mat4) {
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

    const CUR: Mat4 = [
        [1.2, 0.0, 0.0, 0.0],
        [0.0, 2.1, 0.0, 0.0],
        [0.3, 0.1, -1.0, -1.0],
        [0.0, 0.0, -0.2, 0.0],
    ];

    #[test]
    fn the_first_motion_frame_reprojects_through_its_own_view() {
        assert_eq!(ViewHistory::default().prev_or(CUR), CUR);
    }

    #[test]
    fn an_advanced_history_returns_the_recorded_view() {
        let mut history = ViewHistory::default();
        history.advance(IDENTITY);
        assert_eq!(history.prev_or(CUR), IDENTITY);
    }

    #[test]
    fn a_reset_history_starts_over() {
        let mut history = ViewHistory::default();
        history.advance(IDENTITY);
        history.reset();
        assert_eq!(history.prev_or(CUR), CUR);
    }
}
