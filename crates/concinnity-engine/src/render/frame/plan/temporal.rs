//! What one frame hands the next: the jitter phase, the camera the motion
//! vectors reproject to, and the view the occlusion test reprojects last
//! frame's pyramid through.

use crate::planet::Rebase;
use crate::render::temporal::occlusion_history::OcclusionHistory;
use crate::render::temporal::view_history::{ViewFrame, ViewHistory};
use crate::transform::Mat4;

/// The temporal state every backend's frame reads and advances.
#[derive(Debug, Default)]
pub struct TemporalState {
    jitter_phase: u32,
    view: ViewHistory,
    occlusion: OcclusionHistory,
}

impl TemporalState {
    /// The phase this frame's jitter is taken at.
    pub fn jitter_phase(&self) -> u32 {
        self.jitter_phase
    }

    /// The camera the motion vectors reproject to, or `cur` with no history.
    pub fn prev_view_or(&self, cur: ViewFrame) -> ViewFrame {
        self.view.prev_or(cur)
    }

    /// The move that carries the previous frame's model transforms into this
    /// frame, identity unless the world was rebased since.
    pub fn prev_model_carry(&self) -> Mat4 {
        self.view.carry()
    }

    /// The occlusion test's history.
    pub fn occlusion(&self) -> &OcclusionHistory {
        &self.occlusion
    }

    /// Drop the occlusion history, for a pyramid that no longer matches the
    /// view (a resize).
    pub fn invalidate_occlusion(&mut self) {
        self.occlusion.reset();
    }

    /// Drop every history for a frame that does not continue the last. The
    /// jitter keeps its phase.
    pub fn reset(&mut self) {
        self.view.reset();
        self.occlusion.reset();
    }

    /// Carry every history into the frame the world moved to.
    pub fn rebase(&mut self, rebase: &Rebase) {
        self.view.rebase(rebase);
        self.occlusion.rebase(rebase);
    }

    /// Close frame `cur`. The occlusion test reprojects through it next frame
    /// once a pyramid was built. While a pass reads motion the camera history
    /// and the jitter advance with it; otherwise the camera history restarts
    /// with the per-object half, which the pre-pass stops filling.
    pub fn advance(&mut self, cur: ViewFrame, motion_read: bool, pyramid: bool) {
        self.occlusion.advance(cur.vp, pyramid);
        if motion_read {
            self.view.advance(cur);
            self.jitter_phase = self.jitter_phase.wrapping_add(1);
        } else {
            self.view.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::IDENTITY;

    fn frame(elapsed: f32) -> ViewFrame {
        ViewFrame {
            vp: IDENTITY,
            elapsed,
            cam_pos: [0.0; 3],
        }
    }

    #[test]
    fn motion_advances_the_jitter_and_the_camera_history_together() {
        let mut t = TemporalState::default();
        t.advance(frame(1.0), true, true);
        assert_eq!(t.jitter_phase(), 1);
        assert_eq!(t.prev_view_or(frame(2.0)).elapsed, 1.0);
        assert!(t.occlusion().valid());
    }

    #[test]
    fn a_frame_without_motion_holds_the_jitter_and_drops_the_camera() {
        let mut t = TemporalState::default();
        t.advance(frame(1.0), true, false);
        t.advance(frame(2.0), false, false);
        assert_eq!(t.jitter_phase(), 1);
        assert_eq!(t.prev_view_or(frame(3.0)).elapsed, 3.0);
        assert!(!t.occlusion().valid());
    }

    #[test]
    fn a_reset_keeps_the_jitter_phase() {
        let mut t = TemporalState::default();
        t.advance(frame(1.0), true, true);
        t.reset();
        assert_eq!(t.jitter_phase(), 1);
        assert_eq!(t.prev_view_or(frame(2.0)).elapsed, 2.0);
        assert!(!t.occlusion().valid());
    }
}
