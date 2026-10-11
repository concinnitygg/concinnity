//! When the atmosphere's per-frame tables can be read outside the frame that
//! wrote them.
//!
//! The sky-view table and the irradiance cube are written by the frame's
//! atmosphere pass. A pass inside that frame is ordered after it by the frame
//! graph, but work the renderer records apart from the frame, such as a
//! reflection-probe face, is not: before the first frame has run the pass the
//! tables hold nothing, and a capture would bake a black sky. Once a frame
//! that ran the pass has retired, every later submission reads written
//! tables, whatever queue it runs on.

use core::sync::atomic::{AtomicU32, Ordering};

/// Counts the frames that recorded the atmosphere pass, answering whether the
/// first of them has retired.
#[derive(Debug, Default)]
pub struct TableReadiness {
    recorded: AtomicU32,
}

impl TableReadiness {
    /// Note that this frame records the atmosphere pass.
    pub fn note_recorded(&self) {
        // Saturating: a long run never wraps back to "not ready".
        let _ = self
            .recorded
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1));
    }

    /// Whether the tables hold this atmosphere's sky for work recorded now,
    /// outside the frame graph. A renderer that has waited for the frame
    /// `frames_in_flight` frames back knows every frame recorded at least that
    /// many frames ago has retired. Call it before this frame records the pass.
    pub fn ready(&self, frames_in_flight: usize) -> bool {
        let recorded = self.recorded.load(Ordering::Relaxed) as usize;
        recorded >= frames_in_flight.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tables_are_not_ready_before_any_frame_records_them() {
        let tables = TableReadiness::default();
        assert!(!tables.ready(2));
        assert!(!tables.ready(0));
    }

    #[test]
    fn the_tables_are_ready_once_the_first_recording_frame_retired() {
        // Frame 0 records the pass; with two frames in flight, frame 2 is the
        // first to have waited for frame 0.
        let tables = TableReadiness::default();
        for frame in 0..4 {
            let ready = tables.ready(2);
            assert_eq!(ready, frame >= 2, "frame {frame}");
            tables.note_recorded();
        }
    }

    #[test]
    fn a_single_frame_in_flight_still_waits_for_one_frame() {
        let tables = TableReadiness::default();
        assert!(!tables.ready(1));
        tables.note_recorded();
        assert!(tables.ready(1));
    }
}
