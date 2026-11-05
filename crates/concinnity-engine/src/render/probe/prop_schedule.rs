//! When a backend captures the prop probe: the frame it captures this frame,
//! and the failure that stops it for good. Every backend keeps one beside the
//! GPU half it builds.

use core::fmt::Display;

use crate::render::probe::book::ProbeBook;
use crate::render::probe::prop::PropProbeFrame;

/// The prop probe's capture schedule: where this frame captures, and whether
/// building the capture has failed, which is not retried.
#[derive(Clone, Copy, Debug, Default)]
pub struct PropProbeSchedule {
    frame: Option<PropProbeFrame>,
    failed: bool,
}

impl PropProbeSchedule {
    /// Where this frame captures; `None` on a frame without a capture.
    pub fn frame(&self) -> Option<PropProbeFrame> {
        self.frame
    }

    /// Whether this frame captures the probe.
    pub fn capturing(&self) -> bool {
        self.frame.is_some()
    }

    /// Whether a capture can still be built: none has failed and `book`
    /// keeps a cube for the probe.
    pub fn available(&self, book: &ProbeBook) -> bool {
        !self.failed && book.prop_cube().is_some()
    }

    /// Admit `frame` for capture when `ready` and [`Self::available`],
    /// returning the frame whose capture the backend builds next. Otherwise
    /// nothing is captured this frame and the probe's record is withdrawn.
    pub fn admit(
        &mut self,
        book: &mut ProbeBook,
        frame: Option<PropProbeFrame>,
        ready: bool,
    ) -> Option<PropProbeFrame> {
        let admitted = frame.filter(|_| ready && self.available(book));
        if admitted.is_none() {
            self.withdraw(book);
        }
        admitted
    }

    /// Settle an admitted `frame` on whether its capture was `built`: a built
    /// one captures this frame and its record goes live, and a failure
    /// withdraws the record for good. Returns whether the capture is usable;
    /// the backend drops what it built on `false`.
    pub fn settle<E: Display>(
        &mut self,
        book: &mut ProbeBook,
        frame: PropProbeFrame,
        built: Result<(), E>,
    ) -> bool {
        if let Err(e) = built {
            tracing::error!("reflection probes: the prop probe could not be built: {e}");
            self.failed = true;
            self.withdraw(book);
            return false;
        }
        self.frame = Some(frame);
        book.set_prop_probe(Some(&frame));
        true
    }

    fn withdraw(&mut self, book: &mut ProbeBook) {
        self.frame = None;
        book.set_prop_probe(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::probe::capture::ProbePlacement;

    fn frame() -> PropProbeFrame {
        PropProbeFrame {
            bounds_min: [-1.0; 3],
            bounds_max: [1.0; 3],
            ground: [0.0, 1.0, 0.0, -1.0],
            records: [4, 6],
            resolution: 128,
            capture_distance: None,
            sea: false,
        }
    }

    // A book whose placements have all installed, so the prop probe's record
    // is read once set.
    fn book(prop_probe: bool) -> ProbeBook {
        let mut book = ProbeBook::new();
        book.reset(
            alloc::vec![ProbePlacement::from_center_extents([0.0; 3], [1.0; 3])],
            prop_probe,
        );
        while let Some((index, _)) = book.take_next() {
            book.install(index).unwrap();
        }
        book
    }

    #[test]
    fn a_built_capture_goes_live() {
        let mut book = book(true);
        let mut schedule = PropProbeSchedule::default();
        let admitted = schedule.admit(&mut book, Some(frame()), true);
        assert_eq!(admitted, Some(frame()));
        assert!(schedule.settle(&mut book, frame(), Ok::<(), &str>(())));
        assert_eq!(schedule.frame(), Some(frame()));
        assert_eq!(book.count(), 2);
    }

    #[test]
    fn a_frame_without_the_probe_withdraws_its_record() {
        let mut book = book(true);
        let mut schedule = PropProbeSchedule::default();
        schedule.admit(&mut book, Some(frame()), true);
        schedule.settle(&mut book, frame(), Ok::<(), &str>(()));
        assert_eq!(schedule.admit(&mut book, None, true), None);
        assert!(!schedule.capturing());
        assert_eq!(book.count(), 1);
    }

    #[test]
    fn a_backend_that_is_not_ready_captures_nothing() {
        let mut book = book(true);
        let mut schedule = PropProbeSchedule::default();
        assert_eq!(schedule.admit(&mut book, Some(frame()), false), None);
        assert!(!schedule.capturing());
    }

    #[test]
    fn a_book_without_a_prop_cube_captures_nothing() {
        let mut book = book(false);
        let mut schedule = PropProbeSchedule::default();
        assert!(!schedule.available(&book));
        assert_eq!(schedule.admit(&mut book, Some(frame()), true), None);
    }

    // A capture that cannot be built is not retried on later frames.
    #[test]
    fn a_failed_build_stops_the_probe_for_good() {
        let mut book = book(true);
        let mut schedule = PropProbeSchedule::default();
        schedule.admit(&mut book, Some(frame()), true);
        assert!(!schedule.settle(&mut book, frame(), Err("out of memory")));
        assert!(!schedule.capturing());
        assert!(!schedule.available(&book));
        assert_eq!(book.count(), 1);
        assert_eq!(schedule.admit(&mut book, Some(frame()), true), None);
    }
}
