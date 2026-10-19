// The frames a run records, stored in fixed-size blocks.

use alloc::vec::Vec;

use crate::frame_report::sample::FrameSample;

// Samples per block, about 5 MB. A full block is never grown: the next frame
// starts a new one, so recording never copies what it has already recorded.
const BLOCK_SAMPLES: usize = 4096;

/// The frames of a run, in the order they were drawn.
///
/// Held in fixed-size blocks rather than one growing buffer. A buffer that
/// doubles copies every sample it holds, tens of megabytes on a long run, and
/// does it inside the frame it is recording; the copy then reads as a hitch in
/// the run it measures, and holds the allocator's lock against every thread
/// that allocates meanwhile.
#[derive(Debug, Default)]
pub(super) struct SampleLog {
    blocks: Vec<Vec<FrameSample>>,
    len: usize,
}

impl SampleLog {
    /// An empty log. Nothing is allocated until the first frame.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Record one frame after the others.
    pub(super) fn push(&mut self, sample: &FrameSample) {
        match self.blocks.last_mut() {
            Some(block) if block.len() < BLOCK_SAMPLES => block.push(*sample),
            _ => {
                let mut block = Vec::with_capacity(BLOCK_SAMPLES);
                block.push(*sample);
                self.blocks.push(block);
            }
        }
        self.len += 1;
    }

    /// How many frames were recorded.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Whether no frame was recorded.
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Frame `index`, counting from the first recorded.
    #[cfg(test)]
    pub(super) fn get(&self, index: usize) -> Option<&FrameSample> {
        self.blocks
            .get(index / BLOCK_SAMPLES)
            .and_then(|block| block.get(index % BLOCK_SAMPLES))
    }

    /// The frames in the order they were drawn.
    pub(super) fn iter(&self) -> impl Iterator<Item = &FrameSample> {
        self.blocks.iter().flatten()
    }
}

impl FromIterator<FrameSample> for SampleLog {
    fn from_iter<I: IntoIterator<Item = FrameSample>>(samples: I) -> Self {
        let mut log = Self::new();
        for sample in samples {
            log.push(&sample);
        }
        log
    }
}

impl From<Vec<FrameSample>> for SampleLog {
    fn from(samples: Vec<FrameSample>) -> Self {
        samples.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_report::sample::FrameTiming;
    use crate::math::SaturatingFrom;
    use crate::profile::RenderStats;

    fn frame(n: usize) -> FrameSample {
        FrameSample::new(
            FrameTiming {
                run_seconds: n as f32,
                frame_us: u32::saturating_from(n),
                ..Default::default()
            },
            &RenderStats::default(),
            &[],
            &[],
            &[],
        )
    }

    #[test]
    fn frames_keep_their_order_across_blocks() {
        let log: SampleLog = (0..BLOCK_SAMPLES * 2 + 3).map(frame).collect();
        assert_eq!(log.len(), BLOCK_SAMPLES * 2 + 3);
        assert!(
            log.iter()
                .enumerate()
                .all(|(n, s)| s.frame_us == u32::saturating_from(n))
        );
        assert_eq!(
            log.get(BLOCK_SAMPLES).map(|s| s.frame_us),
            Some(u32::saturating_from(BLOCK_SAMPLES))
        );
        assert_eq!(log.get(BLOCK_SAMPLES * 2 + 3), None);
    }

    // The property the blocks exist for: a recorded frame is never copied.
    #[test]
    fn a_recorded_frame_never_moves() {
        let mut log = SampleLog::new();
        log.push(&frame(0));
        let first: *const FrameSample = log.get(0).unwrap();
        for n in 1..BLOCK_SAMPLES * 3 {
            log.push(&frame(n));
        }
        assert!(core::ptr::eq(first, log.get(0).unwrap()));
        assert_eq!(log.blocks.len(), 3);
    }

    #[test]
    fn an_empty_log_allocates_nothing() {
        let log = SampleLog::new();
        assert!(log.is_empty());
        assert_eq!(log.iter().count(), 0);
        assert!(log.blocks.is_empty());
    }
}
