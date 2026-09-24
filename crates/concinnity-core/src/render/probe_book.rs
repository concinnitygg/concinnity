//! The bookkeeping of a renderer's reflection probes, shared by every backend:
//! where each probe is placed, the record of every probe whose cube has been
//! baked, and the queue handing placements to the staggered bake. The backends
//! hold only the GPU resources (the cube array and the record buffers).

use alloc::format;
use alloc::vec::Vec;
use core::fmt;

use crate::render::error::{RenderError, RenderResult};
use crate::render::reflection_probe::{PrefilterPlan, ProbeBakeQueue, ProbePlacement};
use crate::render::uniforms::{ProbeSet, ProbeUniforms};

/// A world's reflection probes: the placements, one parallax record per
/// installed probe, and the bake queue over the placements.
///
/// Installs run in placement order, so record `i` (and cube `i` of the array
/// the backend bakes into) always describes placement `i`. The shaders read
/// the first [`ProbeBook::count`] records, so a probe that has not been baked
/// yet reflects the sky.
pub struct ProbeBook {
    placements: Vec<ProbePlacement>,
    records: Vec<ProbeUniforms>,
    queue: ProbeBakeQueue,
}

/// How far the bake has come: probes installed out of probes placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeProgress {
    /// Probes whose cube is baked and whose record the shaders read.
    pub installed: usize,
    /// Probes placed.
    pub placed: usize,
}

impl fmt::Display for ProbeProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "baked {}/{}", self.installed, self.placed)
    }
}

impl Default for ProbeBook {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbeBook {
    /// No placements, so reflections read the sky.
    pub fn new() -> ProbeBook {
        ProbeBook {
            placements: Vec::new(),
            records: Vec::new(),
            queue: ProbeBakeQueue::new(0),
        }
    }

    /// Replace the placements, forgetting every installed probe and queueing
    /// every placement to bake. An empty list leaves reflections on the sky.
    pub fn reset(&mut self, placements: Vec<ProbePlacement>) {
        self.records.clear();
        self.queue = ProbeBakeQueue::new(placements.len());
        self.placements = placements;
    }

    /// The next placement to bake and its index, advancing the queue. `None`
    /// once every placement has been handed out.
    pub fn take_next(&mut self) -> Option<(usize, ProbePlacement)> {
        self.queue
            .take_next()
            .map(|index| (index, self.placements[index]))
    }

    /// Whether any placement is still waiting to bake.
    pub fn pending(&self) -> bool {
        self.queue.pending()
    }

    /// Abandon every placement not yet handed out, keeping what installed.
    pub fn abort(&mut self) {
        self.queue.abort();
    }

    /// Install the probe at `index`, whose cube has finished baking, so the
    /// shaders read its record. Only the next probe in placement order can
    /// install: anything else would pair a record with another probe's cube.
    pub fn install(&mut self, index: usize) -> RenderResult<ProbeProgress> {
        let next = self.records.len();
        let placement = self.placements.get(index).filter(|_| index == next);
        let placement = placement.ok_or_else(|| {
            RenderError::Other(format!(
                "reflection probes: probe {index} baked while probe {next} of {} was next",
                self.placements.len()
            ))
        })?;
        self.records.push(placement.uniforms());
        Ok(self.progress())
    }

    /// Installed probes: the live count the shaders read.
    pub fn count(&self) -> usize {
        self.records.len()
    }

    // Placed probes, baked or not.
    fn placed(&self) -> usize {
        self.placements.len()
    }

    // Installed out of placed.
    fn progress(&self) -> ProbeProgress {
        ProbeProgress {
            installed: self.count(),
            placed: self.placed(),
        }
    }

    /// The record of every installed probe, in placement order.
    pub fn records(&self) -> &[ProbeUniforms] {
        &self.records
    }

    /// The header the shaders read the live count and the cubes' mip count
    /// from.
    pub fn header(&self) -> ProbeSet {
        ProbeSet::new(self.count(), PrefilterPlan::RUNTIME.mips())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement(x: f32) -> ProbePlacement {
        ProbePlacement::from_center_extents([x, 0.0, 0.0], [1.0; 3])
    }

    fn placed(n: usize) -> ProbeBook {
        let mut book = ProbeBook::new();
        book.reset((0..n).map(|i| placement(i as f32)).collect());
        book
    }

    fn bake_all(book: &mut ProbeBook) {
        while let Some((index, _)) = book.take_next() {
            book.install(index).unwrap();
        }
    }

    #[test]
    fn an_empty_book_reads_the_sky() {
        let mut book = ProbeBook::new();
        assert_eq!(book.count(), 0);
        assert_eq!(book.header().count, 0);
        assert!(!book.pending());
        assert!(book.take_next().is_none());
    }

    // Each record describes the placement at its own index, which is the cube
    // the backend baked it into.
    #[test]
    fn records_stay_aligned_with_placements() {
        let mut book = placed(4);
        bake_all(&mut book);
        assert_eq!(book.count(), 4);
        for (i, record) in book.records().iter().enumerate() {
            assert_eq!(record.probe_pos[0], i as f32);
            assert_eq!(record.box_min[3], 1.0, "record {i} enables parallax");
        }
        let header = book.header();
        assert_eq!(header.count, 4);
        assert_eq!(header.mip_count, PrefilterPlan::RUNTIME.mips());
    }

    #[test]
    fn take_next_hands_out_placements_in_order() {
        let mut book = placed(3);
        let handed: Vec<(usize, f32)> = core::iter::from_fn(|| book.take_next())
            .map(|(i, p)| (i, p.position[0]))
            .collect();
        assert_eq!(handed, [(0, 0.0), (1, 1.0), (2, 2.0)]);
        assert!(!book.pending());
    }

    // An install out of order would pair one probe's record with another's
    // cube, so it is refused and leaves the book as it was.
    #[test]
    fn an_install_out_of_order_is_refused() {
        let mut book = placed(3);
        book.take_next();
        book.take_next();
        assert!(book.install(1).is_err());
        assert!(book.install(7).is_err());
        assert_eq!(book.count(), 0);
        book.install(0).unwrap();
        assert!(book.install(0).is_err(), "a probe installs once");
        assert_eq!(book.install(1).unwrap().installed, 2);
    }

    #[test]
    fn install_reports_progress() {
        let mut book = placed(2);
        book.take_next();
        let progress = book.install(0).unwrap();
        assert_eq!(
            progress,
            ProbeProgress {
                installed: 1,
                placed: 2
            }
        );
        assert_eq!(format!("{progress}"), "baked 1/2");
    }

    // An abort keeps what installed and hands nothing more out.
    #[test]
    fn abort_keeps_the_installed_probes() {
        let mut book = placed(3);
        let (index, _) = book.take_next().unwrap();
        book.install(index).unwrap();
        book.abort();
        assert!(!book.pending());
        assert!(book.take_next().is_none());
        assert_eq!(book.count(), 1);
        assert_eq!(book.placed(), 3);
    }

    // A re-placement forgets the installed probes and queues the new list.
    #[test]
    fn reset_forgets_installs_and_requeues() {
        let mut book = placed(2);
        bake_all(&mut book);
        book.reset(alloc::vec![placement(9.0)]);
        assert_eq!(book.count(), 0);
        assert_eq!(book.placed(), 1);
        assert_eq!(
            book.take_next().map(|(i, p)| (i, p.position[0])),
            Some((0, 9.0))
        );
    }

    // A backend that cannot make room for the cubes resets to nothing, which
    // keeps the sky: nothing placed and nothing queued.
    #[test]
    fn a_reset_to_nothing_keeps_the_sky() {
        let mut book = placed(2);
        bake_all(&mut book);
        book.reset(Vec::new());
        assert_eq!(book.count(), 0);
        assert_eq!(book.placed(), 0);
        assert!(!book.pending());
        assert!(book.take_next().is_none());
    }
}
