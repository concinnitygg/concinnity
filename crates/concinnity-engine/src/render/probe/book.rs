//! The bookkeeping of a renderer's reflection probes, shared by every backend:
//! where each probe is placed, the record of every probe whose cube has been
//! baked, and the queue handing placements to the staggered bake. The backends
//! hold only the GPU resources (the cube array and the record buffers).

use alloc::format;
use alloc::vec::Vec;
use core::fmt;

use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::probe::bake::ProbeBakeQueue;
use crate::render::probe::capture::{PrefilterPlan, ProbePlacement};
use crate::render::probe::prop::PropProbeFrame;
use crate::render::uniforms::{ProbeSet, ProbeUniforms};

/// A world's reflection probes: the placements, one parallax record per
/// installed probe, the bake queue over the placements, and the prop probe's
/// record after them with the object records of its prop.
///
/// Installs run in placement order, so record `i` (and cube `i` of the array
/// the backend bakes into) always describes placement `i`. The shaders read
/// the first [`ProbeBook::count`] records, so a probe that has not been baked
/// yet reflects the sky. The prop probe's cube is the one after the
/// placements', and its record joins once every placement has installed.
pub struct ProbeBook {
    placements: Vec<ProbePlacement>,
    records: Vec<ProbeUniforms>,
    queue: ProbeBakeQueue,
    prop_probe: bool,
    prop_subject: [u32; 2],
    prop_ground: [f32; 4],
    revision: u64,
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
    pub fn new() -> Self {
        Self {
            placements: Vec::new(),
            records: Vec::new(),
            queue: ProbeBakeQueue::new(0),
            prop_probe: false,
            prop_subject: [0; 2],
            prop_ground: [0.0; 4],
            revision: 0,
        }
    }

    /// Replace the placements, forgetting every installed probe and queueing
    /// every placement to bake, with a cube for a prop probe after theirs when
    /// `prop_probe`. An empty list leaves reflections on the sky.
    pub fn reset(&mut self, placements: Vec<ProbePlacement>, prop_probe: bool) {
        self.records.clear();
        self.queue = ProbeBakeQueue::new(placements.len());
        self.placements = placements;
        self.prop_probe = prop_probe;
        self.revision += 1;
    }

    /// The cube of the array the prop probe is captured into, after every
    /// placement's; `None` when the placements left it none.
    pub fn prop_cube(&self) -> Option<usize> {
        self.prop_probe.then_some(self.placements.len())
    }

    /// Set the prop probe's record from `frame`, which the shaders read once
    /// every placement has installed; `None` withdraws it. Changes the
    /// revision only when the records the shaders read change.
    pub fn set_prop_probe(&mut self, frame: Option<&PropProbeFrame>) {
        let Some(cube) = self.prop_cube() else {
            return;
        };
        if self.records.len() < cube {
            return;
        }
        self.prop_subject = frame.map_or([0; 2], |f| f.records);
        self.prop_ground = frame.map_or([0.0; 4], |f| f.ground);
        let record = frame.map(PropProbeFrame::record);
        let current = self.records.get(cube).copied();
        match (current, record) {
            (Some(old), Some(new)) if bytemuck::bytes_of(&old) == bytemuck::bytes_of(&new) => {
                return;
            }
            (Some(_), Some(new)) => self.records[cube] = new,
            (None, Some(new)) => self.records.push(new),
            (Some(_), None) => {
                self.records.truncate(cube);
            }
            (None, None) => return,
        }
        self.revision += 1;
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
    pub(crate) fn install(&mut self, index: usize) -> RenderResult<ProbeProgress> {
        let next = self.records.len();
        let placement = self.placements.get(index).filter(|_| index == next);
        let placement = placement.ok_or_else(|| {
            RenderError::Other(format!(
                "reflection probes: probe {index} baked while probe {next} of {} was next",
                self.placements.len()
            ))
        })?;
        self.records.push(placement.uniforms());
        self.revision += 1;
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

    /// Changes whenever [`Self::records`] does, so a backend can skip
    /// rewriting a copy of them that is already current.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The header the shaders read the live count and the cubes' mip count
    /// from, with the prop probe's prop and ground while its record is live.
    pub fn header(&self) -> ProbeSet {
        let set = ProbeSet::new(self.count(), PrefilterPlan::RUNTIME.mips());
        let live = self
            .prop_cube()
            .is_some_and(|cube| self.records.len() > cube);
        if !live {
            return set;
        }
        ProbeSet {
            subject: self.prop_subject,
            ground: self.prop_ground,
            ..set
        }
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
        book.reset((0..n).map(|i| placement(i as f32)).collect(), false);
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
        book.reset(alloc::vec![placement(9.0)], false);
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
        book.reset(Vec::new(), false);
        assert_eq!(book.count(), 0);
        assert_eq!(book.placed(), 0);
        assert!(!book.pending());
        assert!(book.take_next().is_none());
    }

    fn prop_frame(x: f32) -> PropProbeFrame {
        PropProbeFrame {
            bounds_min: [x - 1.0, -1.0, -1.0],
            bounds_max: [x + 1.0, 1.0, 1.0],
            ground: [0.0, 1.0, 0.0, -1.0],
            records: [4, 6],
            resolution: 128,
            capture_distance: None,
            sea: false,
        }
    }

    // The prop probe's cube follows the placements', so its record waits for
    // every placement to install and then rides after theirs.
    #[test]
    fn the_prop_record_joins_after_every_placement() {
        let mut book = ProbeBook::new();
        book.reset(alloc::vec![placement(0.0), placement(1.0)], true);
        assert_eq!(book.prop_cube(), Some(2));
        book.set_prop_probe(Some(&prop_frame(5.0)));
        assert_eq!(book.count(), 0, "the placements bake first");
        bake_all(&mut book);
        book.set_prop_probe(Some(&prop_frame(5.0)));
        assert_eq!(book.count(), 3);
        assert_eq!(book.records()[2].probe_pos[0], 5.0);
        book.set_prop_probe(Some(&prop_frame(6.0)));
        assert_eq!(book.count(), 3);
        assert_eq!(book.records()[2].probe_pos[0], 6.0);
        book.set_prop_probe(None);
        assert_eq!(book.count(), 2);
        assert_eq!(book.records()[1].probe_pos[0], 1.0);
    }

    // The header names the prop's records and ground only while its record
    // is live, so a world without one reads an empty range and no plane.
    #[test]
    fn the_header_names_the_prop_only_while_its_record_is_live() {
        let mut book = ProbeBook::new();
        book.reset(alloc::vec![placement(0.0)], true);
        book.set_prop_probe(Some(&prop_frame(5.0)));
        assert_eq!(book.header().subject, [0, 0], "the placement bakes first");
        assert_eq!(book.header().ground, [0.0; 4]);
        bake_all(&mut book);
        book.set_prop_probe(Some(&prop_frame(5.0)));
        assert_eq!(book.header().subject, [4, 6]);
        assert_eq!(book.header().ground, [0.0, 1.0, 0.0, -1.0]);
        book.set_prop_probe(None);
        assert_eq!(book.header().subject, [0, 0]);
        assert_eq!(book.header().ground, [0.0; 4]);
        assert_eq!(placed(1).header().subject, [0, 0]);
    }

    #[test]
    fn a_world_without_a_prop_probe_takes_no_prop_record() {
        let mut book = placed(1);
        bake_all(&mut book);
        assert_eq!(book.prop_cube(), None);
        book.set_prop_probe(Some(&prop_frame(5.0)));
        assert_eq!(book.count(), 1);
    }

    // The records are rewritten only when they change: a prop at rest costs
    // no buffer write.
    #[test]
    fn an_unchanged_prop_record_keeps_the_revision() {
        let mut book = ProbeBook::new();
        book.reset(Vec::new(), true);
        book.set_prop_probe(Some(&prop_frame(1.0)));
        let joined = book.revision();
        book.set_prop_probe(Some(&prop_frame(1.0)));
        assert_eq!(book.revision(), joined);
        book.set_prop_probe(Some(&prop_frame(2.0)));
        assert_ne!(book.revision(), joined);
        let moved = book.revision();
        book.set_prop_probe(None);
        book.set_prop_probe(None);
        assert_eq!(book.revision(), moved + 1);
    }

    // A backend skips its copy of the records while the revision holds, so it
    // must move with every change to them and with nothing else.
    #[test]
    fn the_revision_moves_exactly_when_the_records_do() {
        let mut book = ProbeBook::new();
        let start = book.revision();
        book.reset(alloc::vec![placement(0.0), placement(1.0)], false);
        let placed = book.revision();
        assert_ne!(placed, start);
        let (index, _) = book.take_next().unwrap();
        assert_eq!(
            book.revision(),
            placed,
            "handing out a bake changes no record"
        );
        book.install(index).unwrap();
        let installed = book.revision();
        assert_ne!(installed, placed);
        assert!(book.install(0).is_err());
        assert_eq!(
            book.revision(),
            installed,
            "a refused install changes nothing"
        );
        book.abort();
        assert_eq!(book.revision(), installed, "an abort keeps every record");
    }
}
