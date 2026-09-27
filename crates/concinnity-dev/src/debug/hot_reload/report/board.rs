//! The latest reload outcome of every reload subject, shared with an editor
//! session so its panels can show what became of a save.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, PoisonError};

use super::{ReloadOutcome, ReloadReport, ReloadSubject};

// One subject's latest outcome, stamped with the board sequence it arrived at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Latest {
    pub seq: u64,
    pub outcome: ReloadOutcome,
}

// What the handle holds. Every change takes the next sequence number, so a
// reader holding the last number it saw can tell what is new.
#[derive(Debug, Clone, Default)]
pub(crate) struct ReportBoard {
    seq: u64,
    armed_at: Option<u64>,
    live: BTreeSet<ReloadSubject>,
    latest: BTreeMap<ReloadSubject, Latest>,
}

impl ReportBoard {
    // The newest outcome reported for `subject` since the catalogs were last
    // armed.
    pub(crate) fn latest(&self, subject: &ReloadSubject) -> Option<&Latest> {
        self.latest.get(subject)
    }

    // The sequence the current catalogs were armed at, or `None` before the
    // first arm.
    pub(crate) fn armed_at(&self) -> Option<u64> {
        self.armed_at
    }

    // Whether `subject` is in the armed catalogs, so a save of its files
    // recompiles it. `None` before the first arm.
    pub(crate) fn is_live(&self, subject: &ReloadSubject) -> Option<bool> {
        self.armed_at.map(|_| self.live.contains(subject))
    }

    fn next(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }
}

// A cloneable handle to the board: the hot-reload driver writes it, the editor
// reads it.
#[derive(Debug, Clone, Default)]
pub(crate) struct ReloadReports(Arc<Mutex<ReportBoard>>);

impl ReloadReports {
    // Catalogs were armed over `subjects`. The world they were captured from
    // built every pipeline from the files as they are on disk, so the outcomes
    // reported against the previous catalogs are dropped.
    pub(crate) fn arm(&self, subjects: impl IntoIterator<Item = ReloadSubject>) {
        let mut board = self.lock();
        let seq = board.next();
        board.armed_at = Some(seq);
        board.live = subjects.into_iter().collect();
        board.latest.clear();
    }

    // Record each report as its subject's latest outcome, in order.
    pub(crate) fn publish(&self, reports: &[ReloadReport]) {
        let mut board = self.lock();
        for report in reports {
            let seq = board.next();
            board.latest.insert(
                report.subject.clone(),
                Latest {
                    seq,
                    outcome: report.outcome.clone(),
                },
            );
        }
    }

    // The sequence of the board's latest change.
    pub(crate) fn seq(&self) -> u64 {
        self.lock().seq
    }

    // A copy of the board as it stands.
    pub(crate) fn snapshot(&self) -> ReportBoard {
        self.lock().clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ReportBoard> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::super::ReloadFailure;
    use super::*;

    fn report(subject: ReloadSubject, outcome: ReloadOutcome) -> ReloadReport {
        ReloadReport { subject, outcome }
    }

    fn failed(why: &str) -> ReloadOutcome {
        ReloadOutcome::Failed(ReloadFailure::Unstarted(why.to_string()))
    }

    fn applies() -> ReloadOutcome {
        ReloadOutcome::AppliesOnLoad {
            warnings: Vec::new(),
        }
    }

    fn shader(name: &str) -> ReloadSubject {
        ReloadSubject::shader(name)
    }

    // Each subject keeps only its newest outcome, and a newer report carries a
    // higher sequence than anything already on the board.
    #[test]
    fn each_subject_keeps_its_newest_outcome() {
        let reports = ReloadReports::default();
        let reader = reports.clone();
        reports.arm([shader("lit"), shader("water")]);
        let armed = reader.seq();
        reports.publish(&[
            report(shader("lit"), failed("first")),
            report(shader("water"), applies()),
        ]);
        let before = reader.snapshot().latest(&shader("lit")).unwrap().seq;
        reports.publish(&[report(shader("lit"), applies())]);
        let board = reader.snapshot();
        let lit = board.latest(&shader("lit")).unwrap();
        assert_eq!(lit.outcome, applies());
        assert!(lit.seq > before);
        assert!(lit.seq > board.latest(&shader("water")).unwrap().seq);
        assert!(board.latest(&shader("cave")).is_none());
        assert_eq!(reader.seq(), lit.seq, "the latest change");
        assert!(reader.seq() > armed);
    }

    // Arming fresh catalogs drops the old outcomes and names what is live.
    #[test]
    fn arming_starts_a_fresh_board() {
        let reports = ReloadReports::default();
        assert_eq!(reports.snapshot().is_live(&shader("lit")), None);
        assert_eq!(reports.snapshot().armed_at(), None);
        reports.arm([shader("lit")]);
        reports.publish(&[report(shader("lit"), failed("broken"))]);
        let published = reports.snapshot().latest(&shader("lit")).unwrap().seq;
        reports.arm([shader("lit"), shader("new")]);
        let board = reports.snapshot();
        assert!(board.latest(&shader("lit")).is_none());
        assert!(board.armed_at().unwrap() > published);
        assert_eq!(board.is_live(&shader("new")), Some(true));
        assert_eq!(board.is_live(&shader("gone")), Some(false));
    }

    // A Shader and an SdfVolume may share a name without sharing a row.
    #[test]
    fn a_shader_and_a_volume_of_one_name_are_two_subjects() {
        let reports = ReloadReports::default();
        let volume = ReloadSubject::sdf_volume("water");
        reports.arm([shader("water"), volume.clone()]);
        reports.publish(&[report(volume.clone(), failed("broken"))]);
        let board = reports.snapshot();
        assert!(board.latest(&shader("water")).is_none());
        assert_eq!(board.latest(&volume).unwrap().outcome, failed("broken"));
        assert_eq!(board.is_live(&shader("water")), Some(true));
    }
}
