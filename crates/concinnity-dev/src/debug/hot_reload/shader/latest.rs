//! The latest reload outcome of every world Shader, shared with an editor
//! session so its Shader panels can show what became of a save.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, PoisonError};

use super::{ShaderReloadOutcome, ShaderReloadReport};

// One Shader's latest outcome, stamped with the board sequence it arrived at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Latest {
    pub seq: u64,
    pub outcome: ShaderReloadOutcome,
}

// What the handle holds. Every change takes the next sequence number, so a
// reader holding the last number it saw can tell what is new.
#[derive(Debug, Clone, Default)]
pub(crate) struct ReportBoard {
    seq: u64,
    armed_at: Option<u64>,
    live: BTreeSet<String>,
    latest: BTreeMap<String, Latest>,
}

impl ReportBoard {
    // The newest outcome reported for the Shader `name` since the catalog was
    // last armed.
    pub(crate) fn latest(&self, name: &str) -> Option<&Latest> {
        self.latest.get(name)
    }

    // The sequence the current catalog was armed at, or `None` before the
    // first arm.
    pub(crate) fn armed_at(&self) -> Option<u64> {
        self.armed_at
    }

    // Whether the Shader `name` is in the armed catalog, so a save of its
    // files recompiles it. `None` before the first arm.
    pub(crate) fn is_live(&self, name: &str) -> Option<bool> {
        self.armed_at.map(|_| self.live.contains(name))
    }

    fn next(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }
}

// A cloneable handle to the board: the hot-reload driver writes it, the editor
// reads it.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShaderReports(Arc<Mutex<ReportBoard>>);

impl ShaderReports {
    // A catalog was armed over the Shaders `names`. The world it was captured
    // from built every pipeline from the files as they are on disk, so the
    // outcomes reported against the previous catalog are dropped.
    pub(crate) fn arm(&self, names: impl IntoIterator<Item = String>) {
        let mut board = self.lock();
        let seq = board.next();
        board.armed_at = Some(seq);
        board.live = names.into_iter().collect();
        board.latest.clear();
    }

    // Record each report as its Shader's latest outcome, in order.
    pub(crate) fn publish(&self, reports: &[ShaderReloadReport]) {
        let mut board = self.lock();
        for report in reports {
            let seq = board.next();
            board.latest.insert(
                report.name.clone(),
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
    use super::super::ShaderReloadFailure;
    use super::*;

    fn report(name: &str, outcome: ShaderReloadOutcome) -> ShaderReloadReport {
        ShaderReloadReport {
            name: name.to_string(),
            outcome,
        }
    }

    fn failed(why: &str) -> ShaderReloadOutcome {
        ShaderReloadOutcome::Failed(ShaderReloadFailure::Unstarted(why.to_string()))
    }

    fn applies() -> ShaderReloadOutcome {
        ShaderReloadOutcome::AppliesOnLoad {
            warnings: Vec::new(),
        }
    }

    // Each Shader keeps only its newest outcome, and a newer report carries a
    // higher sequence than anything already on the board.
    #[test]
    fn each_shader_keeps_its_newest_outcome() {
        let reports = ShaderReports::default();
        let reader = reports.clone();
        reports.arm(["lit".to_string(), "water".to_string()]);
        let armed = reader.seq();
        reports.publish(&[report("lit", failed("first")), report("water", applies())]);
        let before = reader.snapshot().latest("lit").unwrap().seq;
        reports.publish(&[report("lit", applies())]);
        let board = reader.snapshot();
        let lit = board.latest("lit").unwrap();
        assert_eq!(lit.outcome, applies());
        assert!(lit.seq > before);
        assert!(lit.seq > board.latest("water").unwrap().seq);
        assert!(board.latest("cave").is_none());
        assert_eq!(reader.seq(), lit.seq, "the latest change");
        assert!(reader.seq() > armed);
    }

    // Arming a fresh catalog drops the old outcomes and names what is live.
    #[test]
    fn arming_starts_a_fresh_board() {
        let reports = ShaderReports::default();
        assert_eq!(reports.snapshot().is_live("lit"), None);
        assert_eq!(reports.snapshot().armed_at(), None);
        reports.arm(["lit".to_string()]);
        reports.publish(&[report("lit", failed("broken"))]);
        let published = reports.snapshot().latest("lit").unwrap().seq;
        reports.arm(["lit".to_string(), "new".to_string()]);
        let board = reports.snapshot();
        assert!(board.latest("lit").is_none());
        assert!(board.armed_at().unwrap() > published);
        assert_eq!(board.is_live("new"), Some(true));
        assert_eq!(board.is_live("gone"), Some(false));
    }
}
