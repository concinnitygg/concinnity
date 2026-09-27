//! One toast per group of reports that came to the same end, so a save of a
//! field several volumes share is one toast rather than one per volume.

use super::{ReloadOutcome, ReloadReport, SubjectKind, warned};

// Names a toast lists before it counts the rest.
const NAMED: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Toast {
    pub error: bool,
    pub text: String,
}

// A toast per kind and ending, in the order each first appears in `reports`.
pub(super) fn toasts(reports: &[ReloadReport]) -> Vec<Toast> {
    let mut groups: Vec<(SubjectKind, bool, String, Vec<&str>)> = Vec::new();
    for report in reports {
        let kind = report.subject.kind;
        let (error, ending) = ending(kind, &report.outcome);
        let name = report.subject.name.as_str();
        match groups
            .iter_mut()
            .find(|(k, e, t, _)| *k == kind && *e == error && *t == ending)
        {
            Some((_, _, _, names)) => names.push(name),
            None => groups.push((kind, error, ending, vec![name])),
        }
    }
    groups
        .into_iter()
        .map(|(kind, error, ending, names)| Toast {
            error,
            text: format!("{}{ending}", subjects(kind, &names)),
        })
        .collect()
}

// What a toast says after its subjects, and whether it is an error.
fn ending(kind: SubjectKind, outcome: &ReloadOutcome) -> (bool, String) {
    match outcome {
        ReloadOutcome::Swapped { warnings, .. } => {
            (false, format!(" reloaded{}", warned(warnings)))
        }
        ReloadOutcome::AppliesOnLoad { warnings } => (
            false,
            format!(" reloaded{} ({})", warned(warnings), kind.applies_when()),
        ),
        ReloadOutcome::Failed(e) => {
            let at = e.first_error_at().map(|at| format!(" at {at}"));
            (true, format!(" reload failed{}", at.unwrap_or_default()))
        }
    }
}

// "Shader 'lit'", or "SdfVolumes 'a', 'b', 'c' and 2 more".
fn subjects(kind: SubjectKind, names: &[&str]) -> String {
    if let [name] = names {
        return format!("{} '{name}'", kind.label());
    }
    let listed: Vec<String> = names.iter().take(NAMED).map(|n| format!("'{n}'")).collect();
    let rest = names.len().saturating_sub(NAMED);
    let more = if rest > 0 {
        format!(" and {rest} more")
    } else {
        String::new()
    };
    format!("{} {}{more}", kind.plural(), listed.join(", "))
}

#[cfg(test)]
mod tests {
    use super::super::{ReloadFailure, ReloadSubject};
    use super::*;
    use std::time::Duration;

    fn swapped() -> ReloadOutcome {
        ReloadOutcome::Swapped {
            frame_time: Duration::from_millis(3),
            warnings: Vec::new(),
        }
    }

    fn volume(name: &str, outcome: ReloadOutcome) -> ReloadReport {
        ReloadReport {
            subject: ReloadSubject::sdf_volume(name),
            outcome,
        }
    }

    fn texts(reports: &[ReloadReport]) -> Vec<(bool, String)> {
        toasts(reports)
            .into_iter()
            .map(|t| (t.error, t.text))
            .collect()
    }

    // A lone subject reads as it always has.
    #[test]
    fn one_subject_is_one_toast_naming_it() {
        let lit = ReloadReport {
            subject: ReloadSubject::shader("lit"),
            outcome: swapped(),
        };
        assert_eq!(
            texts(&[lit]),
            [(false, "Shader 'lit' reloaded".to_string())]
        );
    }

    // The volumes a shared field swapped are one toast; one that failed, and
    // one of another kind, stand apart.
    #[test]
    fn subjects_that_came_to_the_same_end_share_a_toast() {
        let reports = [
            volume("cloud_a", swapped()),
            ReloadReport {
                subject: ReloadSubject::shader("cloud_b"),
                outcome: swapped(),
            },
            volume("cloud_b", swapped()),
            volume(
                "blob",
                ReloadOutcome::Failed(ReloadFailure::Unstarted("gone".to_string())),
            ),
            volume(
                "lost",
                ReloadOutcome::AppliesOnLoad {
                    warnings: Vec::new(),
                },
            ),
        ];
        assert_eq!(
            texts(&reports),
            [
                (
                    false,
                    "SdfVolumes 'cloud_a', 'cloud_b' reloaded".to_string()
                ),
                (false, "Shader 'cloud_b' reloaded".to_string()),
                (true, "SdfVolume 'blob' reload failed".to_string()),
                (
                    false,
                    "SdfVolume 'lost' reloaded (applies when the world is next built)".to_string()
                ),
            ]
        );
    }

    // A long list names the first few and counts the rest.
    #[test]
    fn a_long_list_counts_what_it_does_not_name() {
        let reports: Vec<ReloadReport> = ["a", "b", "c", "d", "e"]
            .into_iter()
            .map(|n| volume(n, swapped()))
            .collect();
        assert_eq!(
            texts(&reports),
            [(
                false,
                "SdfVolumes 'a', 'b', 'c' and 2 more reloaded".to_string()
            )]
        );
    }
}
