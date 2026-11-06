//! How the editor reports a world that failed to build: a console line per
//! problem, and a toast leading with the first that takes the author to the
//! asset it names.

use crate::authoring::ValidationFailed;
use crate::editor::notify::Action;

/// What to show for a failed `what` ("Cook", "Preview rebuild").
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FailureReport {
    /// The console's lines, in order.
    pub(crate) lines: Vec<String>,
    /// The toast's text.
    pub(crate) toast: String,
    /// What clicking the toast does.
    pub(crate) action: Action,
}

/// The report for `what` failing with `e`. A validation failure lists every
/// problem on the console, and its toast leads with the first one that names an
/// asset, which a click selects; with none, the toast leads with the first and
/// opens the console. Anything else is one line that opens the console.
pub(crate) fn report(what: &str, e: &std::io::Error) -> FailureReport {
    let Some(failed) = ValidationFailed::of(e) else {
        let line = format!("{what} failed: {e}");
        return FailureReport {
            toast: toast_line(&line),
            lines: vec![line],
            action: Action::OpenConsole,
        };
    };
    let diagnostics = failed.diagnostics();
    let mut lines = vec![format!("{what} failed: {failed}")];
    lines.extend(diagnostics.iter().map(ToString::to_string));
    let named = diagnostics.iter().find(|d| d.asset_ref().is_some());
    let action = named
        .and_then(|d| d.asset_ref())
        .map_or(Action::OpenConsole, |a| Action::SelectAsset(a.name.clone()));
    let toast = match named.or_else(|| diagnostics.first()) {
        None => format!("{what} failed: {failed}"),
        Some(lead) if diagnostics.len() > 1 => format!(
            "{what} failed: {} (+{} more)",
            toast_line(&lead.to_string()),
            diagnostics.len() - 1
        ),
        Some(lead) => format!("{what} failed: {}", toast_line(&lead.to_string())),
    };
    FailureReport {
        lines,
        toast,
        action,
    }
}

// Longest a toast shows of one problem, in characters.
const TOAST_CHARS: usize = 120;

// The first line of `text`, cut to fit a toast; the console keeps the rest.
fn toast_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default();
    if line.chars().count() <= TOAST_CHARS {
        return line.to_string();
    }
    let cut: String = line.chars().take(TOAST_CHARS - 3).collect();
    format!("{cut}...")
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_cook::Diagnostic;

    fn failed(diagnostics: &[Diagnostic]) -> std::io::Error {
        crate::authoring::validation_failed(diagnostics)
    }

    #[test]
    fn a_validation_failure_leads_with_the_problem_its_click_selects() {
        let e = failed(&[
            Diagnostic::world("a world-wide rule"),
            Diagnostic::asset("Prop", "rock", "unknown mesh 'boulder'"),
        ]);
        let r = report("Cook", &e);
        assert_eq!(
            r.toast,
            "Cook failed: Prop 'rock': unknown mesh 'boulder' (+1 more)"
        );
        assert_eq!(r.action, Action::SelectAsset("rock".into()));
        assert_eq!(
            r.lines,
            [
                "Cook failed: validation failed with 2 error(s)",
                "a world-wide rule",
                "Prop 'rock': unknown mesh 'boulder'",
            ]
        );
    }

    #[test]
    fn a_failure_no_asset_owns_opens_the_console() {
        let r = report(
            "Preview rebuild",
            &failed(&[Diagnostic::world("no Window")]),
        );
        assert_eq!(r.toast, "Preview rebuild failed: no Window");
        assert_eq!(r.action, Action::OpenConsole);
    }

    #[test]
    fn any_other_failure_is_one_line_that_opens_the_console() {
        let e = std::io::Error::other("disk full");
        let r = report("Cook", &e);
        assert_eq!(r.lines, ["Cook failed: disk full"]);
        assert_eq!(r.toast, "Cook failed: disk full");
        assert_eq!(r.action, Action::OpenConsole);
    }

    // A multi-line or long problem shows its first line, cut to fit; the
    // console keeps all of it.
    #[test]
    fn the_toast_is_one_short_line() {
        let long = format!("{}\nsecond line", "x".repeat(200));
        let r = report("Cook", &failed(&[Diagnostic::world(long.clone())]));
        assert!(!r.toast.contains('\n'), "{}", r.toast);
        assert_eq!(r.toast.chars().count(), "Cook failed: ".len() + TOAST_CHARS);
        assert!(r.toast.ends_with("..."));
        assert_eq!(r.lines[1], long);
    }
}
