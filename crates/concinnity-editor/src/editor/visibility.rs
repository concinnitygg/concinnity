// src/editor/visibility.rs
//
// Pure composition of the editor's two hide mechanisms: the manual per-asset
// hide set (outliner eye, H) and an active isolate (Shift+H), which keeps a
// snapshot of names visible and hides everything else. The manual set is
// never mutated by isolate, so leaving isolate restores exactly the
// manually-hidden state, and a name in both stays hidden.

use std::collections::BTreeSet;

// The set of names to hide this frame. With no isolate active this is the
// manual set; with one active it is the manual set plus everything outside
// the kept snapshot.
pub(crate) fn effective_hidden<'a>(
    manual: &BTreeSet<String>,
    isolate: Option<&BTreeSet<String>>,
    all: impl IntoIterator<Item = &'a str>,
) -> BTreeSet<String> {
    let mut hidden = manual.clone();
    if let Some(keep) = isolate {
        hidden.extend(
            all.into_iter()
                .filter(|n| !keep.contains(*n))
                .map(str::to_string),
        );
    }
    hidden
}

// Whether one name is hidden under the same rule, for per-entry filters that
// never materialize the full set.
pub(crate) fn is_hidden(
    name: &str,
    manual: &BTreeSet<String>,
    isolate: Option<&BTreeSet<String>>,
) -> bool {
    manual.contains(name) || isolate.is_some_and(|keep| !keep.contains(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_isolate_is_just_the_manual_set() {
        let manual = set(&["a"]);
        let hidden = effective_hidden(&manual, None, ["a", "b", "c"]);
        assert_eq!(hidden, set(&["a"]));
    }

    #[test]
    fn isolate_hides_everything_outside_the_kept_set() {
        let manual = BTreeSet::new();
        let keep = set(&["b"]);
        let hidden = effective_hidden(&manual, Some(&keep), ["a", "b", "c"]);
        assert_eq!(hidden, set(&["a", "c"]));
    }

    #[test]
    fn manual_wins_inside_the_kept_set() {
        // An asset both isolated and manually hidden stays hidden.
        let manual = set(&["b"]);
        let keep = set(&["a", "b"]);
        let hidden = effective_hidden(&manual, Some(&keep), ["a", "b", "c"]);
        assert_eq!(hidden, set(&["b", "c"]));
        assert!(is_hidden("b", &manual, Some(&keep)));
        assert!(!is_hidden("a", &manual, Some(&keep)));
    }

    #[test]
    fn leaving_isolate_restores_the_manual_state() {
        let manual = set(&["a"]);
        let keep = set(&["b"]);
        let during = effective_hidden(&manual, Some(&keep), ["a", "b", "c"]);
        assert_eq!(during, set(&["a", "c"]));
        // Dropping the isolate recomputes to exactly the manual set: nothing
        // was mutated while it was active.
        let after = effective_hidden(&manual, None, ["a", "b", "c"]);
        assert_eq!(after, set(&["a"]));
    }

    #[test]
    fn per_name_check_matches_the_set() {
        let manual = set(&["a"]);
        let keep = set(&["b"]);
        for n in ["a", "b", "c"] {
            let in_set = effective_hidden(&manual, Some(&keep), ["a", "b", "c"]).contains(n);
            assert_eq!(is_hidden(n, &manual, Some(&keep)), in_set, "{n}");
        }
    }
}
