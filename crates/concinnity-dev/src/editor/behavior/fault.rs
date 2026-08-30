// src/editor/behavior/fault.rs
//
// Turning the checker's complaint into a place in the panel. The checker
// addresses the authored args it read (`concinnity_cook::check::fault`) and the
// outline addresses the same args, so the two agree on where a value is -- the
// only work here is mapping one hop type onto the other and settling for the
// nearest row when the exact spot has none of its own.
//
// Not every location names a row: an unknown verb is reported under itself, a
// list is reported when no one entry is to blame. Trimming the location back
// until a row answers is what keeps those pointing at the node rather than at
// nothing.

use concinnity_cook::check::fault::Step as FaultStep;

use super::outline::Row;
use super::path::{Path, Step};

// The checker's location as a path into the same args the outline addresses.
pub(crate) fn to_path(at: &[FaultStep]) -> Path {
    at.iter()
        .map(|step| match step {
            FaultStep::Field(name) => Step::Field(name.clone()),
            FaultStep::Index(i) => Step::Index(*i),
        })
        .collect()
}

// The row the panel should point at for a fault at `path`: the row addressing
// that exact value, or the innermost row containing it. Resolved against the
// rows on screen rather than remembered, so a path the args no longer hold (an
// undo since the check ran) settles for an ancestor or points at nothing.
pub(crate) fn row_of(rows: &[Row], path: &[Step]) -> Option<usize> {
    (0..=path.len())
        .rev()
        .find_map(|n| rows.iter().position(|r| r.path == path[..n]))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::outline;
    use super::*;

    fn steps(parts: &[&str]) -> Vec<FaultStep> {
        parts
            .iter()
            .map(|s| match s.parse::<usize>() {
                Ok(i) => FaultStep::Index(i),
                Err(_) => FaultStep::Field(s.to_string()),
            })
            .collect()
    }

    // What the panel resolves: a checker location, mapped and then looked up.
    fn at(parts: &[&str]) -> Path {
        to_path(&steps(parts))
    }

    fn sample() -> Vec<Row> {
        outline::rows(&json!({
            "on": "tick",
            "scope": ["Prop"],
            "do": [
                {"if": {"cond": {"bool": true}, "then": [{"hide": {"target": "self"}}]}},
                {"save": {}},
            ],
        }))
    }

    #[test]
    fn an_addressed_value_resolves_to_its_own_row() {
        let rows = sample();
        let cond = row_of(&rows, &at(&["do", "0", "if", "cond"])).expect("the cond row");
        assert_eq!(rows[cond].label, "cond");
        let node = row_of(&rows, &at(&["do", "1"])).expect("the second node's row");
        assert_eq!(rows[node].label, "save");
    }

    // A location no row answers to settles for the nearest one that does, which
    // is how an unknown verb still points at the node carrying it.
    #[test]
    fn an_unaddressed_value_settles_for_the_row_containing_it() {
        let rows = outline::rows(&json!({"on": "start", "do": [{"teleport": {}}]}));
        let node = row_of(&rows, &at(&["do", "0", "teleport"])).expect("the node's row");
        assert_eq!(
            rows[node].path,
            vec![Step::Field("do".into()), Step::Index(0)]
        );
    }

    #[test]
    fn a_location_pointing_nowhere_at_all_falls_back_to_the_whole_asset() {
        let rows = sample();
        // Nothing is at `nonsense`, and no row addresses the root, so there is
        // nothing to point at rather than something wrong to point at.
        assert_eq!(row_of(&rows, &at(&["nonsense", "4"])), None);
        assert_eq!(row_of(&[], &at(&["do", "0"])), None);
    }

    #[test]
    fn hop_kinds_carry_across_unchanged() {
        assert_eq!(
            to_path(&steps(&["do", "2", "if"])),
            vec![
                Step::Field("do".into()),
                Step::Index(2),
                Step::Field("if".into()),
            ],
        );
        assert!(to_path(&[]).is_empty());
    }
}
