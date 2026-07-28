// src/editor/behavior/fields.rs
//
// Which outline rows belong to one node. The chart draws a node as a card, and
// selecting a card shows that node's settings beside it -- so the chart needs
// the node's own fields without the nodes nested inside it, which the outline
// lists inline.
//
// The answer comes from the cards themselves: a row belongs to the node it sits
// under unless some other card already owns it. That keeps the two views from
// disagreeing, because a node reachable as a card is never also a field.

use super::graph::Card;
use super::outline::Row;
use super::path::{self, Path};

// The card the row at `path` belongs to: the innermost one containing it. A
// node's own field answers with the node, so selecting a field keeps the same
// node in the inspector rather than emptying it.
pub(crate) fn owning_card(cards: &[Card], path: &Path) -> Option<usize> {
    cards
        .iter()
        .enumerate()
        .filter(|(_, c)| path::starts_with(path, &c.path))
        .max_by_key(|(_, c)| c.path.len())
        .map(|(i, _)| i)
}

// The rows the node at `path` settles: its own row first, then everything under
// it that no other card owns. An `if` keeps its condition and its two branch
// lists; the nodes stacked in those branches are cards of their own.
pub(crate) fn own_rows(rows: &[Row], cards: &[Card], path: &Path) -> Vec<usize> {
    let others: Vec<&Path> = cards
        .iter()
        .map(|c| &c.path)
        .filter(|p| *p != path)
        .collect();
    rows.iter()
        .enumerate()
        .filter(|(_, r)| {
            path::starts_with(&r.path, path)
                && !others.iter().any(|o| path::starts_with(&r.path, o))
        })
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::super::{graph, outline};
    use super::*;

    // The labels the inspector would list for the card at `card`, which is what
    // the pane actually draws.
    fn listed(args: &Value, card: usize) -> Vec<String> {
        let rows = outline::rows(args);
        let chart = graph::chart(args);
        own_rows(&rows, &chart.cards, &chart.cards[card].path)
            .into_iter()
            .map(|i| rows[i].label.clone())
            .collect()
    }

    #[test]
    fn a_node_settles_its_own_fields() {
        let args = json!({"do": [{"spawn": {"template": "drop", "lifetime": 4.0}}]});
        let listed = listed(&args, 1);
        assert_eq!(listed.first().map(String::as_str), Some("spawn"));
        for field in ["template", "position", "lifetime"] {
            assert!(listed.iter().any(|l| l == field), "{field} in {listed:?}");
        }
    }

    // The condition is the `if`'s own; the nodes in its branches are not, because
    // each of those is a card the user selects directly.
    #[test]
    fn a_branch_keeps_its_condition_and_not_its_children() {
        let args = json!({"do": [{"if": {
            "cond": {"bool": true},
            "then": [{"save": null}],
            "else": [],
        }}]});
        let listed = listed(&args, 1);
        assert!(listed.iter().any(|l| l == "cond"));
        assert!(listed.iter().any(|l| l == "then"), "the branch list itself");
        assert!(
            !listed.iter().any(|l| l == "save"),
            "the node inside the branch is its own card: {listed:?}"
        );
    }

    // An empty branch draws an `empty` card whose path is the branch list, so
    // the list row belongs to that card rather than to the node above it.
    #[test]
    fn an_empty_branch_belongs_to_its_own_card() {
        let args = json!({"do": [{"if": {"cond": {"bool": true}, "then": [], "else": []}}]});
        assert!(!listed(&args, 1).iter().any(|l| l == "then"));
    }

    #[test]
    fn the_trigger_settles_the_source_and_its_parameters() {
        let args = json!({"on": {"timer": {"interval": 5.0, "repeat": true}}, "do": []});
        let listed = listed(&args, 0);
        assert_eq!(listed, ["on", "interval", "repeat"]);
    }

    #[test]
    fn a_node_with_no_fields_lists_only_itself() {
        let args = json!({"do": [{"save": null}]});
        assert_eq!(listed(&args, 1), ["save"]);
    }

    // Selecting one of a node's fields keeps that node in the inspector: the
    // field answers with the node it belongs to, not with nothing.
    #[test]
    fn a_field_belongs_to_the_node_holding_it() {
        let args = json!({"do": [{"spawn": {"template": "drop", "lifetime": 4.0}}]});
        let rows = outline::rows(&args);
        let chart = graph::chart(&args);
        let node = &chart.cards[1].path;
        for label in ["spawn", "lifetime"] {
            let row = rows.iter().find(|r| r.label == label).expect(label);
            let card = owning_card(&chart.cards, &row.path).expect(label);
            assert_eq!(
                &chart.cards[card].path, node,
                "`{label}` answered elsewhere"
            );
        }
    }

    // A declaration is nowhere in the chart, so it answers with no card at all
    // rather than with whichever one happens to sit nearest it.
    #[test]
    fn a_declaration_belongs_to_no_card() {
        let args = json!({"scope": ["Prop"], "do": [{"save": null}]});
        let rows = outline::rows(&args);
        let chart = graph::chart(&args);
        let scope = rows.iter().find(|r| r.label == "scope").expect("scope");
        assert_eq!(owning_card(&chart.cards, &scope.path), None);
    }
}
