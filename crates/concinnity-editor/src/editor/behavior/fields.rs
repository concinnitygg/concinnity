// src/editor/behavior/fields.rs
//
// Which outline rows belong to one node. The chart draws a node as a card, and
// selecting a card shows that node's settings beside it -- so the chart needs
// the node's own fields without the nodes nested inside it, which the outline
// lists inline.
//
// One rule answers both questions: a row belongs to the card with the longest
// `settles` containing it. The trigger settles the whole asset, each node its
// own subtree, and each chain's tail its list -- so every row of the outline
// has exactly one card that reaches it, and no row is reachable twice.

use super::graph::Card;
use super::outline::Row;
use super::path::{self, Path};

// The card the row at `path` belongs to. A node's own field answers with the
// node, so selecting a field keeps the same node in the inspector rather than
// emptying it.
pub(crate) fn owning_card(cards: &[Card], path: &Path) -> Option<usize> {
    cards
        .iter()
        .enumerate()
        .filter(|(_, c)| path::starts_with(path, &c.settles))
        .max_by_key(|(_, c)| c.settles.len())
        .map(|(i, _)| i)
}

// The rows the card at `card` settles, in outline order.
pub(crate) fn own_rows(rows: &[Row], cards: &[Card], card: usize) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter(|(_, r)| owning_card(cards, &r.path) == Some(card))
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
        own_rows(&rows, &chart.cards, card)
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

    // The condition is the `if`'s own. Neither the nodes in its branches nor the
    // branch lists themselves are: each branch ends in a tail card, and that is
    // what appending to the branch goes through.
    #[test]
    fn a_branch_keeps_its_condition_and_nothing_below_it() {
        let args = json!({"do": [{"if": {
            "cond": {"bool": true},
            "then": [{"save": null}],
            "else": [],
        }}]});
        let listed = listed(&args, 1);
        assert!(listed.iter().any(|l| l == "cond"));
        for below in ["then", "else", "save"] {
            assert!(
                !listed.iter().any(|l| l == below),
                "`{below}` has a card of its own: {listed:?}"
            );
        }
    }

    // The trigger card stands for the behavior firing, so it settles what the
    // behavior declares once -- which is otherwise unreachable from the chart,
    // since none of it hangs off a node.
    #[test]
    fn the_trigger_settles_the_source_and_what_the_behavior_declares() {
        let args = json!({
            "on": {"timer": {"interval": 5.0, "repeat": true}},
            "scope": ["Prop"],
            "do": [{"save": null}],
        });
        let listed = listed(&args, 0);
        for want in [
            "on", "interval", "repeat", "once", "delay", "cooldown", "scope",
        ] {
            assert!(listed.iter().any(|l| l == want), "{want} in {listed:?}");
        }
        assert!(!listed.iter().any(|l| l == "do"), "the body is not its own");
    }

    // The point of the whole rule: no row of the outline is stranded, so every
    // setting is reachable from the chart, and none is reachable from two cards.
    #[test]
    fn every_outline_row_belongs_to_exactly_one_card() {
        let args = json!({
            "on": "tick",
            "scope": ["Prop"],
            "locals": [{"name": "speed", "value": {"float": 3.0}}],
            "queries": [{"name": "player", "has": ["Camera3D"]}],
            "do": [
                {"if": {"cond": {"bool": true}, "then": [{"save": null}], "else": []}},
                {"spawn": {"template": "drop"}},
            ],
        });
        let rows = outline::rows(&args);
        let chart = graph::chart(&args);
        let stranded: Vec<&str> = rows
            .iter()
            .filter(|r| owning_card(&chart.cards, &r.path).is_none())
            .map(|r| r.label.as_str())
            .collect();
        assert!(stranded.is_empty(), "no card reaches {stranded:?}");
        let listed: usize = (0..chart.cards.len())
            .map(|i| own_rows(&rows, &chart.cards, i).len())
            .sum();
        assert_eq!(listed, rows.len(), "a row is listed twice");
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

    // A declaration hangs off no node, so it answers with the trigger -- the
    // card that stands for the behavior itself.
    #[test]
    fn a_declaration_belongs_to_the_trigger() {
        let args = json!({"scope": ["Prop"], "do": [{"save": null}]});
        let rows = outline::rows(&args);
        let chart = graph::chart(&args);
        let scope = rows.iter().find(|r| r.label == "scope").expect("scope");
        assert_eq!(owning_card(&chart.cards, &scope.path), Some(0));
        assert_eq!(chart.cards[0].kind, graph::CardKind::Trigger);
    }

    // A list's rows belong to the card that appends to it, so picking on that
    // card is what grows the list.
    #[test]
    fn a_list_belongs_to_the_card_that_appends_to_it() {
        let args = json!({"do": [{"save": null}]});
        let rows = outline::rows(&args);
        let chart = graph::chart(&args);
        let body = rows.iter().find(|r| r.label == "do").expect("do");
        let card = owning_card(&chart.cards, &body.path).expect("a card reaches it");
        assert_eq!(chart.cards[card].kind, graph::CardKind::Add);
        assert_eq!(chart.cards[card].path, body.path);
    }
}
