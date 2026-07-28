// src/editor/behavior/graph.rs
//
// Laying a behavior's body out as a left-to-right chart. The body is a tree --
// `do` is an ordered list and a branching node owns child lists -- so a card's
// place is computed from the asset each time rather than stored in it, and a
// node carries no authored position. Sequence runs left to right, one column
// per node; a branching node stacks its branches in the columns to its right,
// and the node after it resumes past the whole branch. Every card addresses a
// path the outline addresses too, so selecting a card selects its row and the
// toolbar, palette, and value field act on it unchanged.

use serde_json::Value;

use super::outline;
use super::palette::{self, Shape};
use super::path::{Path, Step, child, field};

// How deep an expression renders on a card before the rest is elided: enough
// for the shape of a condition to read, short of crowding out the title.
const EXPR_DEPTH: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CardKind {
    // The `on` source, one column left of the body.
    Trigger,
    // One node of the body.
    Node,
    // A branch holding no nodes yet. Its path is the branch's list, so picking
    // from it appends the branch's first node.
    Empty,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Card {
    pub column: usize,
    pub row: usize,
    pub title: String,
    pub detail: String,
    pub kind: CardKind,
    pub path: Path,
}

// A parent-to-child link, labelled with the branch it leaves the parent by when
// the parent has more than one way out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Wire {
    pub from: usize,
    pub to: usize,
    pub label: Option<&'static str>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Chart {
    pub cards: Vec<Card>,
    pub wires: Vec<Wire>,
    pub columns: usize,
    pub rows: usize,
}

// The whole behavior as a chart: the trigger, then its body branching right.
pub(crate) fn chart(args: &Value) -> Chart {
    let mut build = Build {
        cards: Vec::new(),
        wires: Vec::new(),
    };
    let on = args.get("on");
    let verb = on.map_or("start", palette::verb_of);
    build.push(Card {
        column: 0,
        row: 0,
        title: format!("on {}", outline::source_summary(verb, on)),
        detail: trigger_detail(args),
        kind: CardKind::Trigger,
        path: vec![field("on")],
    });
    let base = vec![field("do")];
    let (columns, rows) = build.chain(array(args.get("do")), &base, 1, 0, 0, None);
    Chart {
        cards: build.cards,
        wires: build.wires,
        columns: columns + 1,
        rows,
    }
}

// The branch lists a node owns, in the order they stack downward.
fn branches(verb: &str) -> &'static [&'static str] {
    match verb {
        "if" => &["then", "else"],
        "for_each" => &["do"],
        _ => &[],
    }
}

struct Build {
    cards: Vec<Card>,
    wires: Vec<Wire>,
}

impl Build {
    fn push(&mut self, card: Card) -> usize {
        self.cards.push(card);
        self.cards.len() - 1
    }

    // Place `items` as a chain starting at (`column`, `row`), its first card
    // hanging off `parent` by `pin`. Returns the columns and rows the chain and
    // everything branching off it consumed, which is what lets the node after a
    // branch resume clear of it.
    fn chain(
        &mut self,
        items: &[Value],
        base: &Path,
        column: usize,
        row: usize,
        parent: usize,
        pin: Option<&'static str>,
    ) -> (usize, usize) {
        if items.is_empty() {
            let to = self.push(Card {
                column,
                row,
                title: "empty".to_string(),
                detail: String::new(),
                kind: CardKind::Empty,
                path: base.clone(),
            });
            self.wires.push(Wire {
                from: parent,
                to,
                label: pin,
            });
            return (1, 1);
        }
        let mut columns = 0;
        let mut rows = 1;
        let mut prev = parent;
        let mut prev_pin = pin;
        for (i, node) in items.iter().enumerate() {
            let path = child(base, Step::Index(i));
            let verb = palette::verb_of(node);
            let known = palette::known(palette::NODES, verb);
            let body = palette::body_of(node);
            let to = self.push(Card {
                column: column + columns,
                row,
                title: if known { verb } else { "(unknown)" }.to_string(),
                detail: if known {
                    node_caption(verb, body)
                } else {
                    String::new()
                },
                kind: CardKind::Node,
                path: path.clone(),
            });
            self.wires.push(Wire {
                from: prev,
                to,
                label: prev_pin,
            });
            prev = to;
            prev_pin = None;
            let mut branch_columns = 0;
            let mut branch_rows = 0;
            for name in branches(verb) {
                let list = array(body.and_then(|b| b.get(name)));
                let branch = child(&child(&path, field(verb)), field(name));
                let (c, r) = self.chain(
                    list,
                    &branch,
                    column + columns + 1,
                    row + branch_rows,
                    to,
                    Some(name),
                );
                branch_columns = branch_columns.max(c);
                branch_rows += r;
            }
            columns += 1 + branch_columns;
            rows = rows.max(branch_rows.max(1));
        }
        (columns, rows)
    }
}

// What the trigger card says below its source: the entities the behavior runs
// against, and whatever else it declares.
fn trigger_detail(args: &Value) -> String {
    let scope: Vec<&str> = array(args.get("scope"))
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let mut parts = vec![if scope.is_empty() {
        "world-scoped".to_string()
    } else {
        scope.join(", ")
    }];
    for (key, noun) in [("locals", "local"), ("queries", "query")] {
        let n = array(args.get(key)).len();
        if n > 0 {
            parts.push(format!("{n} {noun}{}", if n == 1 { "" } else { "s" }));
        }
    }
    parts.join(", ")
}

// The one line a node card carries under its verb: the field that says what
// this node does, rendered the way it would be written.
fn node_caption(verb: &str, body: Option<&Value>) -> String {
    let at = |key: &str| body.and_then(|v| v.get(key));
    let name = |key: &str| at(key).and_then(Value::as_str).unwrap_or("").to_string();
    let expr = |key: &str| match at(key) {
        Some(v) if !v.is_null() => expr_text(v, EXPR_DEPTH),
        _ => "unset".to_string(),
    };
    let assign = |target: String, key: &str| {
        let op = if at("add").and_then(Value::as_bool).unwrap_or(false) {
            "+="
        } else {
            "="
        };
        format!("{target} {op} {}", expr(key))
    };
    match verb {
        "if" => expr("cond"),
        "for_each" => format!("{} -> {}", name("query"), name("bind")),
        "let" => format!("{} = {}", name("name"), expr("value")),
        "set" => assign(name("var"), "value"),
        "set_local" => assign(name("local"), "value"),
        "set_transform" => transform_caption(body, expr("entity")),
        "spawn" => name("template"),
        "despawn" | "show" | "hide" => expr("target"),
        "reparent" => format!("{} -> {}", expr("child"), expr("parent")),
        "sound" => name("clip"),
        "scene" => name("scene"),
        "screen" => name("screen"),
        "story" => body.and_then(Value::as_str).unwrap_or("start").to_string(),
        _ => String::new(),
    }
}

// A transform node reads as the entity plus which of its parts are written, so
// a card says what moves without unfolding three expression trees.
fn transform_caption(body: Option<&Value>, entity: String) -> String {
    let written: Vec<&str> = ["position", "rotation_deg", "scale"]
        .into_iter()
        .filter(|key| body.and_then(|v| v.get(key)).is_some_and(|v| !v.is_null()))
        .collect();
    if written.is_empty() {
        entity
    } else {
        format!("{entity}: {}", written.join(", "))
    }
}

// An expression as one line, nested `depth` levels before the rest elides.
// Comparisons and arithmetic read infix and everything else as a call, so a
// condition on a card reads the way it would be spoken.
pub(crate) fn expr_text(value: &Value, depth: usize) -> String {
    let verb = palette::verb_of(value);
    if !palette::known(palette::EXPRS, verb) {
        return "(unknown)".to_string();
    }
    let body = palette::body_of(value);
    match palette::shape(verb) {
        Shape::Unit => verb.to_string(),
        Shape::Literal => literal(verb, body),
        Shape::Name => {
            let name = body.and_then(Value::as_str).unwrap_or("");
            match (verb, name) {
                ("first" | "count", _) => format!("{verb}({name})"),
                (_, "") => format!("{verb} (unnamed)"),
                _ => name.to_string(),
            }
        }
        _ if depth == 0 => "...".to_string(),
        Shape::Unary => {
            let operand = operand(body, depth);
            if verb == "not" {
                format!("not {operand}")
            } else {
                format!("{verb}({operand})")
            }
        }
        Shape::Binary => {
            let items = array(body);
            let a = operand(items.first(), depth);
            let b = operand(items.get(1), depth);
            match infix(verb) {
                Some(op) => format!("{a} {op} {b}"),
                None => format!("{verb}({a}, {b})"),
            }
        }
        Shape::List => {
            let joiner = if verb == "all" { " and " } else { " or " };
            let items: Vec<String> = array(body)
                .iter()
                .map(|v| operand(Some(v), depth))
                .collect();
            if items.is_empty() {
                format!("{verb} (empty)")
            } else {
                items.join(joiner)
            }
        }
    }
}

// One operand of a nested expression, parenthesised when it is itself infix so
// the precedence a reader assumes is the one the tree has.
fn operand(value: Option<&Value>, depth: usize) -> String {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return "unset".to_string();
    };
    let text = expr_text(value, depth - 1);
    if infix(palette::verb_of(value)).is_some() && depth > 1 {
        format!("({text})")
    } else {
        text
    }
}

fn infix(verb: &str) -> Option<&'static str> {
    Some(match verb {
        "add" => "+",
        "sub" => "-",
        "mul" => "*",
        "div" => "/",
        "eq" => "==",
        "ne" => "!=",
        "lt" => "<",
        "le" => "<=",
        "gt" => ">",
        "ge" => ">=",
        _ => return None,
    })
}

// A constant's payload; a vector keeps its brackets so its components do not
// read as separate operands.
fn literal(verb: &str, body: Option<&Value>) -> String {
    let text = outline::literal_text(body);
    if verb == "vec3" {
        format!("({text})")
    } else {
        text
    }
}

fn array(value: Option<&Value>) -> &[Value] {
    value
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn at(chart: &Chart, title: &str) -> Card {
        chart
            .cards
            .iter()
            .find(|c| c.title == title)
            .unwrap_or_else(|| panic!("no card titled {title}: {:?}", chart.cards))
            .clone()
    }

    fn wire_to(chart: &Chart, title: &str) -> Wire {
        let to = chart.cards.iter().position(|c| c.title == title).unwrap();
        *chart.wires.iter().find(|w| w.to == to).unwrap()
    }

    #[test]
    fn a_sequence_chains_left_to_right_from_the_trigger() {
        let chart = chart(&json!({
            "on": "start",
            "do": [{"save": {}}, {"hide": {"target": {"named": "door"}}}],
        }));
        let trigger = &chart.cards[0];
        assert_eq!(trigger.kind, CardKind::Trigger);
        assert_eq!((trigger.column, trigger.row), (0, 0));
        assert_eq!(trigger.title, "on start");
        assert_eq!((at(&chart, "save").column, at(&chart, "save").row), (1, 0));
        assert_eq!((at(&chart, "hide").column, at(&chart, "hide").row), (2, 0));
        assert_eq!(chart.columns, 3);
        assert_eq!(chart.rows, 1);
        // Every card past the trigger hangs off the one before it.
        assert_eq!(wire_to(&chart, "save").from, 0);
        assert_eq!(wire_to(&chart, "hide").from, 1);
        assert!(chart.wires.iter().all(|w| w.label.is_none()));
    }

    #[test]
    fn a_branch_stacks_downward_and_the_next_node_resumes_past_it() {
        let chart = chart(&json!({
            "on": "tick",
            "do": [
                {"if": {
                    "cond": {"bool": true},
                    "then": [{"show": {"target": "self"}}],
                    "else": [{"hide": {"target": "self"}}],
                }},
                {"save": {}},
            ],
        }));
        let branch = at(&chart, "if");
        assert_eq!((branch.column, branch.row), (1, 0));
        // The first branch keeps the parent's row, the second drops below it.
        assert_eq!((at(&chart, "show").column, at(&chart, "show").row), (2, 0));
        assert_eq!((at(&chart, "hide").column, at(&chart, "hide").row), (2, 1));
        // The node after the branch clears the whole subtree.
        assert_eq!((at(&chart, "save").column, at(&chart, "save").row), (3, 0));
        assert_eq!(chart.rows, 2);
        assert_eq!(wire_to(&chart, "show").label, Some("then"));
        assert_eq!(wire_to(&chart, "hide").label, Some("else"));
        assert_eq!(wire_to(&chart, "save").label, None);
    }

    #[test]
    fn an_empty_branch_becomes_a_card_addressing_the_branch_list() {
        let chart = chart(&json!({
            "on": "start",
            "do": [{"if": {"cond": {"bool": true}, "then": [{"save": {}}]}}],
        }));
        let empty = at(&chart, "empty");
        assert_eq!(empty.kind, CardKind::Empty);
        assert_eq!(wire_to(&chart, "empty").label, Some("else"));
        assert_eq!(
            empty.path,
            vec![field("do"), Step::Index(0), field("if"), field("else"),],
        );
    }

    #[test]
    fn an_empty_body_still_offers_somewhere_to_start() {
        let chart = chart(&json!({"on": "start"}));
        assert_eq!(chart.cards.len(), 2);
        assert_eq!(chart.cards[1].kind, CardKind::Empty);
        assert_eq!(chart.cards[1].path, vec![field("do")]);
    }

    // The chart is a second view over the outline's rows, so every card has to
    // land on a path the outline also addresses: that is what lets a card's
    // selection drive the existing toolbar, palette, and value field.
    #[test]
    fn every_card_addresses_a_row_the_outline_addresses() {
        let args = json!({
            "on": {"timer": {"interval": 2.0, "repeat": true}},
            "scope": ["Prop"],
            "locals": [{"name": "speed", "value": {"float": 3.0}}],
            "queries": [{"name": "player", "has": ["Camera3D"]}],
            "do": [
                {"let": {"name": "target", "value": {"first": "player"}}},
                {"if": {
                    "cond": {"lt": [{"distance": ["self", {"bind": "target"}]}, {"float": 20.0}]},
                    "then": [{"for_each": {"query": "player", "bind": "p", "do": []}}],
                }},
            ],
        });
        let chart = chart(&args);
        let rows = outline::rows(&args);
        for card in &chart.cards {
            assert!(
                rows.iter().any(|r| r.path == card.path),
                "no outline row at {:?} (card {:?})",
                card.path,
                card.title,
            );
        }
    }

    #[test]
    fn an_unknown_verb_gets_a_card_but_no_invented_branches() {
        let chart = chart(&json!({"on": "start", "do": [{"teleport": {"then": [{"save": {}}]}}]}));
        assert_eq!(chart.cards.len(), 2);
        assert_eq!(chart.cards[1].title, "(unknown)");
        assert_eq!(chart.cards[1].detail, "");
    }

    #[test]
    fn conditions_render_infix_with_calls_nested_inside() {
        let cond = json!({"lt": [{"distance": ["self", {"bind": "target"}]}, {"float": 20.0}]});
        assert_eq!(expr_text(&cond, EXPR_DEPTH), "distance(self, target) < 20");
    }

    #[test]
    fn nested_arithmetic_keeps_the_precedence_its_tree_has() {
        let value = json!({"mul": [{"add": [{"local": "a"}, {"local": "b"}]}, "dt"]});
        assert_eq!(expr_text(&value, EXPR_DEPTH), "(a + b) * dt");
    }

    #[test]
    fn a_deep_expression_elides_rather_than_running_off_the_card() {
        let value = json!({"add": [{"add": [{"add": [{"add": ["dt", "dt"]}, "dt"]}, "dt"]}, "dt"]});
        let text = expr_text(&value, EXPR_DEPTH);
        assert!(text.contains("..."), "{text}");
        assert!(text.len() < 40, "{text}");
    }

    #[test]
    fn node_captions_say_what_each_node_writes() {
        let captions = |body: Value| {
            let chart = chart(&json!({"on": "start", "do": [body]}));
            chart.cards[1].detail.clone()
        };
        assert_eq!(
            captions(json!({"set": {"var": "visits", "value": {"int": 1}, "add": true}})),
            "visits += 1",
        );
        assert_eq!(
            captions(json!({"set": {"var": "visits", "value": {"int": 1}}})),
            "visits = 1",
        );
        assert_eq!(
            captions(json!({"set_transform": {"entity": "self", "position": {"vec3": [0, 1, 0]}}})),
            "self: position",
        );
        assert_eq!(
            captions(json!({"for_each": {"query": "player", "bind": "p", "do": []}})),
            "player -> p",
        );
        assert_eq!(
            captions(json!({"hide": {"target": {"named": "door"}}})),
            "door"
        );
        assert_eq!(captions(json!({"despawn": {}})), "unset");
    }

    #[test]
    fn the_trigger_card_carries_the_source_and_what_it_runs_against() {
        let chart = chart(&json!({
            "on": {"timer": {"interval": 2.0, "repeat": true}},
            "scope": ["Prop"],
            "locals": [{"name": "speed", "value": {"float": 3.0}}],
        }));
        assert_eq!(chart.cards[0].title, "on timer 2s repeating");
        assert_eq!(chart.cards[0].detail, "Prop, 1 local");
        let world = super::chart(&json!({"on": "start"}));
        assert_eq!(world.cards[0].detail, "world-scoped");
    }
}
