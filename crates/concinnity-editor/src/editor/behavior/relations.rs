// src/editor/behavior/relations.rs
//
// The world's behaviors as one map, so what a body does is legible next to what
// it sets off. A behavior on its own says what it runs; it does not say what
// started it or what it starts, and that is the part a panel showing one
// behavior at a time cannot answer.
//
// Behaviors reach each other through the world, never directly: one writes a
// variable another fires on, or spawns an entity another picks up. So the map
// draws those middlemen as cards of their own -- a trigger, a variable -- and a
// chain reads left to right, from what starts something to what it starts.
//
// The result is an ordinary `Chart`, so the chart view draws it unchanged.

use serde_json::Value;

use super::graph::{Card, CardKind, Chart, Wire};
use super::outline;
use super::palette;

// Far enough right for any chain worth reading, and a stop for the relaxation
// below: variables can carry a cycle (two behaviors each firing on what the
// other sets), which has no leftmost order to find.
const MAX_COLUMN: usize = 12;

// How a behavior reads a variable: the ones that fire it are what a chain is
// made of, the ones it merely consults are worth seeing but weaker.
const FIRES: &str = "on";
const READS: &str = "reads";
// What a behavior does to reach one waiting on a spawn. Every wire label is
// kept short enough to sit in the gap between two cards without being clipped.
const SPAWNS: &str = "spawn";

// The world's behaviors, in the order the panel steps through them. Built in
// passes, because a wire can only be drawn once both ends have a card: every
// behavior, then what fires each of them, then what their bodies do.
pub(crate) fn map(behaviors: &[(String, Value)]) -> Chart {
    let mut build = Build::default();
    for (i, (name, args)) in behaviors.iter().enumerate() {
        build.behavior(i, name, args);
    }
    for (i, (_, args)) in behaviors.iter().enumerate() {
        build.source(i, args);
    }
    for (i, (_, args)) in behaviors.iter().enumerate() {
        build.body(i, args);
    }
    build.finish()
}

// What one behavior's body does that another can see.
#[derive(Default)]
struct Body {
    writes: Vec<(String, bool)>,
    reads: Vec<String>,
    spawns: bool,
}

#[derive(Default)]
struct Build {
    cards: Vec<Card>,
    wires: Vec<Wire>,
    // Card index of each behavior, of each trigger by its caption, and of each
    // variable by name, so a second behavior on the same trigger joins the card
    // the first one made.
    behaviors: Vec<usize>,
    triggers: Vec<(String, usize)>,
    variables: Vec<(String, usize)>,
}

impl Build {
    fn behavior(&mut self, at: usize, name: &str, args: &Value) {
        let card = self.push(Card {
            column: 0,
            row: 0,
            title: name.to_string(),
            detail: scope_summary(args),
            kind: CardKind::Behavior,
            path: Vec::new(),
            behavior: Some(at),
        });
        self.behaviors.push(card);
    }

    // What fires this behavior, reaching its card from the left.
    fn source(&mut self, at: usize, args: &Value) {
        let to = self.behaviors[at];
        let on = args.get("on");
        let verb = on.map_or("start", palette::verb_of);
        match verb {
            // Firing on a variable is the variable's card reaching this one, so
            // the behavior that sets it stands to the left of both.
            "variable" => {
                let name = palette::body_of(on.unwrap_or(&Value::Null))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let from = self.variable(name);
                self.link(from, to, Some(FIRES));
            }
            _ => {
                let from = self.trigger(&outline::source_summary(verb, on));
                self.link(from, to, None);
            }
        }
    }

    // What this behavior's body does that another behavior can see. Run once
    // every trigger has a card, so a spawn can reach one declared later.
    fn body(&mut self, at: usize, args: &Value) {
        let to = self.behaviors[at];
        let body = scan(args);
        for (name, adds) in &body.writes {
            let var = self.variable(name);
            self.link(to, var, Some(if *adds { "adds" } else { "sets" }));
        }
        for name in &body.reads {
            let var = self.variable(name);
            self.link(var, to, Some(READS));
        }
        // A spawn only relates to something when a behavior is waiting on it.
        if body.spawns
            && let Some(spawned) = self.find(&self.triggers, "spawned")
        {
            self.link(to, spawned, Some(SPAWNS));
        }
    }

    fn push(&mut self, card: Card) -> usize {
        self.cards.push(card);
        self.cards.len() - 1
    }

    fn find(&self, index: &[(String, usize)], key: &str) -> Option<usize> {
        index.iter().find(|(k, _)| k == key).map(|(_, i)| *i)
    }

    fn trigger(&mut self, caption: &str) -> usize {
        if let Some(at) = self.find(&self.triggers, caption) {
            return at;
        }
        let card = self.push(Card {
            column: 0,
            row: 0,
            title: caption.to_string(),
            detail: "trigger".to_string(),
            kind: CardKind::Trigger,
            path: Vec::new(),
            behavior: None,
        });
        self.triggers.push((caption.to_string(), card));
        card
    }

    fn variable(&mut self, name: &str) -> usize {
        if let Some(at) = self.find(&self.variables, name) {
            return at;
        }
        let card = self.push(Card {
            column: 0,
            row: 0,
            title: name.to_string(),
            detail: "variable".to_string(),
            kind: CardKind::Variable,
            path: Vec::new(),
            behavior: None,
        });
        self.variables.push((name.to_string(), card));
        card
    }

    // A wire, unless the pair already has one: a behavior that sets a variable
    // twice relates to it once.
    fn link(&mut self, from: usize, to: usize, label: Option<&str>) {
        if self.wires.iter().any(|w| w.from == from && w.to == to) {
            return;
        }
        self.wires.push(Wire {
            from,
            to,
            label: label.map(str::to_string),
        });
    }

    fn finish(mut self) -> Chart {
        let (columns, rows) = place(&mut self.cards, &self.wires);
        Chart {
            cards: self.cards,
            wires: self.wires,
            columns,
            rows,
        }
    }
}

// Put every card one column right of whatever reaches it, then stack the cards
// sharing a column. Repeated until it settles, capped so a cycle of variables
// stops rather than marching off to the right forever.
fn place(cards: &mut [Card], wires: &[Wire]) -> (usize, usize) {
    for _ in 0..cards.len() {
        let mut moved = false;
        for w in wires {
            let want = (cards[w.from].column + 1).min(MAX_COLUMN);
            if cards[w.to].column < want {
                cards[w.to].column = want;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    let mut next = [0usize; MAX_COLUMN + 1];
    for card in cards.iter_mut() {
        card.row = next[card.column];
        next[card.column] += 1;
    }
    let columns = cards.iter().map(|c| c.column + 1).max().unwrap_or(0);
    let rows = cards.iter().map(|c| c.row + 1).max().unwrap_or(0);
    (columns, rows)
}

// The entities a behavior runs against, as its card says it.
fn scope_summary(args: &Value) -> String {
    let scope: Vec<&str> = array(args.get("scope"))
        .iter()
        .filter_map(Value::as_str)
        .collect();
    match scope.is_empty() {
        true => "world-scoped".to_string(),
        false => format!("per {}", scope.join(", ")),
    }
}

fn scan(args: &Value) -> Body {
    let mut body = Body::default();
    scan_nodes(array(args.get("do")), &mut body);
    body
}

fn scan_nodes(nodes: &[Value], body: &mut Body) {
    for node in nodes {
        scan_node(node, body);
    }
}

// A node's own effect, then whatever it holds. `set` is the one node whose
// `var` names a variable being written rather than read, so it is taken apart
// by hand and only its value is scanned for reads.
fn scan_node(node: &Value, body: &mut Body) {
    let verb = palette::verb_of(node);
    let Some(at) = palette::body_of(node) else {
        return;
    };
    let field = |key: &str| at.get(key);
    match verb {
        "set" => {
            if let Some(name) = field("var").and_then(Value::as_str) {
                let adds = field("add").and_then(Value::as_bool).unwrap_or(false);
                body.writes.push((name.to_string(), adds));
            }
            if let Some(value) = field("value") {
                scan_expr(value, body);
            }
        }
        "spawn" => body.spawns = true,
        "if" => {
            if let Some(cond) = field("cond") {
                scan_expr(cond, body);
            }
            for branch in ["then", "else"] {
                scan_nodes(array(field(branch)), body);
            }
        }
        "for_each" => scan_nodes(array(field("do")), body),
        _ => scan_expr(at, body),
    }
}

// Every world variable an expression reads, however deep. `{"var": name}` is
// the only shape that reads one.
fn scan_expr(value: &Value, body: &mut Body) {
    if palette::verb_of(value) == "var"
        && let Some(name) = palette::body_of(value).and_then(Value::as_str)
    {
        body.reads.push(name.to_string());
        return;
    }
    match value {
        Value::Array(items) => items.iter().for_each(|v| scan_expr(v, body)),
        Value::Object(map) => map.values().for_each(|v| scan_expr(v, body)),
        _ => {}
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
    use serde_json::json;

    use super::*;

    fn behaviors(entries: &[(&str, Value)]) -> Vec<(String, Value)> {
        entries
            .iter()
            .map(|(n, a)| ((*n).to_string(), a.clone()))
            .collect()
    }

    fn card(chart: &Chart, title: &str) -> usize {
        chart
            .cards
            .iter()
            .position(|c| c.title == title)
            .unwrap_or_else(|| panic!("no `{title}` card in {:?}", titles(chart)))
    }

    fn titles(chart: &Chart) -> Vec<&str> {
        chart.cards.iter().map(|c| c.title.as_str()).collect()
    }

    fn wire<'a>(chart: &'a Chart, from: &str, to: &str) -> &'a Wire {
        let (a, b) = (card(chart, from), card(chart, to));
        chart
            .wires
            .iter()
            .find(|w| w.from == a && w.to == b)
            .unwrap_or_else(|| panic!("no wire `{from}` -> `{to}`"))
    }

    // The relation the whole view exists for: one behavior writes a variable,
    // another fires on it, and the map says so left to right.
    #[test]
    fn a_variable_joins_the_behavior_that_sets_it_to_the_one_it_fires() {
        let chart = map(&behaviors(&[
            (
                "award",
                json!({"on": "start", "do": [{"set": {"var": "score", "value": {"int": 1}}}]}),
            ),
            ("react", json!({"on": {"variable": "score"}, "do": []})),
        ]));
        assert_eq!(
            wire(&chart, "award", "score").label.as_deref(),
            Some("sets")
        );
        assert_eq!(wire(&chart, "score", "react").label.as_deref(), Some(FIRES));
        // And the chain runs rightward, so cause reads before effect.
        let (award, score, react) = (
            &chart.cards[card(&chart, "award")],
            &chart.cards[card(&chart, "score")],
            &chart.cards[card(&chart, "react")],
        );
        assert!(award.column < score.column && score.column < react.column);
    }

    #[test]
    fn behaviors_sharing_a_trigger_share_its_card() {
        let chart = map(&behaviors(&[
            ("a", json!({"on": "start", "do": []})),
            ("b", json!({"on": "start", "do": []})),
        ]));
        assert_eq!(titles(&chart).iter().filter(|t| **t == "start").count(), 1);
        let start = card(&chart, "start");
        assert_eq!(chart.wires.iter().filter(|w| w.from == start).count(), 2);
    }

    // A card stands for the behavior the panel opens, which is what makes the
    // map navigable rather than just readable.
    #[test]
    fn a_behavior_card_points_back_at_its_behavior() {
        let chart = map(&behaviors(&[
            ("first", json!({"on": "start"})),
            ("second", json!({"on": "tick"})),
        ]));
        assert_eq!(chart.cards[card(&chart, "second")].behavior, Some(1));
        assert_eq!(chart.cards[card(&chart, "start")].behavior, None);
    }

    #[test]
    fn a_variable_only_read_is_still_drawn_reaching_the_behavior() {
        let chart = map(&behaviors(&[(
            "gate",
            json!({"on": "tick", "do": [{"if": {"cond": {"gt": [{"var": "health"}, {"int": 0}]},
                "then": [], "else": []}}]}),
        )]));
        assert_eq!(wire(&chart, "health", "gate").label.as_deref(), Some(READS));
    }

    // A variable read deep inside a nested body is still found, and a `set`'s
    // own `var` is a write rather than a read of itself.
    #[test]
    fn a_set_names_a_write_not_a_read() {
        let chart = map(&behaviors(&[(
            "tally",
            json!({"on": "tick", "do": [{"for_each": {"query": "q", "bind": "e", "do": [
                {"set": {"var": "total", "value": {"add": [{"var": "total"}, {"int": 1}]},
                    "add": false}},
            ]}}]}),
        )]));
        assert_eq!(
            wire(&chart, "tally", "total").label.as_deref(),
            Some("sets")
        );
        assert_eq!(wire(&chart, "total", "tally").label.as_deref(), Some(READS));
    }

    // Two behaviors each firing on what the other sets is a legal world; the
    // map has to place it rather than run off to the right.
    #[test]
    fn a_cycle_settles_instead_of_marching_right() {
        let chart = map(&behaviors(&[
            (
                "ping",
                json!({"on": {"variable": "b"}, "do": [{"set": {"var": "a", "value": {"int": 1}}}]}),
            ),
            (
                "pong",
                json!({"on": {"variable": "a"}, "do": [{"set": {"var": "b", "value": {"int": 1}}}]}),
            ),
        ]));
        assert!(chart.columns <= MAX_COLUMN + 1);
        assert!(chart.cards.iter().all(|c| c.column <= MAX_COLUMN));
    }

    // Spawning relates to something only when a behavior is waiting on it.
    #[test]
    fn a_spawn_reaches_the_behaviors_waiting_on_it() {
        let spawner = json!({"on": "start", "do": [{"spawn": {"template": "drop"}}]});
        let alone = map(&behaviors(&[("drip", spawner.clone())]));
        assert!(!titles(&alone).contains(&"spawned"));

        let paired = map(&behaviors(&[
            ("drip", spawner),
            (
                "greet",
                json!({"on": "spawned", "scope": ["Prop"], "do": []}),
            ),
        ]));
        assert_eq!(
            wire(&paired, "drip", "spawned").label.as_deref(),
            Some(SPAWNS)
        );
        assert_eq!(wire(&paired, "spawned", "greet").label, None);
    }

    #[test]
    fn a_card_says_whether_its_behavior_runs_per_entity() {
        let chart = map(&behaviors(&[
            ("world", json!({"on": "start"})),
            ("each", json!({"on": "tick", "scope": ["Prop"]})),
        ]));
        assert_eq!(chart.cards[card(&chart, "world")].detail, "world-scoped");
        assert_eq!(chart.cards[card(&chart, "each")].detail, "per Prop");
    }

    #[test]
    fn an_empty_world_maps_to_an_empty_chart() {
        let chart = map(&[]);
        assert!(chart.cards.is_empty() && chart.wires.is_empty());
        assert_eq!((chart.columns, chart.rows), (0, 0));
    }

    // Cards in one column stack rather than landing on top of each other.
    #[test]
    fn cards_sharing_a_column_take_their_own_rows() {
        let chart = map(&behaviors(&[
            ("a", json!({"on": "start"})),
            ("b", json!({"on": "start"})),
            ("c", json!({"on": "start"})),
        ]));
        let mut seen: Vec<(usize, usize)> = chart.cards.iter().map(|c| (c.column, c.row)).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), chart.cards.len(), "no two cards share a place");
    }
}
