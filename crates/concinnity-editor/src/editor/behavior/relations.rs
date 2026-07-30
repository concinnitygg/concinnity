// src/editor/behavior/relations.rs
//
// The world's behaviors as one map, so what a body does is legible next to what
// it sets off. A behavior on its own says what it runs; it does not say what
// started it or what it starts, and that is the part a panel showing one
// behavior at a time cannot answer.
//
// Behaviors reach each other through the world, never directly: one writes a
// variable another fires on, spawns an entity another picks up, or sends the
// world somewhere a third is waiting. So the map draws those middlemen as cards
// of their own -- a trigger, a variable, a world asset -- and a chain reads left
// to right, from what starts something to what it starts.
//
// An asset earns a card where reaching it is itself a relation. What fires a
// behavior (a trigger volume, an interactable prop) and where a behavior sends
// the world (a scene, a screen, the story) always are. An entity a body merely
// acts on is one only once a second behavior reaches the same entity: a card per
// named prop would bury the couplings worth reading under the ones that are just
// one body's own business.
//
// Naming an asset is not declaring one, so a name the world does not answer is
// drawn as its own kind of card rather than passed off as real: it is a build
// error waiting to happen, and two behaviors sharing a typo still share it.
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
const SETS: &str = "sets";
const ADDS: &str = "adds";
// What a behavior does to reach one waiting on a spawn.
const SPAWNS: &str = "spawn";
// What a world asset does to the behavior it fires.
const ENTERS: &str = "enter";
const EXITS: &str = "exit";
const INTERACTS: &str = "use";
// What a behavior does to the asset a wire reaches. Every label is kept short
// enough to sit in the gap between two cards without being clipped, which
// `behavior_chart::LABEL_CHARS` fixes and the tests below hold it to.
const JUMPS: &str = "jumps";
const SHOWS: &str = "shows";
const PLAYS: &str = "plays";
const HIDES: &str = "hides";
const ENDS: &str = "ends";
const PINS: &str = "pins";
const MOVES: &str = "moves";

// Every word the map puts on a wire, listed where they are declared so nothing
// joins the set without being held to the width one can be drawn at.
#[cfg(test)]
const LABELS: &[&str] = &[
    FIRES, READS, SETS, ADDS, SPAWNS, ENTERS, EXITS, INTERACTS, JUMPS, SHOWS, PLAYS, HIDES, ENDS,
    PINS, MOVES,
];

// What a name has to be for the reference to hold. `Entity` takes any declared
// asset, which is what the build resolves a `named` target against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ref {
    Volume,
    Scene,
    Screen,
    Entity,
}

impl Ref {
    // The type the world has to declare the name as, or `None` when any does.
    fn asset_type(self) -> Option<&'static str> {
        match self {
            Ref::Volume => Some("TriggerVolume"),
            Ref::Scene => Some("Scene"),
            Ref::Screen => Some("Screen"),
            Ref::Entity => None,
        }
    }

    // What the card says the reference wanted, when the world has no such name.
    fn noun(self) -> &'static str {
        match self {
            Ref::Volume => "volume",
            Ref::Scene => "scene",
            Ref::Screen => "screen",
            Ref::Entity => "entity",
        }
    }
}

// The world's behaviors, in the order the panel steps through them, against the
// name and type of every entry the world declares -- which is what tells an
// asset a behavior reaches from a name nothing answers to. Built in passes,
// because a wire can only be drawn once both ends have a card: every behavior,
// then what fires each of them, then what their bodies do.
pub(crate) fn map(behaviors: &[(String, Value)], world: &[(&str, &str)]) -> Chart {
    let mut build = Build {
        world,
        shared: shared_entities(behaviors),
        ..Build::default()
    };
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

// The entities more than one behavior reaches for. One behavior acting on an
// entity is its own business; two meeting at the same one is the relation the
// map exists to show, so only those are worth a card.
fn shared_entities(behaviors: &[(String, Value)]) -> Vec<String> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for (_, args) in behaviors {
        let mut names: Vec<String> = scan(args)
            .reaches
            .into_iter()
            .filter(|r| r.kind == Ref::Entity)
            .map(|r| r.name)
            .collect();
        names.sort_unstable();
        names.dedup();
        for name in names {
            match counts.iter_mut().find(|(n, _)| *n == name) {
                Some((_, seen)) => *seen += 1,
                None => counts.push((name, 1)),
            }
        }
    }
    counts
        .into_iter()
        .filter(|(_, seen)| *seen > 1)
        .map(|(name, _)| name)
        .collect()
}

// One world asset a body names, and the word the wire to it carries.
struct Reach {
    name: String,
    kind: Ref,
    label: &'static str,
}

// What one behavior's body does that another can see.
#[derive(Default)]
struct Body {
    writes: Vec<(String, bool)>,
    reads: Vec<String>,
    spawns: bool,
    reaches: Vec<Reach>,
    // Whether it drives the story, which it names nothing to reach.
    story: bool,
}

impl Body {
    fn reach(&mut self, name: Option<&str>, kind: Ref, label: &'static str) {
        if let Some(name) = name.filter(|n| !n.is_empty()) {
            self.reaches.push(Reach {
                name: name.to_string(),
                kind,
                label,
            });
        }
    }
}

#[derive(Default)]
struct Build<'a> {
    cards: Vec<Card>,
    wires: Vec<Wire>,
    // Card index of each behavior, of each trigger by its caption, and of each
    // variable and asset by name, so a second behavior on the same trigger joins
    // the card the first one made.
    behaviors: Vec<usize>,
    triggers: Vec<(String, usize)>,
    variables: Vec<(String, usize)>,
    assets: Vec<(String, usize)>,
    // Every entry's name and type, and the entities more than one behavior
    // reaches.
    world: &'a [(&'a str, &'a str)],
    shared: Vec<String>,
}

impl Build<'_> {
    fn behavior(&mut self, at: usize, name: &str, args: &Value) {
        let mut card = card(name, scope_summary(args), CardKind::Behavior);
        card.behavior = Some(at);
        let card = self.push(card);
        self.behaviors.push(card);
    }

    // What fires this behavior, reaching its card from the left.
    fn source(&mut self, at: usize, args: &Value) {
        let to = self.behaviors[at];
        let on = args.get("on");
        let verb = on.map_or("start", palette::verb_of);
        let named = on
            .and_then(palette::body_of)
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty());
        match (verb, named) {
            // Firing on a variable is the variable's card reaching this one, so
            // the behavior that sets it stands to the left of both.
            ("variable", _) => {
                let name = palette::body_of(on.unwrap_or(&Value::Null))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let from = self.variable(name);
                self.link(from, to, Some(FIRES));
            }
            // What fires the behavior is a world asset, so the asset is the
            // card: two behaviors watching the same volume meet there, and so
            // does whatever else in the world reaches it.
            ("enter" | "exit", Some(name)) => {
                let from = self.asset(name, Ref::Volume);
                let label = if verb == "enter" { ENTERS } else { EXITS };
                self.link(from, to, Some(label));
            }
            ("interact", Some(name)) => {
                let from = self.asset(name, Ref::Entity);
                self.link(from, to, Some(INTERACTS));
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
            self.link(to, var, Some(if *adds { ADDS } else { SETS }));
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
        for reach in &body.reaches {
            if reach.kind == Ref::Entity && !self.meets(&reach.name) {
                continue;
            }
            let asset = self.asset(&reach.name, reach.kind);
            self.link(to, asset, Some(reach.label));
        }
        if body.story {
            let story = self.story();
            self.link(to, story, Some(PLAYS));
        }
    }

    // Whether an entity is one the map draws: reached by more than one behavior,
    // or already a card because something in the world fires on it.
    fn meets(&self, name: &str) -> bool {
        self.shared.iter().any(|n| n == name) || self.find(&self.assets, name).is_some()
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
        let card = self.push(card(caption, "trigger".to_string(), CardKind::Trigger));
        self.triggers.push((caption.to_string(), card));
        card
    }

    fn variable(&mut self, name: &str) -> usize {
        if let Some(at) = self.find(&self.variables, name) {
            return at;
        }
        let card = self.push(card(name, "variable".to_string(), CardKind::Variable));
        self.variables.push((name.to_string(), card));
        card
    }

    // The asset `name` addresses. A card says which type the world declares it
    // as, or that the world declares nothing by that name at all.
    fn asset(&mut self, name: &str, want: Ref) -> usize {
        if let Some(at) = self.find(&self.assets, name) {
            return at;
        }
        let declared = self
            .world
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, ty)| *ty)
            .filter(|ty| want.asset_type().is_none_or(|wanted| wanted == *ty));
        let card = self.push(match declared {
            Some(ty) => card(name, ty.to_string(), CardKind::Asset),
            None => card(name, format!("missing {}", want.noun()), CardKind::Missing),
        });
        self.assets.push((name.to_string(), card));
        card
    }

    // The world's story, which a `story` node drives without naming: there is
    // only one, so every behavior driving it meets at the same card. An authored
    // world declares the import and a built one the story it expands to.
    fn story(&mut self) -> usize {
        let declared = self
            .world
            .iter()
            .find(|(_, ty)| *ty == "Story" || *ty == "StoryImport")
            .map(|(name, _)| *name);
        let name = declared.unwrap_or("story");
        if let Some(at) = self.find(&self.assets, name) {
            return at;
        }
        let card = self.push(match declared {
            Some(_) => card(name, "story".to_string(), CardKind::Asset),
            None => card(name, "missing story".to_string(), CardKind::Missing),
        });
        self.assets.push((name.to_string(), card));
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

    // Middlemen first, behaviors after, each keeping the order it was built in.
    // What a map too big for the card pool loses is then a behavior, which the
    // panel opens either way; losing a middleman would take every wire through
    // it as well, which is the whole of what the map has to say.
    fn order(&mut self) {
        let mut from: Vec<usize> = (0..self.cards.len()).collect();
        from.sort_by_key(|&i| self.cards[i].behavior.is_some());
        let mut to = vec![0; self.cards.len()];
        for (moved, &card) in from.iter().enumerate() {
            to[card] = moved;
        }
        self.cards = from.iter().map(|&i| self.cards[i].clone()).collect();
        for wire in &mut self.wires {
            wire.from = to[wire.from];
            wire.to = to[wire.to];
        }
    }

    fn finish(mut self) -> Chart {
        self.order();
        let (columns, rows) = place(&mut self.cards, &self.wires);
        Chart {
            cards: self.cards,
            wires: self.wires,
            columns,
            rows,
        }
    }
}

// A card of the map: one of the world's own things, placed by `place` below.
fn card(title: &str, detail: String, kind: CardKind) -> Card {
    Card {
        column: 0,
        row: 0,
        title: title.to_string(),
        detail,
        kind,
        path: Vec::new(),
        settles: Vec::new(),
        behavior: None,
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

// Every variable name the given behaviors read or write, sorted and without
// repeats. What the world's table has to account for, which is how the panel
// editing that table can say what is missing from it.
pub(crate) fn variables_used(behaviors: &[(String, Value)]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for (_, args) in behaviors {
        let body = scan(args);
        let watched = args
            .get("on")
            .and_then(|v| v.get("variable"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let touched = body
            .writes
            .into_iter()
            .map(|(name, _)| name)
            .chain(body.reads)
            .chain(watched);
        for name in touched.filter(|n| !n.is_empty()) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names.sort();
    names
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
        _ => {
            scan_reaches(verb, at, body);
            scan_expr(at, body);
        }
    }
}

// The world a node reaches out to. An entity addressed as anything but a
// `named` expression -- `self`, a binding, whatever a query turned up -- is not
// a name another behavior can be met at, so only `named` counts.
fn scan_reaches(verb: &str, at: &Value, body: &mut Body) {
    let name = |key: &str| at.get(key).and_then(Value::as_str);
    let entity = |key: &str| at.get(key).and_then(named_of);
    match verb {
        "scene" => body.reach(name("scene"), Ref::Scene, JUMPS),
        "screen" => body.reach(name("screen"), Ref::Screen, SHOWS),
        "story" => body.story = true,
        "hide" => body.reach(entity("target"), Ref::Entity, HIDES),
        "show" => body.reach(entity("target"), Ref::Entity, SHOWS),
        "despawn" => body.reach(entity("target"), Ref::Entity, ENDS),
        "set_transform" => body.reach(entity("entity"), Ref::Entity, MOVES),
        "reparent" => {
            body.reach(entity("child"), Ref::Entity, PINS);
            body.reach(entity("parent"), Ref::Entity, PINS);
        }
        _ => {}
    }
}

// The asset a `{"named": ...}` expression addresses.
fn named_of(value: &Value) -> Option<&str> {
    (palette::verb_of(value) == "named")
        .then(|| palette::body_of(value).and_then(Value::as_str))
        .flatten()
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

    // The map of a world declaring nothing but its behaviors, which is what
    // every relation carried by a variable or a trigger needs.
    fn mapped(entries: &[(&str, Value)]) -> Chart {
        map(&behaviors(entries), &[])
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
        let chart = mapped(&[
            (
                "award",
                json!({"on": "start", "do": [{"set": {"var": "score", "value": {"int": 1}}}]}),
            ),
            ("react", json!({"on": {"variable": "score"}, "do": []})),
        ]);
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
        let chart = mapped(&[
            ("a", json!({"on": "start", "do": []})),
            ("b", json!({"on": "start", "do": []})),
        ]);
        assert_eq!(titles(&chart).iter().filter(|t| **t == "start").count(), 1);
        let start = card(&chart, "start");
        assert_eq!(chart.wires.iter().filter(|w| w.from == start).count(), 2);
    }

    // A card stands for the behavior the panel opens, which is what makes the
    // map navigable rather than just readable.
    #[test]
    fn a_behavior_card_points_back_at_its_behavior() {
        let chart = mapped(&[
            ("first", json!({"on": "start"})),
            ("second", json!({"on": "tick"})),
        ]);
        assert_eq!(chart.cards[card(&chart, "second")].behavior, Some(1));
        assert_eq!(chart.cards[card(&chart, "start")].behavior, None);
    }

    #[test]
    fn a_variable_only_read_is_still_drawn_reaching_the_behavior() {
        let chart = mapped(&[(
            "gate",
            json!({"on": "tick", "do": [{"if": {"cond": {"gt": [{"var": "health"}, {"int": 0}]},
                "then": [], "else": []}}]}),
        )]);
        assert_eq!(wire(&chart, "health", "gate").label.as_deref(), Some(READS));
    }

    // A variable read deep inside a nested body is still found, and a `set`'s
    // own `var` is a write rather than a read of itself.
    #[test]
    fn a_set_names_a_write_not_a_read() {
        let chart = mapped(&[(
            "tally",
            json!({"on": "tick", "do": [{"for_each": {"query": "q", "bind": "e", "do": [
                {"set": {"var": "total", "value": {"add": [{"var": "total"}, {"int": 1}]},
                    "add": false}},
            ]}}]}),
        )]);
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
        let chart = mapped(&[
            (
                "ping",
                json!({"on": {"variable": "b"}, "do": [{"set": {"var": "a", "value": {"int": 1}}}]}),
            ),
            (
                "pong",
                json!({"on": {"variable": "a"}, "do": [{"set": {"var": "b", "value": {"int": 1}}}]}),
            ),
        ]);
        assert!(chart.columns <= MAX_COLUMN + 1);
        assert!(chart.cards.iter().all(|c| c.column <= MAX_COLUMN));
    }

    // Spawning relates to something only when a behavior is waiting on it.
    #[test]
    fn a_spawn_reaches_the_behaviors_waiting_on_it() {
        let spawner = json!({"on": "start", "do": [{"spawn": {"template": "drop"}}]});
        let alone = mapped(&[("drip", spawner.clone())]);
        assert!(!titles(&alone).contains(&"spawned"));

        let paired = mapped(&[
            ("drip", spawner),
            (
                "greet",
                json!({"on": "spawned", "scope": ["Prop"], "do": []}),
            ),
        ]);
        assert_eq!(
            wire(&paired, "drip", "spawned").label.as_deref(),
            Some(SPAWNS)
        );
        assert_eq!(wire(&paired, "spawned", "greet").label, None);
    }

    #[test]
    fn a_card_says_whether_its_behavior_runs_per_entity() {
        let chart = mapped(&[
            ("world", json!({"on": "start"})),
            ("each", json!({"on": "tick", "scope": ["Prop"]})),
        ]);
        assert_eq!(chart.cards[card(&chart, "world")].detail, "world-scoped");
        assert_eq!(chart.cards[card(&chart, "each")].detail, "per Prop");
    }

    #[test]
    fn an_empty_world_maps_to_an_empty_chart() {
        let chart = map(&[], &[]);
        assert!(chart.cards.is_empty() && chart.wires.is_empty());
        assert_eq!((chart.columns, chart.rows), (0, 0));
    }

    // A label wider than the gap between two cards is drawn clipped, which is
    // how "spawns" once reached the canvas as "sp...". The whole set is held to
    // what the gap can draw rather than each new word being eyeballed.
    #[test]
    fn every_wire_label_fits_the_gap_it_is_drawn_in() {
        for label in LABELS {
            assert!(
                label.chars().count() <= crate::editor::behavior_chart::LABEL_CHARS,
                "`{label}` is too wide to draw on a wire",
            );
        }
    }

    // Two behaviors watching the same volume are related through it, whichever
    // way each of them watches it.
    #[test]
    fn behaviors_watching_one_volume_meet_at_its_card() {
        let chart = map(
            &behaviors(&[
                ("arrive", json!({"on": {"enter": "door_zone"}, "do": []})),
                ("leave", json!({"on": {"exit": "door_zone"}, "do": []})),
            ]),
            &[("door_zone", "TriggerVolume")],
        );
        assert_eq!(
            titles(&chart),
            vec!["door_zone", "arrive", "leave"],
            "one card for the volume, not one per way in",
        );
        assert_eq!(chart.cards[card(&chart, "door_zone")].kind, CardKind::Asset);
        assert_eq!(
            chart.cards[card(&chart, "door_zone")].detail,
            "TriggerVolume"
        );
        assert_eq!(
            wire(&chart, "door_zone", "arrive").label.as_deref(),
            Some(ENTERS)
        );
        assert_eq!(
            wire(&chart, "door_zone", "leave").label.as_deref(),
            Some(EXITS)
        );
    }

    // Where a behavior sends the world is a relation on its own: the scene is
    // the map's rightmost card, and everything routing there meets at it.
    #[test]
    fn where_a_behavior_sends_the_world_is_a_card_of_its_own() {
        let chart = map(
            &behaviors(&[
                (
                    "finish",
                    json!({"on": "start", "do": [{"scene": {"scene": "hub", "transition": "Cut"}}]}),
                ),
                (
                    "quit",
                    json!({"on": "tick", "do": [
                        {"scene": {"scene": "hub"}},
                        {"screen": {"screen": "pause"}},
                    ]}),
                ),
            ]),
            &[("hub", "Scene"), ("pause", "Screen")],
        );
        assert_eq!(wire(&chart, "finish", "hub").label.as_deref(), Some(JUMPS));
        assert_eq!(wire(&chart, "quit", "hub").label.as_deref(), Some(JUMPS));
        assert_eq!(wire(&chart, "quit", "pause").label.as_deref(), Some(SHOWS));
        assert_eq!(chart.cards[card(&chart, "hub")].detail, "Scene");
        // Both behaviors reach it, so it settles to the right of both.
        let hub = chart.cards[card(&chart, "hub")].column;
        assert!(hub > chart.cards[card(&chart, "finish")].column);
        assert!(hub > chart.cards[card(&chart, "quit")].column);
    }

    // The world's story is one thing however many behaviors drive it, and none
    // of them names it, so the map takes its name from the world.
    #[test]
    fn the_worlds_story_is_the_card_everything_driving_it_meets_at() {
        let driving = behaviors(&[
            ("open", json!({"on": "start", "do": [{"story": "start"}]})),
            (
                "resume",
                json!({"on": "tick", "do": [{"story": "continue"}]}),
            ),
        ]);
        let chart = map(&driving, &[("tale", "StoryImport")]);
        assert_eq!(chart.cards[card(&chart, "tale")].detail, "story");
        assert_eq!(wire(&chart, "open", "tale").label.as_deref(), Some(PLAYS));
        assert_eq!(wire(&chart, "resume", "tale").label.as_deref(), Some(PLAYS));
        // A world with no story to drive says so rather than inventing one.
        let none = map(&driving, &[]);
        assert_eq!(none.cards[card(&none, "story")].kind, CardKind::Missing);
    }

    // A name the world does not declare is a build error waiting to happen, so
    // the map draws it as one rather than as an asset that is really there.
    #[test]
    fn a_name_the_world_does_not_declare_is_drawn_as_missing() {
        let chart = map(
            &behaviors(&[(
                "escape",
                json!({"on": {"enter": "porch"}, "do": [{"scene": {"scene": "hubb"}}]}),
            )]),
            &[("hub", "Scene"), ("porch", "Prop")],
        );
        for (name, detail) in [("hubb", "missing scene"), ("porch", "missing volume")] {
            let at = &chart.cards[card(&chart, name)];
            assert_eq!(at.kind, CardKind::Missing, "{name}");
            assert_eq!(at.detail, detail, "{name}");
        }
        // Wrong type or no declaration at all, the wire still reads the same.
        assert_eq!(wire(&chart, "escape", "hubb").label.as_deref(), Some(JUMPS));
    }

    // An entity one behavior acts on is that body's own business; the same
    // entity in a second body is the relation worth drawing.
    #[test]
    fn an_entity_is_a_card_once_a_second_behavior_reaches_it() {
        let shut = (
            "shut",
            json!({"on": "start", "do": [{"hide": {"target": {"named": "door"}}}]}),
        );
        let alone = map(&behaviors(std::slice::from_ref(&shut)), &[("door", "Prop")]);
        assert!(!titles(&alone).contains(&"door"), "{:?}", titles(&alone));

        let both = map(
            &behaviors(&[
                shut,
                (
                    "open",
                    json!({"on": "tick", "do": [
                        {"show": {"target": {"named": "door"}}},
                        {"set_transform": {"entity": {"named": "door"}, "scale": null}},
                        {"despawn": {"target": "self"}},
                    ]}),
                ),
            ]),
            &[("door", "Prop")],
        );
        assert_eq!(both.cards[card(&both, "door")].detail, "Prop");
        assert_eq!(wire(&both, "shut", "door").label.as_deref(), Some(HIDES));
        // The pair already has a wire, so the first thing the second behavior
        // does to it is what the wire says.
        assert_eq!(wire(&both, "open", "door").label.as_deref(), Some(SHOWS));
        // `self` names nothing another behavior could be met at.
        assert_eq!(both.cards.len(), 5, "{:?}", titles(&both));
    }

    // An entity the world already fires on has a card whatever else touches it,
    // so a body reaching the same one joins there rather than being dropped for
    // want of a second behavior.
    #[test]
    fn a_body_reaching_what_fires_it_joins_that_card() {
        let chart = map(
            &behaviors(&[(
                "toggle",
                json!({"on": {"interact": "lamp"}, "do": [{"hide": {"target": {"named": "lamp"}}}]}),
            )]),
            &[("lamp", "Prop")],
        );
        assert_eq!(
            wire(&chart, "lamp", "toggle").label.as_deref(),
            Some(INTERACTS)
        );
        assert_eq!(wire(&chart, "toggle", "lamp").label.as_deref(), Some(HIDES));
    }

    // The card pool draws the map from the front, so what a map too big for it
    // loses has to be the behaviors: a middleman dropped takes every wire
    // through it, and a behavior dropped is still one the panel can open.
    #[test]
    fn the_middlemen_come_before_the_behaviors_they_join() {
        let world: Vec<(&str, Value)> = (0..8)
            .map(|_| ("watch", json!({"on": {"variable": "score"}, "do": []})))
            .collect();
        let chart = mapped(&world);
        let first = chart
            .cards
            .iter()
            .position(|c| c.behavior.is_some())
            .expect("the world has behaviors");
        assert!(
            chart.cards[first..].iter().all(|c| c.behavior.is_some()),
            "{:?}",
            chart.cards.iter().map(|c| c.kind).collect::<Vec<_>>(),
        );
        // And the wires still join the cards they did before the reordering.
        assert_eq!(wire(&chart, "score", "watch").label.as_deref(), Some(FIRES));
    }

    // Cards in one column stack rather than landing on top of each other.
    #[test]
    fn cards_sharing_a_column_take_their_own_rows() {
        let chart = mapped(&[
            ("a", json!({"on": "start"})),
            ("b", json!({"on": "start"})),
            ("c", json!({"on": "start"})),
        ]);
        let mut seen: Vec<(usize, usize)> = chart.cards.iter().map(|c| (c.column, c.row)).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), chart.cards.len(), "no two cards share a place");
    }
}
