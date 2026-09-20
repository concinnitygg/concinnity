// The cards and wires a world's flow graph makes: a card per place, a wire per
// move, and the relaxation that puts each place right of whatever reaches it.

use concinnity_cook::authoring::flow::{FlowEdge, FlowGraph, Move, Place, PlaceKind};

use super::Entries;
use super::contents;
use crate::editor::asset_handle::AssetHandle;
use crate::editor::behavior::chart::LABEL_CHARS;
use crate::editor::behavior::graph::{Card, CardKind, Chart, Wire};

// Far enough right for any flow worth reading, and the stop the relaxation
// below needs: a pause menu that opens over the scene that opens it is a
// cycle, which has no leftmost order to find.
const MAX_COLUMN: usize = 12;

// The world itself, outside any place it declares: where a move that can be
// made from anywhere leaves from, and the one card a world declaring nowhere to
// be still has.
const WORLD: &str = "world";

/// The map of `graph`, as the chart view draws any other.
pub(super) fn chart(entries: &Entries, graph: &FlowGraph) -> Chart {
    let mut build = Build {
        entries,
        graph,
        cards: Vec::new(),
        wires: Vec::new(),
        pinned: Vec::new(),
        by_place: Vec::new(),
        by_name: Vec::new(),
        world: None,
    };
    build.places();
    build.moves();
    build.finish()
}

struct Build<'a> {
    entries: &'a Entries,
    graph: &'a FlowGraph,
    cards: Vec<Card>,
    wires: Vec<Wire>,
    // The cards the relaxation leaves where they are, which is what keeps the
    // places a world starts in leftmost however much reaches them.
    pinned: Vec<bool>,
    // Card index by place, by a name the world does not declare, and the
    // world's own, so a second move onto one joins the card the first made.
    by_place: Vec<(String, usize)>,
    by_name: Vec<(String, usize)>,
    world: Option<usize>,
}

impl Build<'_> {
    // A card per place, the ones the world starts in first: they take the
    // leftmost column, and a map past the card pool keeps them rather than
    // losing where the world begins.
    fn places(&mut self) {
        let graph = self.graph;
        let starts = graph.entries();
        for place in &starts {
            self.place(place, true);
        }
        for place in &graph.places {
            if !starts.iter().any(|start| start.id == place.id) {
                self.place(place, false);
            }
        }
    }

    // Every move, from the place that declares it to the place it names.
    fn moves(&mut self) {
        let graph = self.graph;
        for edge in &graph.edges {
            let Some(to) = self.destination(edge) else {
                continue;
            };
            let from = self.source(edge);
            self.link(from, to, label(edge));
        }
    }

    fn place(&mut self, place: &Place, start: bool) -> usize {
        let held = contents::held_by(self.entries.assets(), &place.id);
        let kind = noun(place.kind);
        let detail = match held.is_empty() {
            true => kind.to_string(),
            false => format!("{kind}, {held}"),
        };
        let handle = self.entries.handle(place);
        let at = self.push(card(&place.id, detail, CardKind::Asset, handle), start);
        self.by_place.push((place.id.clone(), at));
        at
    }

    // The place a move leaves from, or the world when it leaves from no place:
    // a key on no screen and a behavior alike are live wherever the world is.
    fn source(&mut self, edge: &FlowEdge) -> usize {
        let Some(name) = edge.from.as_deref() else {
            return self.world();
        };
        match find(&self.by_place, name) {
            Some(at) => at,
            None => self.missing(name, noun(PlaceKind::Screen)),
        }
    }

    // The place a move leads to. One naming a place the world does not declare
    // reaches a card of its own rather than being passed off as real, and one
    // naming no place at all -- closing a screen, leaving the world -- reaches
    // nothing a map can draw.
    fn destination(&mut self, edge: &FlowEdge) -> Option<usize> {
        let graph = self.graph;
        if let Some(place) = graph.destination(edge) {
            return find(&self.by_place, &place.id);
        }
        match &edge.action {
            Move::Scene(name) => Some(self.missing(name, noun(PlaceKind::Scene))),
            Move::Show(name) | Move::Push(name) | Move::Toggle(name) => {
                Some(self.missing(name, noun(PlaceKind::Screen)))
            }
            Move::Story => self.story(),
            Move::Back | Move::Quit => None,
        }
    }

    // The world's story, which a story move drives without naming: there is
    // only one, so everything driving it meets at the same card.
    fn story(&mut self) -> Option<usize> {
        let graph = self.graph;
        match graph.places.iter().find(|p| p.kind == PlaceKind::Story) {
            Some(story) => find(&self.by_place, &story.id),
            None => Some(self.missing(noun(PlaceKind::Story), noun(PlaceKind::Story))),
        }
    }

    fn missing(&mut self, name: &str, noun: &str) -> usize {
        if let Some(at) = find(&self.by_name, name) {
            return at;
        }
        let at = self.push(
            card(name, format!("missing {noun}"), CardKind::Missing, None),
            false,
        );
        self.by_name.push((name.to_string(), at));
        at
    }

    fn world(&mut self) -> usize {
        if let Some(at) = self.world {
            return at;
        }
        let held = contents::unplaced(self.entries.assets(), &self.graph.places);
        let at = self.push(card(WORLD, held, CardKind::Asset, None), false);
        self.world = Some(at);
        at
    }

    fn push(&mut self, card: Card, pinned: bool) -> usize {
        self.cards.push(card);
        self.pinned.push(pinned);
        self.cards.len() - 1
    }

    // A wire, unless the pair already carries that word: a menu with two items
    // onto one scene says so once. A move that leads back where it started
    // takes the world nowhere, so it is no wire at all -- and drawing one would
    // march its card rightward a column per pass.
    fn link(&mut self, from: usize, to: usize, label: String) {
        let drawn = |w: &Wire| w.from == from && w.to == to && w.label.as_deref() == Some(&*label);
        if from == to || self.wires.iter().any(drawn) {
            return;
        }
        self.wires.push(Wire {
            from,
            to,
            label: Some(label),
        });
    }

    fn finish(mut self) -> Chart {
        // A world declaring nowhere to be still declares what it is made of, so
        // its map is the card saying so rather than an empty canvas.
        if self.cards.is_empty() {
            self.world();
        }
        let (columns, rows) = lay_out(&mut self.cards, &self.wires, &self.pinned);
        Chart {
            cards: self.cards,
            wires: self.wires,
            columns,
            rows,
        }
    }
}

fn card(title: &str, detail: String, kind: CardKind, handle: Option<AssetHandle>) -> Card {
    Card {
        column: 0,
        row: 0,
        title: title.to_string(),
        detail,
        kind,
        path: Vec::new(),
        settles: Vec::new(),
        behavior: None,
        handle,
    }
}

fn noun(kind: PlaceKind) -> &'static str {
    match kind {
        PlaceKind::Scene => "scene",
        PlaceKind::Screen => "screen",
        PlaceKind::Story => "story",
    }
}

// The word a wire carries: what the author called the move where they named it,
// as a menu item's button text names what its button does. A name wider than
// the gap between two cards is drawn cut short, which says less than the move
// itself does, so a name that wide falls back to the verb.
fn label(edge: &FlowEdge) -> String {
    edge.label
        .as_deref()
        .filter(|name| name.chars().count() <= LABEL_CHARS)
        .unwrap_or_else(|| edge.action.verb())
        .to_string()
}

fn find(index: &[(String, usize)], key: &str) -> Option<usize> {
    index
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, at)| *at)
}

// Put every card one column right of whatever reaches it, leaving the pinned
// ones where they are, then stack the cards sharing a column. Repeated until it
// settles, capped so a cycle stops rather than marching off to the right.
fn lay_out(cards: &mut [Card], wires: &[Wire], pinned: &[bool]) -> (usize, usize) {
    for _ in 0..cards.len() {
        let mut moved = false;
        for wire in wires.iter().filter(|w| !pinned[w.to]) {
            let want = (cards[wire.from].column + 1).min(MAX_COLUMN);
            if cards[wire.to].column < want {
                cards[wire.to].column = want;
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

#[cfg(test)]
mod tests {
    use concinnity_cook::authoring::flow::flow_graph;
    use concinnity_cook::authoring::registry::RegisteredType;
    use concinnity_cook::authoring::world::WorldJsonlAsset;
    use serde_json::{Value, json};

    use super::*;
    use crate::editor::entry_list::EntryList;

    fn asset(ty: RegisteredType, id: &str, args: Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: id.to_string(),
            asset_type: ty,
            args,
        }
    }

    // The map of a world, over an entry list minting a key per entry the way a
    // session does. Only the keys matter here; the map reads the parsed assets.
    fn mapped(assets: Vec<WorldJsonlAsset>) -> Chart {
        let mut list = EntryList::default();
        let entries = Entries::new(assets.into_iter().map(|a| (list.push(Value::Null), a)));
        chart(&entries, &flow_graph(entries.assets()))
    }

    fn at<'a>(chart: &'a Chart, title: &str) -> &'a Card {
        chart
            .cards
            .iter()
            .find(|card| card.title == title)
            .unwrap_or_else(|| panic!("no `{title}` card in {:?}", titles(chart)))
    }

    fn titles(chart: &Chart) -> Vec<&str> {
        chart.cards.iter().map(|c| c.title.as_str()).collect()
    }

    fn wire<'a>(chart: &'a Chart, from: &str, to: &str) -> &'a Wire {
        let index = |title| chart.cards.iter().position(|c| c.title == title).unwrap();
        let (from, to) = (index(from), index(to));
        chart
            .wires
            .iter()
            .find(|w| w.from == from && w.to == to)
            .unwrap_or_else(|| panic!("no wire between those cards: {:?}", chart.wires))
    }

    fn menu(items: Value) -> WorldJsonlAsset {
        asset(RegisteredType::MainMenu, "main", json!({"items": items}))
    }

    // The world the map exists to draw, and the picture it has to make: where
    // the world starts on the left, what the button says on the arrow, and the
    // scene it opens on the right.
    #[test]
    fn a_button_draws_an_arrow_from_its_menu_to_the_scene_it_opens() {
        let chart = mapped(vec![
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(
                RegisteredType::MainMenu,
                "main",
                json!({"initial": true, "items": [{"label": "Start", "action": "scene:bistro"}]}),
            ),
        ]);
        assert_eq!(at(&chart, "main").column, 0);
        assert_eq!(at(&chart, "bistro").column, 1);
        assert_eq!(
            wire(&chart, "main", "bistro").label.as_deref(),
            Some("Start")
        );
    }

    // What the author called the move is what the wire says, until the word is
    // too wide for the gap between two cards: "Se..." says less than "show".
    #[test]
    fn a_wire_says_what_the_author_called_the_move_unless_it_is_too_wide() {
        let chart = mapped(vec![menu(json!([
            {"label": "Start", "action": "scene:bistro"},
            {"label": "Settings", "action": "settings"},
        ]))]);
        let labels: Vec<&str> = chart
            .wires
            .iter()
            .filter_map(|w| w.label.as_deref())
            .collect();
        assert_eq!(labels, ["Start", "show"]);
    }

    // Naming a place is not declaring one, so a name the world does not answer
    // is drawn as its own kind of card rather than passed off as real.
    #[test]
    fn a_move_onto_a_place_the_world_does_not_declare_says_what_is_missing() {
        let chart = mapped(vec![menu(json!([
            {"label": "Start", "action": "scene:typo"},
            {"label": "Help", "action": "screen:push:typo"},
        ]))]);
        assert_eq!(at(&chart, "typo").kind, CardKind::Missing);
        assert_eq!(at(&chart, "typo").detail, "missing scene");
        assert_eq!(at(&chart, "typo").handle, None);
        // And a second move onto the same name meets at the card the first made.
        assert_eq!(chart.cards.len(), 2);
    }

    // A key on no screen and a behavior alike move the world from wherever it
    // already is, which is the world itself rather than any one place.
    #[test]
    fn a_move_from_no_place_leaves_from_the_world() {
        let chart = mapped(vec![
            asset(RegisteredType::Screen, "pause", json!({})),
            asset(
                RegisteredType::KeyBinding,
                "esc",
                json!({"key": "Escape", "action": "screen:toggle:pause"}),
            ),
        ]);
        assert_eq!(
            wire(&chart, "world", "pause").label.as_deref(),
            Some("toggle")
        );
        assert_eq!(at(&chart, "world").handle, None);
    }

    // There is only one story, so everything driving it meets at the same card
    // whether the world declares one or not.
    #[test]
    fn a_story_move_meets_at_the_story_the_world_declares() {
        let teller = asset(
            RegisteredType::Behavior,
            "teller",
            json!({"do": [{"story": {"advance": {}}}]}),
        );
        let told = mapped(vec![
            asset(
                RegisteredType::StoryImport,
                "tale",
                json!({"source": "t.md"}),
            ),
            teller.clone(),
        ]);
        assert_eq!(wire(&told, "world", "tale").label.as_deref(), Some("story"));
        let untold = mapped(vec![teller]);
        assert_eq!(at(&untold, "story").detail, "missing story");
    }

    // Closing a screen uncovers whatever it sat over and quitting leaves the
    // world: neither reaches a place, so neither is an arrow to one.
    #[test]
    fn a_move_reaching_no_place_draws_no_arrow() {
        let chart = mapped(vec![menu(json!([
            {"label": "Back", "action": "return"},
            {"label": "Quit", "action": "quit"},
        ]))]);
        assert_eq!(titles(&chart), ["main"]);
        assert!(chart.wires.is_empty());
    }

    // A screen whose own key closes it moves the world nowhere, and an arrow
    // from a card to itself would march that card off to the right.
    #[test]
    fn a_move_back_to_where_it_started_draws_no_arrow() {
        let chart = mapped(vec![
            asset(RegisteredType::Screen, "pause", json!({})),
            asset(
                RegisteredType::HitRegion,
                "close",
                json!({"screen": "pause", "action": "screen:toggle:pause"}),
            ),
        ]);
        assert_eq!(at(&chart, "pause").column, 0);
        assert!(chart.wires.is_empty());
    }

    // Where the world starts is the left edge of the map, which is only true if
    // what reaches it cannot push it rightward.
    #[test]
    fn the_place_the_world_starts_in_keeps_the_leftmost_column() {
        let chart = mapped(vec![
            asset(RegisteredType::Screen, "menu", json!({"initial": true})),
            asset(RegisteredType::Scene, "level", json!({})),
            asset(
                RegisteredType::HitRegion,
                "play",
                json!({"screen": "menu", "action": "scene:level"}),
            ),
            asset(
                RegisteredType::HitRegion,
                "give_up",
                json!({"screen": "level", "action": "screen:show:menu"}),
            ),
        ]);
        assert_eq!(at(&chart, "menu").column, 0);
        assert_eq!(at(&chart, "level").column, 1);
    }

    // Screens that open each other round are an ordinary world, so the map has
    // to place them rather than march off to the right forever.
    #[test]
    fn a_cycle_settles_instead_of_marching_right() {
        let mut world = Vec::new();
        for (screen, next) in [("a", "b"), ("b", "c"), ("c", "a")] {
            world.push(asset(RegisteredType::Screen, screen, json!({})));
            world.push(asset(
                RegisteredType::HitRegion,
                &format!("{screen}_go"),
                json!({"screen": screen, "action": format!("screen:show:{next}")}),
            ));
        }
        let chart = mapped(world);
        assert_eq!(chart.cards.len(), 3);
        assert!(chart.columns <= MAX_COLUMN + 1, "{}", chart.columns);
    }

    // Two items onto one scene relate it to the menu once, but two differently
    // named ways of getting there each keep their word.
    #[test]
    fn one_place_reached_twice_the_same_way_is_drawn_once() {
        let chart = mapped(vec![
            asset(RegisteredType::Scene, "level", json!({})),
            menu(json!([
                {"label": "Play", "action": "scene:level"},
                {"label": "Play", "action": "scene:level"},
                {"label": "Again", "action": "screen:show:level"},
            ])),
        ]);
        assert_eq!(chart.wires.len(), 2);
    }

    // Every card sits somewhere, and two cards never sit in the same place.
    #[test]
    fn cards_stack_down_the_column_they_share() {
        let chart = mapped(vec![
            asset(RegisteredType::Scene, "a", json!({})),
            asset(RegisteredType::Scene, "b", json!({})),
            asset(RegisteredType::Screen, "c", json!({})),
        ]);
        let places: Vec<(usize, usize)> = chart.cards.iter().map(|c| (c.column, c.row)).collect();
        assert_eq!(places, [(0, 0), (0, 1), (0, 2)]);
        assert_eq!((chart.columns, chart.rows), (1, 3));
    }
}
