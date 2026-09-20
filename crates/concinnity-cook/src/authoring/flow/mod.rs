//! Where a world sends itself: the places it can be in and the moves between
//! them.
//!
//! The half of a world's shape that [`refs`](super::refs) deliberately leaves
//! out. A Prop naming a Material is a dependency, and every Prop in a world
//! names the same few; a button naming a scene is the world going somewhere.
//! The two graphs answer different questions and only one of them is worth
//! drawing as a map.
//!
//! A move has two spellings. A world declares a `MainMenu` whose items carry
//! actions, or it declares the `Screen` and `HitRegion`s that menu expands
//! into. Both are read here, so a map built on this cannot show an arrow for
//! one world and nothing for the one beside it.

mod action;
mod edges;
mod places;

pub use action::{Move, parse_action};
pub use edges::{FlowEdge, flow_edges};
pub use places::{Place, PlaceKind, places};

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

/// A world's places and the moves between them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlowGraph {
    /// Every place the world declares, in declaration order.
    pub places: Vec<Place>,
    /// Every move the world declares, in declaration order.
    pub edges: Vec<FlowEdge>,
}

impl FlowGraph {
    /// The place a move leads to, or `None` when it names none or names one the
    /// world does not declare. A name nothing answers to is a build error
    /// waiting to happen, so a caller drawing it says so rather than passing it
    /// off as a place.
    pub fn destination(&self, edge: &FlowEdge) -> Option<&Place> {
        let target = edge.action.target()?;
        self.places.iter().find(|place| place.id == target)
    }

    /// The behaviors that move the world into `place`, by handle, in
    /// declaration order and without repeats.
    ///
    /// A menu item or a key says where it leads from the place it sits on; a
    /// behavior moves the world from wherever it already is, so which ones lead
    /// somewhere is a question the place has to be asked.
    pub fn behaviors_into(&self, place: &str) -> Vec<&str> {
        let mut found: Vec<&str> = Vec::new();
        for edge in &self.edges {
            let leads_here =
                edge.declared_by == RegisteredType::Behavior && edge.action.target() == Some(place);
            if leads_here && !found.contains(&edge.source.as_str()) {
                found.push(&edge.source);
            }
        }
        found
    }

    /// The places the world can start in: those declaring themselves an entry,
    /// or its first scene when none does. A world with no place at all starts
    /// nowhere and answers empty.
    pub fn entries(&self) -> Vec<&Place> {
        let declared: Vec<&Place> = self.places.iter().filter(|place| place.entry).collect();
        if !declared.is_empty() {
            return declared;
        }
        self.places
            .iter()
            .find(|place| place.kind == PlaceKind::Scene)
            .into_iter()
            .collect()
    }
}

/// The flow graph of an authored world.
pub fn flow_graph(assets: &[WorldJsonlAsset]) -> FlowGraph {
    FlowGraph {
        places: places(assets),
        edges: flow_edges(assets),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn asset(ty: RegisteredType, id: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: id.to_string(),
            asset_type: ty,
            args,
        }
    }

    // The world the map exists to draw: a menu on the left, an arrow labelled
    // by the button, the scene it starts on the right.
    fn menu_world() -> Vec<WorldJsonlAsset> {
        vec![
            asset(RegisteredType::Scene, "Scene#0", json!({})),
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(
                RegisteredType::MainMenu,
                "menu",
                json!({"initial": true, "items": [
                    {"label": "Start", "action": "scene:bistro"},
                    {"label": "Quit", "action": "quit"},
                ]}),
            ),
        ]
    }

    #[test]
    fn a_menu_world_maps_to_its_menu_its_scenes_and_the_move_between_them() {
        let graph = flow_graph(&menu_world());
        assert_eq!(
            graph
                .places
                .iter()
                .map(|p| p.id.as_str())
                .collect::<Vec<_>>(),
            ["Scene#0", "bistro", "menu"]
        );
        let start = &graph.edges[0];
        assert_eq!(start.from.as_deref(), Some("menu"));
        assert_eq!(start.label.as_deref(), Some("Start"));
        assert_eq!(
            graph.destination(start).map(|p| p.id.as_str()),
            Some("bistro")
        );
    }

    #[test]
    fn a_move_out_of_the_world_reaches_no_place() {
        let graph = flow_graph(&menu_world());
        assert_eq!(graph.edges[1].action, Move::Quit);
        assert_eq!(graph.destination(&graph.edges[1]), None);
    }

    #[test]
    fn a_move_naming_a_place_the_world_does_not_declare_reaches_none() {
        let graph = flow_graph(&[asset(
            RegisteredType::MainMenu,
            "menu",
            json!({"items": [{"label": "Start", "action": "scene:typo"}]}),
        )]);
        assert_eq!(graph.edges[0].action.target(), Some("typo"));
        assert_eq!(graph.destination(&graph.edges[0]), None);
    }

    #[test]
    fn the_world_starts_where_it_says_it_does() {
        let graph = flow_graph(&menu_world());
        assert_eq!(
            graph
                .entries()
                .iter()
                .map(|p| p.id.as_str())
                .collect::<Vec<_>>(),
            ["menu"]
        );
    }

    // A world with no menu declares no entry, so it starts in its first scene.
    #[test]
    fn a_world_declaring_no_entry_starts_in_its_first_scene() {
        let graph = flow_graph(&[
            asset(RegisteredType::Material, "mat", json!({})),
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(RegisteredType::Scene, "alley", json!({})),
        ]);
        assert_eq!(
            graph
                .entries()
                .iter()
                .map(|p| p.id.as_str())
                .collect::<Vec<_>>(),
            ["bistro"]
        );
    }

    // A world of props and lights has one place and no moves, which is the
    // honest picture of it rather than an empty one.
    #[test]
    fn a_world_with_one_scene_maps_to_one_place_and_no_moves() {
        let graph = flow_graph(&[
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(RegisteredType::Prop, "crate", json!({"scene": "bistro"})),
            asset(RegisteredType::DirectionalLight, "sun", json!({})),
        ]);
        assert_eq!(graph.places.len(), 1);
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn a_world_with_nowhere_to_be_starts_nowhere() {
        let graph = flow_graph(&[asset(RegisteredType::Material, "mat", json!({}))]);
        assert!(graph.entries().is_empty());
    }

    // What sends the world into a place is not always drawable from it: a
    // behavior's move leaves from no place, so the place is asked instead.
    #[test]
    fn a_place_answers_which_behaviors_send_the_world_to_it() {
        let graph = flow_graph(&[
            asset(RegisteredType::Scene, "level", json!({})),
            asset(RegisteredType::Screen, "pause", json!({})),
            asset(
                RegisteredType::Behavior,
                "start_level",
                json!({"do": [
                    {"scene": {"scene": "level"}},
                    {"if": {"cond": true, "then": [{"scene": {"scene": "level"}}]}},
                ]}),
            ),
            asset(
                RegisteredType::Behavior,
                "hold",
                json!({"do": [{"screen": {"screen": "pause"}}]}),
            ),
            asset(
                RegisteredType::HitRegion,
                "play",
                json!({"screen": "pause", "action": "scene:level"}),
            ),
        ]);
        // Twice through one body is one behavior, and the region drawing its
        // own arrow is not one of them.
        assert_eq!(graph.behaviors_into("level"), ["start_level"]);
        assert_eq!(graph.behaviors_into("pause"), ["hold"]);
        assert!(graph.behaviors_into("nowhere").is_empty());
    }

    // Pause menu out to a scene and back is the ordinary case, so the graph has
    // to carry a cycle rather than assume an order that does not exist.
    #[test]
    fn a_cycle_between_two_places_is_carried() {
        let graph = flow_graph(&[
            asset(RegisteredType::Scene, "level", json!({})),
            asset(RegisteredType::Screen, "pause", json!({})),
            asset(
                RegisteredType::HitRegion,
                "resume",
                json!({"screen": "pause", "action": "scene:level"}),
            ),
            asset(
                RegisteredType::KeyBinding,
                "esc",
                json!({"key": "Escape", "action": "screen:toggle:pause"}),
            ),
        ]);
        let reached: Vec<&str> = graph
            .edges
            .iter()
            .filter_map(|e| graph.destination(e))
            .map(|p| p.id.as_str())
            .collect();
        assert_eq!(reached, ["level", "pause"]);
    }
}
