// What moves a world between its places, read off its authored entries.

use serde_json::Value;

use super::action::{Move, parse_action};
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use crate::build_only::main_menu::item_action;

/// One move the world can make, and what declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowEdge {
    /// The place the declaring entry sits in, or `None` when it sits in none: a
    /// key binding on no screen, a behavior, anything the world can reach from
    /// wherever it already is.
    pub from: Option<String>,
    /// The move made.
    pub action: Move,
    /// The handle of the entry declaring it.
    pub source: String,
    /// That entry's type, so a map can say what kind of thing moves the world.
    pub declared_by: RegisteredType,
    /// What the move is called where it is named, as a menu item's label names
    /// what its button does.
    pub label: Option<String>,
}

/// Every move `assets` declares, in declaration order.
///
/// A menu's items and the hit regions that menu expands into declare the same
/// moves, so a world that authored its menu by hand maps like the one that
/// declared the shorthand.
pub fn flow_edges(assets: &[WorldJsonlAsset]) -> Vec<FlowEdge> {
    let mut edges = Vec::new();
    for asset in assets {
        match asset.asset_type {
            RegisteredType::HitRegion | RegisteredType::KeyBinding => region(asset, &mut edges),
            RegisteredType::MainMenu => menu(asset, &mut edges),
            RegisteredType::Behavior => behavior(asset, &mut edges),
            _ => {}
        }
    }
    edges
}

/// The move a menu item's action makes, with the menu's conveniences resolved
/// against it. Shared with [`places`](super::places), which needs to know
/// whether an item opens the settings screen the menu generates.
pub(super) fn menu_action(menu: &str, action: &str) -> Option<Move> {
    parse_action(&item_action(menu, action))
}

// A clickable region or a key: the action it fires, from the screen it is on.
fn region(asset: &WorldJsonlAsset, edges: &mut Vec<FlowEdge>) {
    let Some(action) = text(&asset.args, "action").and_then(parse_action) else {
        return;
    };
    edges.push(FlowEdge {
        from: text(&asset.args, "screen").map(str::to_string),
        action,
        source: asset.id.clone(),
        declared_by: asset.asset_type,
        label: None,
    });
}

// Every item of a menu, from the screen the menu expands into -- which carries
// the menu's own handle, so both spellings of a menu land on one place.
fn menu(asset: &WorldJsonlAsset, edges: &mut Vec<FlowEdge>) {
    let items = asset.args.get("items").and_then(Value::as_array);
    for item in items.into_iter().flatten() {
        let Some(action) = text(item, "action").and_then(|a| menu_action(&asset.id, a)) else {
            continue;
        };
        edges.push(FlowEdge {
            from: Some(asset.id.clone()),
            action,
            source: asset.id.clone(),
            declared_by: RegisteredType::MainMenu,
            label: text(item, "label").map(str::to_string),
        });
    }
}

// A behavior's body: the nodes that send the world somewhere. A behavior is on
// no screen, so its moves are reachable from wherever the world is.
fn behavior(asset: &WorldJsonlAsset, edges: &mut Vec<FlowEdge>) {
    let mut moves = Vec::new();
    walk_body(asset.args.get("do"), &mut moves);
    for action in moves {
        edges.push(FlowEdge {
            from: None,
            action,
            source: asset.id.clone(),
            declared_by: RegisteredType::Behavior,
            label: None,
        });
    }
}

// Nodes are single-key objects keyed by their verb, and a branching node holds
// its own lists, so a move is found wherever it is nested.
fn walk_body(body: Option<&Value>, moves: &mut Vec<Move>) {
    let Some(nodes) = body.and_then(Value::as_array) else {
        return;
    };
    for node in nodes {
        let Some(node) = node.as_object().filter(|n| n.len() == 1) else {
            continue;
        };
        for (verb, body) in node {
            match verb.as_str() {
                "scene" => moves.extend(text(body, "scene").map(|n| Move::Scene(n.to_string()))),
                "screen" => moves.extend(text(body, "screen").map(|n| Move::Show(n.to_string()))),
                "story" => moves.push(Move::Story),
                _ => {}
            }
            walk_body(body.get("do"), moves);
            walk_body(body.get("else"), moves);
        }
    }
}

// A non-empty string field. An empty one names nothing, which is how an author
// writes "no screen" and "no action" alike.
fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn asset(ty: RegisteredType, id: &str, args: Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: id.to_string(),
            asset_type: ty,
            args,
        }
    }

    fn actions(edges: &[FlowEdge]) -> Vec<&Move> {
        edges.iter().map(|e| &e.action).collect()
    }

    #[test]
    fn a_hit_region_moves_from_the_screen_it_is_on() {
        let world = [asset(
            RegisteredType::HitRegion,
            "start_btn",
            json!({"screen": "menu", "action": "scene:bistro"}),
        )];
        let edges = flow_edges(&world);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from.as_deref(), Some("menu"));
        assert_eq!(edges[0].action, Move::Scene("bistro".to_string()));
        assert_eq!(edges[0].source, "start_btn");
    }

    // A binding on no screen is live wherever the world is, which is what an
    // Escape key that opens a pause menu has to be.
    #[test]
    fn a_key_binding_on_no_screen_moves_from_nowhere() {
        let world = [asset(
            RegisteredType::KeyBinding,
            "esc",
            json!({"key": "Escape", "action": "screen:toggle:pause"}),
        )];
        let edges = flow_edges(&world);
        assert_eq!(edges[0].from, None);
        assert_eq!(edges[0].action, Move::Toggle("pause".to_string()));
    }

    #[test]
    fn an_entry_declaring_no_action_declares_no_move() {
        let world = [
            asset(RegisteredType::HitRegion, "a", json!({"screen": "menu"})),
            asset(RegisteredType::HitRegion, "b", json!({"action": ""})),
            asset(
                RegisteredType::HitRegion,
                "c",
                json!({"action": "setting:volume_master:next"}),
            ),
            asset(RegisteredType::Prop, "d", json!({"action": "quit"})),
        ];
        assert!(flow_edges(&world).is_empty());
    }

    #[test]
    fn a_menu_item_moves_from_the_menu_and_carries_its_label() {
        let world = [asset(
            RegisteredType::MainMenu,
            "main",
            json!({"items": [
                {"label": "Start", "action": "scene:bistro"},
                {"label": "Quit", "action": "quit"},
            ]}),
        )];
        let edges = flow_edges(&world);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].from.as_deref(), Some("main"));
        assert_eq!(edges[0].label.as_deref(), Some("Start"));
        assert_eq!(edges[0].action, Move::Scene("bistro".to_string()));
        assert_eq!(edges[1].action, Move::Quit);
    }

    // The conveniences only a menu understands, resolved the way the expander
    // resolves them.
    #[test]
    fn a_menu_resolves_its_own_conveniences() {
        let world = [asset(
            RegisteredType::MainMenu,
            "main",
            json!({"items": [
                {"label": "Continue", "action": "return"},
                {"label": "Settings", "action": "settings"},
            ]}),
        )];
        let edges = flow_edges(&world);
        assert_eq!(edges[0].action, Move::Back);
        assert_eq!(
            edges[1].action,
            Move::Show("main_settings_video".to_string())
        );
    }

    // The point of the whole module: one world declares the shorthand, the
    // other declares what it expands into, and the map cannot tell them apart.
    #[test]
    fn a_menu_and_its_hand_authored_equivalent_declare_the_same_moves() {
        let shorthand = [asset(
            RegisteredType::MainMenu,
            "main",
            json!({"items": [{"label": "Start", "action": "scene:bistro"}]}),
        )];
        let by_hand = [
            asset(RegisteredType::Screen, "main", json!({"initial": true})),
            asset(
                RegisteredType::HitRegion,
                "main_btn_0",
                json!({"screen": "main", "action": "scene:bistro"}),
            ),
        ];
        let shorthand = flow_edges(&shorthand);
        let by_hand = flow_edges(&by_hand);
        assert_eq!(actions(&shorthand), actions(&by_hand));
        assert_eq!(shorthand[0].from, by_hand[0].from);
    }

    #[test]
    fn a_behavior_body_moves_from_nowhere() {
        let world = [asset(
            RegisteredType::Behavior,
            "opener",
            json!({"on": {"start": {}}, "do": [{"scene": {"scene": "bistro"}}]}),
        )];
        let edges = flow_edges(&world);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, None);
        assert_eq!(edges[0].action, Move::Scene("bistro".to_string()));
    }

    // A branch holds its own lists, so a move inside one is still a move.
    #[test]
    fn a_behavior_move_nested_in_a_branch_is_found() {
        let world = [asset(
            RegisteredType::Behavior,
            "gate",
            json!({"do": [{"if": {
                "cond": {"var": "won"},
                "do": [{"screen": {"screen": "victory"}}],
                "else": [{"after": {"seconds": 1, "do": [{"scene": {"scene": "retry"}}]}}],
            }}]}),
        )];
        assert_eq!(
            actions(&flow_edges(&world)),
            [
                &Move::Show("victory".to_string()),
                &Move::Scene("retry".to_string()),
            ]
        );
    }

    // A node that names no target is a move with nowhere to go: the build
    // rejects it, and drawing an arrow to an unnamed place would not help.
    #[test]
    fn a_behavior_node_naming_no_place_declares_no_move() {
        let world = [asset(
            RegisteredType::Behavior,
            "empty",
            json!({"do": [{"scene": {}}, {"screen": {"screen": ""}}]}),
        )];
        assert!(flow_edges(&world).is_empty());
    }

    // A variable named for a place is not a place: a node is read by its verb,
    // never by a name that happens to match one.
    #[test]
    fn a_variable_named_like_a_place_is_not_a_move() {
        let world = [asset(
            RegisteredType::Behavior,
            "counter",
            json!({"do": [{"set": {"var": "scene", "value": {"int": 1}}}]}),
        )];
        assert!(flow_edges(&world).is_empty());
    }

    #[test]
    fn a_story_node_moves_to_the_story_without_naming_it() {
        let world = [asset(
            RegisteredType::Behavior,
            "teller",
            json!({"do": [{"story": {"advance": {}}}]}),
        )];
        assert_eq!(flow_edges(&world)[0].action, Move::Story);
    }
}
