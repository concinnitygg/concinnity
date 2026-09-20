// The places a world can be in, read off its authored entries.

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use crate::build_only::main_menu::{generates_settings, settings_entry_screen};

/// What kind of place a world is in while it is there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceKind {
    /// A 3D scene.
    Scene,
    /// A UI screen, which sits over whatever it was opened from.
    Screen,
    /// A story, which plays through screens of its own.
    Story,
}

/// Somewhere the world can be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Place {
    /// The handle the world addresses it by: its `$id`, or the
    /// `<Type>#<ordinal>` label an anonymous entry loads with. Nothing
    /// references an anonymous place by name, but the world is still in it.
    pub id: String,
    /// What kind of place it is.
    pub kind: PlaceKind,
    /// The type declaring it, which is not always its kind: a `MainMenu`
    /// declares a screen, and a `StoryImport` a story.
    pub declared_by: RegisteredType,
    /// Whether the world is in it as soon as the world loads.
    pub entry: bool,
    /// Whether the build generates it rather than the world declaring it as
    /// its own line, as a menu generates the settings screen its items open.
    pub generated: bool,
}

/// Every place `assets` declares, in declaration order.
///
/// A `MainMenu` declares the screen it expands into under its own `$id`, so a
/// menu and the hand-authored `Screen` it stands for answer the same place.
pub fn places(assets: &[WorldJsonlAsset]) -> Vec<Place> {
    let mut places = Vec::new();
    for asset in assets {
        let kind = match asset.asset_type {
            RegisteredType::Scene => PlaceKind::Scene,
            RegisteredType::Screen | RegisteredType::MainMenu => PlaceKind::Screen,
            RegisteredType::Story | RegisteredType::StoryImport => PlaceKind::Story,
            _ => continue,
        };
        places.push(Place {
            id: asset.id.clone(),
            kind,
            declared_by: asset.asset_type,
            entry: flag(asset, "initial"),
            generated: false,
        });
        if let Some(settings) = settings_screen(asset) {
            places.push(settings);
        }
    }
    places
}

// The settings screen a menu generates for an item that opens one. Its tabs are
// navigation inside it rather than places of their own, so the place is the tab
// an item arrives at.
fn settings_screen(asset: &WorldJsonlAsset) -> Option<Place> {
    if asset.asset_type != RegisteredType::MainMenu {
        return None;
    }
    let items = asset.args.get("items").and_then(|v| v.as_array());
    let actions = items
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("action")?.as_str());
    let back = asset
        .args
        .get("settings_back_action")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    generates_settings(actions, back).then(|| Place {
        id: settings_entry_screen(&asset.id),
        kind: PlaceKind::Screen,
        declared_by: RegisteredType::MainMenu,
        entry: false,
        generated: true,
    })
}

fn flag(asset: &WorldJsonlAsset, key: &str) -> bool {
    asset.args.get(key).and_then(|v| v.as_bool()) == Some(true)
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

    #[test]
    fn scenes_screens_and_stories_are_places_and_nothing_else_is() {
        let world = [
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(RegisteredType::Screen, "pause", json!({})),
            asset(
                RegisteredType::StoryImport,
                "tale",
                json!({"source": "t.md"}),
            ),
            asset(RegisteredType::Prop, "crate", json!({})),
            asset(RegisteredType::Material, "mat", json!({})),
        ];
        let places = places(&world);
        assert_eq!(
            places.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
            ["bistro", "pause", "tale"]
        );
        assert_eq!(places[0].kind, PlaceKind::Scene);
        assert_eq!(places[1].kind, PlaceKind::Screen);
        assert_eq!(places[2].kind, PlaceKind::Story);
    }

    // The two spellings of one menu: the shorthand and the screen it expands
    // into. Both have to answer the same place, or a map drawn from this shows
    // an arrow for one world and nothing for the one beside it.
    #[test]
    fn a_menu_and_the_screen_it_expands_to_are_the_same_place() {
        let menu = places(&[asset(
            RegisteredType::MainMenu,
            "main",
            json!({"initial": true}),
        )]);
        let screen = places(&[asset(
            RegisteredType::Screen,
            "main",
            json!({"initial": true}),
        )]);
        assert_eq!(menu.len(), 1);
        assert_eq!(menu[0].id, screen[0].id);
        assert_eq!(menu[0].kind, screen[0].kind);
        assert_eq!(menu[0].entry, screen[0].entry);
    }

    #[test]
    fn a_screen_is_an_entry_only_when_it_declares_itself_one() {
        let world = [
            asset(RegisteredType::Screen, "menu", json!({"initial": true})),
            asset(RegisteredType::Screen, "pause", json!({"initial": false})),
            asset(RegisteredType::Screen, "hud", json!({})),
        ];
        let entries: Vec<bool> = places(&world).iter().map(|p| p.entry).collect();
        assert_eq!(entries, [true, false, false]);
    }

    // An anonymous entry is a place the world is in even though nothing can
    // name it: `["Scene",{}]` is the empty scene a menu world opens over.
    #[test]
    fn an_anonymous_scene_is_a_place() {
        let places = places(&[asset(RegisteredType::Scene, "Scene#0", json!({}))]);
        assert_eq!(places.len(), 1);
        assert_eq!(places[0].id, "Scene#0");
    }

    #[test]
    fn a_menu_with_a_settings_item_declares_the_screen_it_opens() {
        let world = [asset(
            RegisteredType::MainMenu,
            "main",
            json!({"items": [{"label": "Settings", "action": "settings"}]}),
        )];
        let places = places(&world);
        assert_eq!(places.len(), 2);
        assert_eq!(places[1].id, settings_entry_screen("main"));
        assert!(places[1].generated);
        assert!(!places[0].generated);
    }

    #[test]
    fn a_menu_without_one_declares_no_settings_screen() {
        let world = [asset(
            RegisteredType::MainMenu,
            "main",
            json!({"items": [{"label": "Quit", "action": "quit"}]}),
        )];
        assert_eq!(places(&world).len(), 1);
    }
}
