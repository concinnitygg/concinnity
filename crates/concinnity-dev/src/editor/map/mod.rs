//! A world seen from above: the places it can be in, and the moves between
//! them. Where the Behavior panel answers what one body does, this answers what
//! the world as a whole is shaped like, which no panel showing one asset at a
//! time can.
//!
//! The model is [`concinnity_cook::authoring::flow`], read off the authored
//! entries rather than the built world: a world is a few dozen authored lines
//! against thousands of cooked assets, and the authored lines are what an
//! editor edits.
//! So a card stands for a place and says what sits inside it, never one card
//! per asset -- which is also what keeps a map inside the fixed pool of ids the
//! chart view draws from.
//!
//! Dependencies are deliberately not drawn. Every prop in a world names the
//! same few materials, and the fan-in would bury the flow; the Assets panel
//! already answers what uses what.
//!
//! The result is an ordinary [`Chart`], so the chart view draws it unchanged.

mod build;
mod contents;
pub(crate) mod panel;

use concinnity_cook::authoring::flow::{Place, flow_graph};
use concinnity_cook::authoring::world::WorldJsonlAsset;

use super::asset_handle::AssetHandle;
use super::behavior::graph::Chart;
use super::entry_list::EntryId;

/// The authored world a map is drawn from: every entry, beside the session key
/// the editor addresses it by. The two are taken together so they cannot fall
/// out of step; turning the editor's entry shape into them is the hook's job,
/// as it is for the behavior overview.
pub(crate) struct Entries {
    assets: Vec<WorldJsonlAsset>,
    keys: Vec<EntryId>,
}

impl Entries {
    pub(crate) fn new(entries: impl IntoIterator<Item = (EntryId, WorldJsonlAsset)>) -> Self {
        let (keys, assets) = entries.into_iter().unzip();
        Self { assets, keys }
    }

    fn assets(&self) -> &[WorldJsonlAsset] {
        &self.assets
    }

    // How the editor addresses a place: the entry declaring it, or the identity
    // the build gives one it generates, as a menu generates its settings
    // screen. A card carries it so that selecting a place is a lookup rather
    // than a search.
    fn handle(&self, place: &Place) -> Option<AssetHandle> {
        if place.generated {
            return Some(AssetHandle::Generated(place.id.clone()));
        }
        let at = self.assets.iter().position(|a| a.id == place.id)?;
        Some(AssetHandle::Entry(self.keys[at]))
    }
}

/// The world's places and the moves between them, laid out left to right from
/// where it starts.
pub(crate) fn map(entries: &Entries) -> Chart {
    build::chart(entries, &flow_graph(entries.assets()))
}

/// The card standing for `handle`, so what is selected anywhere in the editor
/// lights up here too. A selection naming no place at all -- a prop inside one,
/// a material -- answers `None`, which is the honest picture: a map draws where
/// a world can be, not everything in it.
pub(crate) fn card_of(chart: &Chart, handle: &AssetHandle) -> Option<usize> {
    chart
        .cards
        .iter()
        .position(|card| card.handle.as_ref() == Some(handle))
}

#[cfg(test)]
mod tests {
    use concinnity_cook::authoring::registry::RegisteredType;
    use serde_json::{Value, json};

    use super::*;
    use crate::editor::behavior::graph::Card;
    use crate::editor::entry_list::EntryList;

    fn asset(ty: RegisteredType, id: &str, args: Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: id.to_string(),
            asset_type: ty,
            args,
        }
    }

    // A world as the editor holds it: every entry beside the key a session
    // minted for it.
    fn entries(assets: Vec<WorldJsonlAsset>) -> (Entries, Vec<EntryId>) {
        let mut list = EntryList::default();
        let keyed: Vec<(EntryId, WorldJsonlAsset)> = assets
            .into_iter()
            .map(|asset| (list.push(Value::Null), asset))
            .collect();
        let keys = keyed.iter().map(|(key, _)| *key).collect();
        (Entries::new(keyed), keys)
    }

    fn card<'a>(chart: &'a Chart, title: &str) -> &'a Card {
        chart.cards.iter().find(|c| c.title == title).unwrap()
    }

    fn index(chart: &Chart, title: &str) -> usize {
        chart.cards.iter().position(|c| c.title == title).unwrap()
    }

    // The exterior-import world: one scene, a file to fill it with, and the
    // settings every world declares. Its whole map is one card, so that card
    // has to say what the scene holds rather than that a scene exists.
    fn imported_scene() -> Vec<WorldJsonlAsset> {
        vec![
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(
                RegisteredType::GraphicsConfig,
                "GraphicsConfig#0",
                json!({}),
            ),
            asset(RegisteredType::Camera3D, "Camera3D#0", json!({})),
            asset(
                RegisteredType::SceneImport,
                "SceneImport#0",
                json!({"source": "BistroExterior.fbx", "scene": "bistro"}),
            ),
        ]
    }

    // The same world with the menu it opens over: the picture the map is for.
    fn menu_over_a_scene() -> Vec<WorldJsonlAsset> {
        let mut world = imported_scene();
        world.insert(0, asset(RegisteredType::Scene, "Scene#0", json!({})));
        world.push(asset(
            RegisteredType::MainMenu,
            "MainMenu#0",
            json!({"initial": true, "items": [
                {"label": "Start", "action": "scene:bistro"},
                {"label": "Settings", "action": "settings"},
                {"label": "Quit", "action": "quit"},
            ]}),
        ));
        world
    }

    #[test]
    fn a_menu_world_maps_to_the_menu_the_arrow_and_the_scene_it_opens() {
        let chart = map(&entries(menu_over_a_scene()).0);
        assert_eq!(
            card(&chart, "MainMenu#0").column,
            0,
            "it starts in its menu"
        );
        assert_eq!(card(&chart, "bistro").column, 1);
        assert_eq!(card(&chart, "bistro").detail, "scene, 1 SceneImport");
        let start = chart
            .wires
            .iter()
            .find(|w| w.label.as_deref() == Some("Start"))
            .expect("the button the world starts through");
        let (menu, scene) = (index(&chart, "MainMenu#0"), index(&chart, "bistro"));
        assert_eq!((start.from, start.to), (menu, scene));
    }

    // A world of one scene is not an empty map; it is one card that has to be
    // worth looking at.
    #[test]
    fn a_world_of_one_scene_maps_to_one_card_saying_what_is_in_it() {
        let chart = map(&entries(imported_scene()).0);
        assert_eq!(chart.cards.len(), 1);
        assert!(chart.wires.is_empty());
        assert_eq!(chart.cards[0].title, "bistro");
        assert_eq!(chart.cards[0].detail, "scene, 1 SceneImport");
        assert_eq!((chart.columns, chart.rows), (1, 1));
    }

    // A world declaring nowhere to be still declares what it is made of, so its
    // map degrades to a summary of the world rather than to a blank canvas.
    #[test]
    fn a_world_with_nowhere_to_be_maps_to_what_it_declares() {
        let showcase = vec![
            asset(
                RegisteredType::GraphicsConfig,
                "GraphicsConfig#0",
                json!({}),
            ),
            asset(RegisteredType::Material, "mat_floor", json!({})),
            asset(RegisteredType::Material, "mat_red", json!({})),
            asset(RegisteredType::Prop, "floor", json!({"mesh": "floor_mesh"})),
            asset(RegisteredType::Prop, "Prop#1", json!({"mesh": "wall_mesh"})),
            asset(RegisteredType::Prop, "Prop#2", json!({"mesh": "wall_mesh"})),
        ];
        let chart = map(&entries(showcase).0);
        assert_eq!(chart.cards.len(), 1);
        assert_eq!(chart.cards[0].title, "world");
        assert_eq!(chart.cards[0].detail, "3 Prop, 2 Material");
    }

    // A card addresses its place the way the rest of the editor does, so a
    // click on one is a selection rather than a search.
    #[test]
    fn a_card_carries_the_handle_the_editor_addresses_its_place_by() {
        let world = menu_over_a_scene();
        let menu_at = world.len() - 1;
        let (entries, keys) = entries(world);
        let chart = map(&entries);
        assert_eq!(
            card(&chart, "MainMenu#0").handle,
            Some(AssetHandle::Entry(keys[menu_at])),
        );
        assert_eq!(
            card(&chart, "bistro").handle,
            Some(AssetHandle::Entry(keys[1])),
        );
    }

    // The screen a menu opens has no authored line of its own, so it is
    // addressed by the identity the build gives it instead.
    #[test]
    fn a_place_the_build_generates_is_addressed_as_one() {
        let chart = map(&entries(menu_over_a_scene()).0);
        let generated: Vec<&str> = chart
            .cards
            .iter()
            .filter(|c| matches!(c.handle, Some(AssetHandle::Generated(_))))
            .map(|c| c.title.as_str())
            .collect();
        assert_eq!(generated.len(), 1, "the settings screen the menu opens");
        assert_eq!(
            card(&chart, generated[0]).handle,
            Some(AssetHandle::Generated(generated[0].to_string())),
        );
    }
}
