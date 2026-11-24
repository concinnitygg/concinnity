//! The viewport's right-click "Create here" menu: an engine `Menu` opened at
//! the cursor that adds an asset at the world position under the click. The
//! offered types come from the registry (every blank-useful type that carries a
//! world position -- the billboard-eligible set), each with its one-line
//! summary, plus a disclosure listing the world's Prefabs (each creates a Prop
//! instancing that prefab). The open / pick / commit flow lives in
//! `hook/drive/create_menu.rs`.

use concinnity_engine::components::{Binding, Menu, UiEvent, UiItem, UiValue};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::panels::assets_panel;
use super::ui::{UiBuilder, state};
use super::viewport::billboards;

// Rows past this are cut (a world with that many Prefabs still creates them
// through the Content panel or the console).
pub(crate) const MAX_ROWS: usize = 24;

// Wide enough for the name plus a useful run of each type's summary, the same
// two columns the assets panel's picker shows.
const MENU_W: f32 = 420.0;
const ROW_H: f32 = 24.0;

// The state the menu shows, and the event its picks send.
const OPEN: &str = "editor.create.open";
const ITEMS: &str = "editor.create.items";
const PICK: &str = "editor.create.pick";

// The heading the menu lists above its rows.
const HEADING: &str = "Create here";

// One menu row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MenuItem {
    // A placeable registered type: picking it creates one at the click point.
    Type(&'static str),
    // The Prefab disclosure row; picking it unfolds / folds the Prefab list.
    PrefabHeader { open: bool },
    // A world Prefab: picking it creates a Prop instancing it.
    Prefab(String),
}

impl MenuItem {
    pub(crate) fn caption(&self) -> String {
        match self {
            Self::Type(ty) => (*ty).to_string(),
            Self::PrefabHeader { open: false } => "Prefab instance >".to_string(),
            Self::PrefabHeader { open: true } => "Prefab instance v".to_string(),
            Self::Prefab(name) => name.clone(),
        }
    }

    // The type's one-line summary. A Prefab row names a world asset rather than
    // a type, so it has none and leaves the column blank.
    fn summary(&self) -> &'static str {
        match self {
            Self::Type(ty) => crate::docs::type_summaries()
                .get(*ty)
                .map_or_default(String::as_str),
            Self::PrefabHeader { .. } | Self::Prefab(_) => "",
        }
    }

    fn indented(&self) -> bool {
        matches!(self, Self::Prefab(_))
    }

    // The item the menu lists for this row.
    fn item(&self) -> UiItem {
        UiItem {
            detail: self.summary().to_string(),
            depth: u16::from(self.indented()),
            expandable: matches!(self, Self::PrefabHeader { .. }),
            ..UiItem::new(&self.caption())
        }
    }
}

// The types the create menu places: blank-addable types with a world
// position, alphabetized.
pub(crate) fn placeable_types() -> Vec<&'static str> {
    let mut types: Vec<&'static str> = assets_panel::add_types()
        .filter(|ty| billboards::eligible(ty))
        .collect();
    types.sort_unstable();
    types
}

// The authored arg key holding the type's world position ("position", or
// "center" for the panel lights).
pub(crate) fn position_key(ty: &str) -> &'static str {
    if super::panels::form::working_args(ty, None).contains_key("center") {
        "center"
    } else {
        "position"
    }
}

// The menu rows for this frame: the placeable types, then the Prefab section
// when the world has any. Clipped to the row pool.
pub(crate) fn items(prefabs: &[String], prefabs_open: bool) -> Vec<MenuItem> {
    let mut out: Vec<MenuItem> = placeable_types().into_iter().map(MenuItem::Type).collect();
    if !prefabs.is_empty() {
        out.push(MenuItem::PrefabHeader { open: prefabs_open });
        if prefabs_open {
            out.extend(prefabs.iter().cloned().map(MenuItem::Prefab));
        }
    }
    out.truncate(MAX_ROWS);
    out
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<AssetId> {
    ui.menu(Menu {
        open: Some(Binding::State(OPEN.to_string())),
        items: ITEMS.to_string(),
        on_pick: PICK.to_string(),
        at_cursor: true,
        width: MENU_W,
        row_height: ROW_H,
        ..Default::default()
    })
}

// The row a pick names, if `event` is one: the menu lists its heading first.
pub(crate) fn pick(event: &UiEvent) -> Option<usize> {
    match (event.name.as_str(), &event.value) {
        (PICK, Some(UiValue::Int(i))) => usize::try_from(*i).ok()?.checked_sub(1),
        _ => None,
    }
}

// Show the menu open, or closed.
pub(crate) fn show(world: &mut World, open: bool) {
    state::set_bool(world, OPEN, open);
}

// List `items` under the menu's heading.
pub(crate) fn list(world: &mut World, items: &[MenuItem]) {
    let listed = std::iter::once(UiItem {
        dimmed: true,
        ..UiItem::new(HEADING)
    })
    .chain(items.iter().map(MenuItem::item))
    .collect();
    state::with_state(world, |s| s.set_list(ITEMS, listed));
}

// Whether the menu is open in `world`.
pub(crate) fn is_open(world: &World) -> bool {
    state::get_bool(world, OPEN)
}

// Whether `(x, y)` lands on the open menu `id` listing `rows` rows.
pub(crate) fn covers(world: &World, id: AssetId, x: f32, y: f32, rows: usize) -> bool {
    world
        .get_by_id::<Menu>(id)
        .is_some_and(|m| m.covers(x, y, rows + 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::UiState;

    // The offered set is registry-derived: position-carrying blank-useful
    // types are in; renderable, positionless, and singleton types are not.
    #[test]
    fn placeable_types_are_the_positioned_blank_useful_set() {
        let types = placeable_types();
        for expected in [
            "PointLight",
            "SpotLight",
            "TriggerVolume",
            "ReflectionProbe",
        ] {
            assert!(types.contains(&expected), "{expected} missing: {types:?}");
        }
        for excluded in ["Prop", "Sprite", "GraphicsConfig", "Window", "Material"] {
            assert!(!types.contains(&excluded), "{excluded} offered: {types:?}");
        }
        let mut sorted = types.clone();
        sorted.sort_unstable();
        assert_eq!(types, sorted, "alphabetized");
    }

    // Every offered type resolves a position key that its default args carry,
    // so the created entry's override always lands on a real field.
    #[test]
    fn every_placeable_type_has_its_position_key() {
        for ty in placeable_types() {
            let key = position_key(ty);
            let args = crate::editor::panels::form::working_args(ty, None);
            assert!(args.contains_key(key), "{ty} lacks {key}");
        }
    }

    #[test]
    fn items_list_types_then_a_prefab_section() {
        let none = items(&[], false);
        assert!(none.iter().all(|i| matches!(i, MenuItem::Type(_))));

        let prefabs = vec!["door".to_string(), "crate".to_string()];
        let closed = items(&prefabs, false);
        assert_eq!(
            closed.last(),
            Some(&MenuItem::PrefabHeader { open: false }),
            "closed section is one header row"
        );
        assert!(!closed.iter().any(|i| matches!(i, MenuItem::Prefab(_))));

        let open = items(&prefabs, true);
        assert_eq!(
            open.iter()
                .filter(|i| matches!(i, MenuItem::Prefab(_)))
                .count(),
            2,
            "open section lists the world's prefabs"
        );
    }

    // The menu lists its heading, then each row with its summary; the
    // Prefab header is a disclosure and the prefabs sit under it.
    #[test]
    fn the_list_holds_the_heading_then_the_rows() {
        let mut world = World::new();
        let rows = vec![
            MenuItem::Type("PointLight"),
            MenuItem::PrefabHeader { open: true },
            MenuItem::Prefab("door".to_string()),
        ];
        show(&mut world, true);
        list(&mut world, &rows);
        let ui = world.resource::<UiState>().unwrap();
        let items = ui.list(ITEMS);
        assert_eq!(items[0].text, HEADING);
        assert!(items[0].dimmed);
        assert_eq!(items[1].text, "PointLight");
        assert!(!items[1].detail.is_empty(), "a type documents itself");
        assert!(items[2].expandable);
        assert_eq!((items[3].depth, items[3].detail.as_str()), (1, ""));
        assert!(is_open(&world));
        let event = UiEvent {
            name: PICK.into(),
            element: None,
            value: Some(UiValue::Int(1)),
        };
        assert_eq!(pick(&event), Some(0), "past the heading");
    }
}
