use super::*;
use crate::editor::palette::{Category, PaletteAction};
use concinnity_engine::components::UiState;
use concinnity_engine::ecs::asset_id::AssetId;

fn built(world: &mut World) -> BuiltPanel {
    build(&mut UiBuilder::new(world, AssetId(9), None)).unwrap()
}

fn item(label: &str) -> PaletteItem {
    PaletteItem {
        label: label.into(),
        hint: "a hint".into(),
        category: Category::Panel,
        action: PaletteAction::OpenPanel(PanelKey::Console),
    }
}

fn view<'a>(items: &'a [PaletteItem], matches: &'a [usize], revision: u64) -> PaletteView<'a> {
    PaletteView {
        items,
        matches,
        query: "con",
        pick: 1,
        focus: true,
        revision,
        moved: 1,
    }
}

fn rows_of(world: &World) -> Vec<UiItem> {
    world.resource::<UiState>().unwrap().list(ROWS).to_vec()
}

#[test]
fn a_row_click_names_its_match() {
    let event = UiEvent {
        name: PICK.into(),
        element: None,
        value: Some(UiValue::Int(3)),
    };
    assert_eq!(pick(&event), Some(3));
}

// The field shows the query, the list the matches with their hints and tags,
// rebuilt only for a new revision of the matches.
#[test]
fn sync_writes_the_query_and_lists_the_matches_once() {
    let mut world = World::new();
    let panel = built(&mut world);
    let items = [item("Console"), item("Content")];
    sync(&view(&items, &[1, 0], 1), &mut world, &panel);
    let field = world
        .get_by_id::<TextField>(panel.slots[FIELD_SLOT])
        .unwrap();
    assert_eq!(field.content, "con");
    assert!(field.focused);
    let rows = rows_of(&world);
    assert_eq!(rows[0].text, "Content");
    assert_eq!(rows[0].detail, "a hint   panel");
    sync(&view(&items, &[0], 1), &mut world, &panel);
    assert_eq!(rows_of(&world).len(), 2, "the same revision keeps the rows");
    sync(&view(&items, &[], 2), &mut world, &panel);
    let rows = rows_of(&world);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].dimmed && rows[0].text == "no match");
}
