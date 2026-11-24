use super::*;
use concinnity_engine::components::{Identity, Sprite};

const SCREEN: AssetId = AssetId(77);

fn spec(min_size: Option<[f32; 2]>) -> PanelSpec<'static> {
    PanelSpec {
        key: PanelKey::View,
        title: "View",
        title_state: None,
        size: [200.0, 120.0],
        min_size,
    }
}

#[test]
fn a_panel_is_a_floating_root_holding_everything_built_for_it() {
    let mut world = World::new();
    let mut ui = UiBuilder::new(&mut world, SCREEN, None);
    let label = ui.label("hello", theme::LABEL).unwrap();
    let panel = ui.panel(&spec(None), vec![label]).unwrap();
    let root = world.get_by_id::<Layout>(panel.root).unwrap();
    assert!(root.movable && !root.resizable);
    assert_eq!(root.width, LayoutSize::Fixed(200.0));
    assert_eq!(root.screen.map(Ref::id), Some(SCREEN));
    assert!(panel.elements.contains(&panel.root));
    let ids: Vec<AssetId> = world.query::<Identity>().map(|i| i.id()).collect();
    assert_eq!(
        ids.len(),
        panel.elements.len(),
        "every pushed id is the panel's"
    );
    assert!(ids.iter().all(|id| id.is_minted()));
    assert!(
        world
            .query::<Sprite>()
            .all(|s| s.screen.map(Ref::id) == Some(SCREEN))
    );
}

#[test]
fn a_sized_panel_is_resizable_down_to_its_minimum() {
    let mut world = World::new();
    let mut ui = UiBuilder::new(&mut world, SCREEN, None);
    let panel = ui.panel(&spec(Some([150.0, 90.0])), Vec::new()).unwrap();
    let root = world.get_by_id::<Layout>(panel.root).unwrap();
    assert!(root.resizable);
    assert_eq!(root.min_size, [150.0, 90.0]);
}

#[test]
fn the_close_button_names_its_panel() {
    let mut world = World::new();
    let mut ui = UiBuilder::new(&mut world, SCREEN, None);
    ui.panel(&spec(None), Vec::new()).unwrap();
    let close = world.query::<Button>().next().unwrap();
    let Some(UiAction::Emit(emit)) = &close.action else {
        panic!("the close button emits");
    };
    assert_eq!(emit.name, CLOSE_EVENT);
    let index = i32::try_from(PanelKey::View.index()).unwrap();
    assert_eq!(emit.value, Some(UiValue::Int(index)));
}

#[test]
fn the_next_panel_starts_empty() {
    let mut world = World::new();
    let mut ui = UiBuilder::new(&mut world, SCREEN, None);
    let first = ui.panel(&spec(None), Vec::new()).unwrap();
    let second = ui.panel(&spec(None), Vec::new()).unwrap();
    assert!(
        first
            .elements
            .iter()
            .all(|id| !second.elements.contains(id))
    );
}
