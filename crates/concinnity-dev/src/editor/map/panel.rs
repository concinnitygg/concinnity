//! The Map panel: the world's places and the moves between them, drawn as a
//! chart. There is one world, so the panel has nothing to step through; its
//! body is the canvas and its chrome is the title bar, which leaves the whole
//! panel to the picture.
//!
//! [`map`](super::map) builds the chart and `behavior/chart.rs` writes it as
//! the canvas's shapes; the hook (`hook/edit/map.rs`) owns the pan and what a
//! press on a card does.

use concinnity_engine::components::Canvas;
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use crate::editor::behavior::chart;
use crate::editor::behavior::graph::{Card, Chart};
use crate::editor::panels::registry::PanelKey;
use crate::editor::theme;
use crate::editor::ui::build::{BODY_PAD, PanelSpec, panel_height};
use crate::editor::ui::{BuiltPanel, UiBuilder};

// The shape list the canvas draws, and the event a press on a card sends with
// the card's index.
pub(crate) const SHAPES: &str = "editor.map";
pub(crate) const CARD_EVENT: &str = "editor.map.card";

// What the canvas opens showing: a place, the arrow out of it, and the place it
// leads to, over enough rows for the ones sharing a column. A panel cannot size
// itself against the viewport, so this has to fit a small window outright.
const COLUMNS: usize = 3;
const ROWS: usize = 4;

pub(crate) fn size() -> [f32; 2] {
    [
        chart::width_for(COLUMNS) + 2.0 * BODY_PAD,
        panel_height(chart::height_for(ROWS)),
    ]
}

// Where the panel sits until the user drags it: centered, below the top bar and
// clear of the Behavior panel's own resting place.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [
        ((vw - size()[0]) * 0.5).max(8.0),
        crate::editor::hud::body_top() + 48.0,
    ]
}

// The canvas's size at panel size `s`, which is all the pan maths needs.
pub(crate) fn canvas(s: [f32; 2]) -> [f32; 2] {
    [
        (s[0] - 2.0 * BODY_PAD).max(0.0),
        (s[1] - theme::TITLE_H - 2.0 * BODY_PAD).max(0.0),
    ]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let canvas = ui.canvas(Canvas {
        shapes: SHAPES.to_string(),
        on_press: CARD_EVENT.to_string(),
        pannable: true,
        ..Default::default()
    })?;
    let canvas_id = canvas.item?.id();
    let spec = PanelSpec {
        key: PanelKey::Map,
        title: "Map",
        title_state: None,
        size: size(),
        min_size: Some(size()),
    };
    let mut panel = ui.panel(&spec, vec![canvas])?;
    panel.slots = vec![canvas_id];
    Some(panel)
}

// The canvas a built panel draws on.
pub(crate) fn canvas_of(built: &BuiltPanel) -> AssetId {
    built.slots[0]
}

// The canvas's pan and the card under the cursor, as the user left them.
pub(crate) fn observed(world: &World, built: &BuiltPanel) -> Option<([f32; 2], Option<usize>)> {
    let c = world.get_by_id::<Canvas>(canvas_of(built))?;
    Some((c.pan, c.hovered.and_then(|t| usize::try_from(t).ok())))
}

// Put the canvas at `pan`.
pub(crate) fn sync(world: &mut World, built: &BuiltPanel, pan: [f32; 2]) {
    if let Some(c) = world.get_mut_by_id::<Canvas>(canvas_of(built)) {
        c.pan = pan;
    }
}

// The place the world starts in: the one `build` roots leftmost, which is the
// topmost card of the first column once the rest have been relaxed rightward.
fn root_card(chart: &Chart) -> Option<&Card> {
    chart.cards.iter().min_by_key(|c| (c.column, c.row))
}

// The pan that opens the canvas on where the world starts, so the panel says
// where the world begins rather than wherever it was last left.
pub(crate) fn root_pan(chart: &Chart, canvas: [f32; 2]) -> [f32; 2] {
    match root_card(chart) {
        Some(card) => chart::pan_to(card, canvas, [0.0, 0.0], chart),
        None => [0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use concinnity_cook::authoring::registry::RegisteredType;
    use concinnity_cook::authoring::world::WorldJsonlAsset;
    use concinnity_engine::components::UiShape;
    use serde_json::{Value, json};

    use super::*;
    use crate::editor::behavior::chart::ChartMarks;
    use crate::editor::entry_list::EntryList;
    use crate::editor::map::{self, Entries};
    use crate::editor::ui::shapes::Shapes;

    fn asset(ty: RegisteredType, id: &str, args: Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: id.to_string(),
            asset_type: ty,
            args,
        }
    }

    // A world as the editor holds it, mapped: every entry beside the key a
    // session minted for it.
    fn mapped(assets: Vec<WorldJsonlAsset>) -> Chart {
        let mut list = EntryList::default();
        map::map(&Entries::new(
            assets.into_iter().map(|a| (list.push(Value::Null), a)),
        ))
    }

    // The canvas's shapes for `chart` with `selected` lit.
    fn drawn(chart: &Chart, selected: Option<usize>) -> Vec<UiShape> {
        let mut list = Vec::new();
        let mut shapes = Shapes::new(&mut list);
        let marks = ChartMarks {
            chart,
            selected,
            faulted: None,
            hovered: None,
            pulses: &[],
            breakpoints: &[],
        };
        chart::push(&mut shapes, &marks, &mut Vec::new());
        shapes.finish();
        list
    }

    // The card body the card called `title` was drawn as: the tagged shape
    // its title line sits in.
    fn card_of(shapes: &[UiShape], chart: &Chart, title: &str) -> UiShape {
        let index = index(chart, title);
        shapes
            .iter()
            .find(|s| s.tag == u32::try_from(index).ok())
            .cloned()
            .unwrap_or_else(|| panic!("no `{title}` card drawn"))
    }

    fn index(chart: &Chart, title: &str) -> usize {
        chart.cards.iter().position(|c| c.title == title).unwrap()
    }

    fn imported_scene() -> Vec<WorldJsonlAsset> {
        vec![
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(
                RegisteredType::SceneImport,
                "SceneImport#0",
                json!({"source": "BistroExterior.fbx", "scene": "bistro"}),
            ),
        ]
    }

    fn menu_world() -> Vec<WorldJsonlAsset> {
        vec![
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(
                RegisteredType::MainMenu,
                "main",
                json!({"initial": true, "items": [{"label": "Start", "action": {"scene": "bistro"}}]}),
            ),
        ]
    }

    // A one-place world is the panel's commonest picture, so its card has to be
    // worth opening the panel for: the place, and what is inside it.
    #[test]
    fn a_world_of_one_place_draws_a_card_saying_what_is_in_it() {
        let chart = mapped(imported_scene());
        let shapes = drawn(&chart, None);
        assert!(shapes.iter().any(|s| s.text == "bistro"));
        assert!(
            shapes.iter().any(|s| s.text.starts_with("scene, 1 Scene")),
            "{shapes:?}"
        );
    }

    // The picture the map is for: the menu the world starts in, and the scene
    // its button opens, to the right of it, with the wire between them.
    #[test]
    fn a_menu_draws_left_of_the_scene_it_opens() {
        let chart = mapped(menu_world());
        let shapes = drawn(&chart, None);
        let menu = card_of(&shapes, &chart, "main");
        let scene = card_of(&shapes, &chart, "bistro");
        assert!(
            menu.rect[0] < scene.rect[0],
            "{menu:?} is not left of {scene:?}"
        );
        assert_eq!(menu.rect[1], scene.rect[1], "one move, one row");
        let wire = shapes.iter().find(|s| {
            s.tag.is_none() && s.fill[3] > 0.0 && s.rect[0] >= menu.rect[0] + menu.rect[2] - 0.5
        });
        assert!(wire.is_some(), "no wire between the two");
    }

    // Every card is a pressable shape carrying its index, a card addressing
    // nothing included: pressing it must not drag the canvas out from under
    // the cursor.
    #[test]
    fn every_card_is_tagged_with_its_index() {
        let chart = mapped(vec![
            asset(RegisteredType::Material, "mat_floor", json!({})),
            asset(RegisteredType::Prop, "floor", json!({"mesh": "floor_mesh"})),
        ]);
        assert_eq!(chart.cards[0].handle, None, "{:?}", chart.cards[0]);
        let tags: Vec<u32> = drawn(&chart, None).iter().filter_map(|s| s.tag).collect();
        assert_eq!(tags, [0]);
    }

    // The chart's frame reaches its margin, so the canvas pans to the last
    // place and no further.
    #[test]
    fn the_frame_spans_the_whole_chart() {
        let chart = mapped(menu_world());
        let frame = drawn(&chart, None)[0].clone();
        assert_eq!(&frame.rect[2..], &chart::extent(&chart)[..]);
        assert_eq!(frame.fill[3], 0.0, "and draws nothing");
    }

    // What is selected anywhere in the editor is what the map lights up, so the
    // two surfaces agree on where the session is looking.
    #[test]
    fn the_selected_place_is_drawn_lit_up() {
        let chart = mapped(menu_world());
        let shapes = drawn(&chart, Some(index(&chart, "bistro")));
        let lit = card_of(&shapes, &chart, "bistro");
        let plain = card_of(&shapes, &chart, "main");
        assert!(
            lit.border_width > plain.border_width,
            "{lit:?} is drawn no differently to {plain:?}",
        );
    }

    // The panel is a canvas under a title bar, pannable, sending its card
    // presses.
    #[test]
    fn the_panel_is_one_pannable_canvas() {
        let mut world = World::new();
        let built = build(&mut UiBuilder::new(&mut world, AssetId(9), None)).unwrap();
        let canvas = world.get_by_id::<Canvas>(canvas_of(&built)).unwrap();
        assert!(canvas.pannable);
        assert_eq!(canvas.shapes, SHAPES);
        assert_eq!(canvas.on_press, CARD_EVENT);
        sync(&mut world, &built, [12.0, 3.0]);
        assert_eq!(observed(&world, &built), Some(([12.0, 3.0], None)));
    }

    // The canvas opens on where the world starts, which is the corner the map
    // is laid out from.
    #[test]
    fn the_root_pan_opens_on_the_place_the_world_starts_in() {
        let chart = mapped(imported_scene());
        assert_eq!(root_pan(&chart, canvas(size())), [0.0, 0.0]);
    }

    // And it is the card that decides, not the corner: a map whose first place
    // is laid out away from the origin is panned to rather than missed.
    #[test]
    fn the_root_pan_follows_the_card_rather_than_the_canvas_corner() {
        let mut chart = mapped(imported_scene());
        chart.cards[0].column = 4;
        chart.cards[0].row = 6;
        chart.columns = 5;
        chart.rows = 7;
        let pan = root_pan(&chart, canvas(size()));
        assert!(
            pan[0] > 0.0 && pan[1] > 0.0,
            "{pan:?} left it off the canvas"
        );
    }
}
