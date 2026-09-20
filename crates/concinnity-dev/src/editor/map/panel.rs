//! The Map panel: the world's places and the moves between them, drawn as a
//! chart. There is one world, so the panel has nothing to step through and
//! nothing to pick; its body is the canvas and its chrome is the title bar,
//! which leaves the whole panel to the picture.
//!
//! [`map`](super::map) builds the chart and `behavior/chart.rs` draws it, into
//! this panel's own id family: the Behavior panel draws charts too, and the two
//! are open at once often enough that sharing one family would leave each
//! blanking the other's cards.

use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use crate::editor::behavior::chart::{self, ChartIds};
use crate::editor::behavior::graph::{Card, Chart};
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::widget::{self, point_in};

const BASE: u32 = registry::base(PanelKey::Map);
const PANEL_BG: AssetId = AssetId(BASE);
const TITLE_LABEL: AssetId = AssetId(BASE + 1);
const CLOSE_BG: AssetId = AssetId(BASE + 2);
const CLOSE_LABEL: AssetId = AssetId(BASE + 3);
const CHART_IDS: ChartIds = ChartIds::of(PanelKey::Map);

// What the canvas opens showing: a place, the arrow out of it, and the place it
// leads to, over enough rows for the ones sharing a column. A panel cannot size
// itself against the viewport, so this has to fit a small window outright.
const COLUMNS: usize = 3;
const ROWS: usize = 4;

// The chrome above the canvas: the title bar, and nothing else.
const CHROME_H: f32 = widget::TITLE_H;

// What to say about the places a bandful of cards leaves out. Which cards hold
// the pool's slots follows the canvas, so panning reaches the rest.
const OVERFLOW: &str = "places in view -- pan to reach the rest";

pub(crate) struct MapView<'a> {
    pub chart: &'a Chart,
    // The card standing for what is selected, wherever the selection was made.
    pub selected: Option<usize>,
    pub pan: [f32; 2],
    pub mouse: [f32; 2],
}

// A resolved Map-panel press.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MapAction {
    // A press on the card at this index: select the place it stands for.
    Select(usize),
    // A press on the canvas: grab it, and the map follows the cursor.
    PanStart,
    // A press on the panel but off the canvas, swallowed so it cannot reach
    // the world behind it.
    Consume,
}

pub(crate) fn size() -> [f32; 2] {
    [
        chart::width_for(COLUMNS),
        CHROME_H + chart::height_for(ROWS),
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

fn body_top(o: [f32; 2]) -> f32 {
    o[1] + CHROME_H
}

// The canvas: the panel's body band, inset off the border.
pub(crate) fn band(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    chart::band(o, s, body_top(o))
}

// The canvas's size at panel size `s`, which is all the pan maths needs.
pub(crate) fn canvas(s: [f32; 2]) -> [f32; 2] {
    let b = band([0.0, 0.0], s);
    [b[2], b[3]]
}

pub(crate) fn cursor_over_body(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    let p = widget::outer_rect(o, s);
    mx >= p[0] && mx < p[0] + p[2] && my >= body_top(o) && my < p[1] + p[3]
}

// Resolve a press at `(mx, my)` against the panel at origin `o`, size `s`.
// `None` means the press missed the panel, so it falls through to whatever is
// behind. Title-bar presses never reach here: the shared routing takes them.
pub(crate) fn hit_test(
    view: &MapView,
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
) -> Option<MapAction> {
    let band = band(o, s);
    if point_in(mx, my, band) {
        // A card addressing nothing -- the world outside its places, or a name
        // a move leads to and no entry answers -- has nothing to select, and is
        // still a card rather than canvas to grab.
        return Some(match chart::hit_card(&chart_view(view), mx, my, band) {
            Some(i) if view.chart.cards[i].handle.is_some() => MapAction::Select(i),
            Some(_) => MapAction::Consume,
            None => MapAction::PanStart,
        });
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(MapAction::Consume)
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

// Position + show the panel (`Some(view)`) at effective size `s`, or blank
// every element (`None`).
pub(crate) fn place(world: &mut World, view: Option<&MapView>, o: [f32; 2], s: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, s[0]);
    widget::place_heading(world, TITLE_LABEL, title, "Map");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);
    chart::place(world, &chart_view(view), band(o, s));
}

fn chart_view<'a>(view: &'a MapView<'a>) -> chart::ChartView<'a> {
    chart::ChartView {
        ids: CHART_IDS,
        chart: view.chart,
        selected: view.selected,
        // A card stands for a place rather than for a node a session can stop
        // in or a row a checker can complain about, so the marks the Behavior
        // panel puts on its cards for those have nothing to point at here.
        faulted: None,
        pulses: &[],
        breakpoints: &[],
        pan: view.pan,
        mouse: view.mouse,
        overflow: OVERFLOW,
    }
}

pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[]);
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG];
    ids.extend(CHART_IDS.all_sprite_ids());
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, CLOSE_LABEL];
    ids.extend(CHART_IDS.all_label_ids());
    ids
}

#[cfg(test)]
mod tests {
    use concinnity_cook::authoring::registry::RegisteredType;
    use concinnity_cook::authoring::world::WorldJsonlAsset;
    use concinnity_core::components::{Sprite, TextLabel};
    use serde_json::{Value, json};

    use super::*;
    use crate::editor::behavior::chart::CARD_POOL;
    use crate::editor::entry_list::EntryList;
    use crate::editor::map::{self, Entries};

    const O: [f32; 2] = [40.0, 60.0];

    fn injected_world() -> World {
        crate::test_support::injected_world(&all_sprite_ids(), &all_label_ids(), &[])
    }

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

    fn view<'a>(chart: &'a Chart, pan: [f32; 2]) -> MapView<'a> {
        MapView {
            chart,
            selected: None,
            pan,
            mouse: [-1.0, -1.0],
        }
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world.get_by_id::<TextLabel>(id).cloned().unwrap()
    }

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world.get_by_id::<Sprite>(id).cloned().unwrap()
    }

    // The slot the card called `title` was drawn in, if it was drawn at all.
    fn slot_of(world: &World, title: &str) -> Option<usize> {
        (0..CARD_POOL).find(|&i| {
            let l = label(world, CHART_IDS.card_title(i));
            l.visible && l.content == title
        })
    }

    fn drawn(world: &World, title: &str) -> Sprite {
        let slot = slot_of(world, title).unwrap_or_else(|| panic!("no `{title}` card drawn"));
        sprite(world, CHART_IDS.card_bg(slot))
    }

    // A world of one scene filled from a file: the shape most worlds are in
    // today, and the one where the map is a single card.
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

    fn scenes(count: usize) -> Vec<WorldJsonlAsset> {
        (0..count)
            .map(|i| asset(RegisteredType::Scene, &format!("scene_{i}"), json!({})))
            .collect()
    }

    // A one-place world is the panel's commonest picture, so its card has to be
    // worth opening the panel for: the place, and what is inside it.
    #[test]
    fn a_world_of_one_place_draws_a_card_saying_what_is_in_it() {
        let chart = mapped(imported_scene());
        let mut world = injected_world();
        place(&mut world, Some(&view(&chart, [0.0, 0.0])), O, size());

        assert!(sprite(&world, PANEL_BG).visible);
        assert_eq!(label(&world, TITLE_LABEL).content, "Map");
        assert_eq!(slot_of(&world, "bistro"), Some(0));
        let detail = label(&world, CHART_IDS.card_detail(0));
        assert!(detail.visible);
        assert!(detail.content.starts_with("scene, 1 Scene"), "{detail:?}");
    }

    // The picture the map is for: the menu the world starts in, and the scene
    // its button opens, to the right of it.
    #[test]
    fn a_menu_draws_left_of_the_scene_it_opens() {
        let chart = mapped(vec![
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(
                RegisteredType::MainMenu,
                "main",
                json!({"initial": true, "items": [{"label": "Start", "action": "scene:bistro"}]}),
            ),
        ]);
        let mut world = injected_world();
        place(&mut world, Some(&view(&chart, [0.0, 0.0])), O, size());

        let menu = drawn(&world, "main");
        let scene = drawn(&world, "bistro");
        assert!(menu.x < scene.x, "{menu:?} is not left of {scene:?}");
        assert_eq!(menu.y, scene.y, "one move, one row");
        // And the arrow between them is drawn.
        assert!(
            (0..CARD_POOL).any(|i| sprite(&world, CHART_IDS.segment(i)).visible),
            "no wire between the two",
        );
    }

    // The canvas holds what it holds; the places past it are panned to rather
    // than lost, which is what makes a map of a world bigger than the panel
    // worth drawing at all.
    #[test]
    fn a_place_off_the_canvas_is_drawn_once_it_is_panned_to() {
        let chart = mapped(scenes(12));
        let far = chart.cards.last().unwrap();
        let mut world = injected_world();
        place(&mut world, Some(&view(&chart, [0.0, 0.0])), O, size());
        assert_eq!(slot_of(&world, &far.title), None, "it starts off the band");

        let pan = chart::pan_to(far, canvas(size()), [0.0, 0.0], &chart);
        place(&mut world, Some(&view(&chart, pan)), O, size());
        assert!(slot_of(&world, &far.title).is_some(), "panning reached it");
    }

    // Slots are finite, so a canvas big enough to hold more places than the
    // pool draws what it can and says how many it left out.
    #[test]
    fn a_canvas_of_more_places_than_the_pool_says_how_many_are_missing() {
        let chart = mapped(scenes(CARD_POOL + 8));
        let mut world = injected_world();
        let tall = [size()[0], CHROME_H + chart::height_for(CARD_POOL + 8)];
        place(&mut world, Some(&view(&chart, [0.0, 0.0])), O, tall);

        let hint = label(&world, CHART_IDS.hint());
        assert!(hint.visible);
        assert!(hint.content.starts_with("8 more places"), "{hint:?}");
        let cards = (0..CARD_POOL)
            .filter(|&i| sprite(&world, CHART_IDS.card_bg(i)).visible)
            .count();
        assert_eq!(cards, CARD_POOL, "every slot went to a place on the canvas");
    }

    // The canvas is grabbed and dragged; the chrome around it swallows a press
    // rather than letting it through to the world, and a press clear of the
    // panel is not the panel's at all.
    #[test]
    fn the_canvas_pans_and_the_rest_of_the_panel_swallows_a_press() {
        let chart = mapped(imported_scene());
        let v = view(&chart, [0.0, 0.0]);
        let s = size();
        let b = band(O, s);
        // Clear of the one card, which sits in the canvas's top-left corner.
        assert_eq!(
            hit_test(&v, b[0] + b[2] - 10.0, b[1] + b[3] - 10.0, O, s),
            Some(MapAction::PanStart)
        );
        assert_eq!(
            hit_test(&v, O[0] + 4.0, O[1] + 2.0, O, s),
            Some(MapAction::Consume),
            "the title bar is still the panel's"
        );
        assert_eq!(hit_test(&v, O[0] - 20.0, O[1] - 20.0, O, s), None);
        assert_eq!(hit_test(&v, O[0] + s[0] + 20.0, O[1] + 10.0, O, s), None);
    }

    // A world of props and materials is nowhere to be, so its map is the one
    // card summarizing it -- which stands for no asset and addresses none.
    fn a_world_of_no_places() -> Vec<WorldJsonlAsset> {
        vec![
            asset(RegisteredType::Material, "mat_floor", json!({})),
            asset(RegisteredType::Prop, "floor", json!({"mesh": "floor_mesh"})),
        ]
    }

    // The middle of the card drawn for `title`, which is where a click on it
    // lands.
    fn center(chart: &Chart, title: &str, s: [f32; 2], pan: [f32; 2]) -> [f32; 2] {
        let card = chart.cards.iter().find(|c| c.title == title).unwrap();
        let r = chart::card_rect(card, band(O, s), pan);
        [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5]
    }

    fn index(chart: &Chart, title: &str) -> usize {
        chart.cards.iter().position(|c| c.title == title).unwrap()
    }

    // The point of addressing every card: a press on one is the place it
    // stands for, not a grab on the canvas under it.
    #[test]
    fn a_press_on_a_place_selects_it_rather_than_grabbing_the_canvas() {
        let chart = mapped(imported_scene());
        let s = size();
        let v = view(&chart, [0.0, 0.0]);
        let at = center(&chart, "bistro", s, [0.0, 0.0]);
        assert_eq!(
            hit_test(&v, at[0], at[1], O, s),
            Some(MapAction::Select(index(&chart, "bistro")))
        );
    }

    // A card standing for no asset has nothing to select, and is still a card:
    // pressing it must not drag the canvas out from under the cursor either.
    #[test]
    fn a_press_on_a_card_addressing_nothing_is_swallowed() {
        let chart = mapped(a_world_of_no_places());
        assert_eq!(chart.cards[0].handle, None, "{:?}", chart.cards[0]);
        let s = size();
        let v = view(&chart, [0.0, 0.0]);
        let at = center(&chart, &chart.cards[0].title.clone(), s, [0.0, 0.0]);
        assert_eq!(hit_test(&v, at[0], at[1], O, s), Some(MapAction::Consume));
    }

    // What is selected anywhere in the editor is what the map lights up, so the
    // two surfaces agree on where the session is looking.
    #[test]
    fn the_selected_place_is_drawn_lit_up() {
        let chart = mapped(vec![
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(
                RegisteredType::MainMenu,
                "main",
                json!({"initial": true, "items": [{"label": "Start", "action": "scene:bistro"}]}),
            ),
        ]);
        let mut world = injected_world();
        let mut v = view(&chart, [0.0, 0.0]);
        v.selected = Some(index(&chart, "bistro"));
        place(&mut world, Some(&v), O, size());

        let lit = sprite(
            &world,
            CHART_IDS.card_bg(slot_of(&world, "bistro").unwrap()),
        );
        let plain = sprite(&world, CHART_IDS.card_bg(slot_of(&world, "main").unwrap()));
        assert!(
            lit.border_width > plain.border_width,
            "{lit:?} is drawn no differently to {plain:?}",
        );
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

        let mut world = injected_world();
        place(&mut world, Some(&view(&chart, pan)), O, size());
        assert!(
            slot_of(&world, "bistro").is_some(),
            "the root is not in view"
        );
    }

    // Toggled off, the panel leaves nothing of the map behind.
    #[test]
    fn hiding_the_panel_blanks_the_map_it_drew() {
        let chart = mapped(imported_scene());
        let mut world = injected_world();
        place(&mut world, Some(&view(&chart, [0.0, 0.0])), O, size());
        assert!(sprite(&world, CHART_IDS.card_bg(0)).visible);

        place(&mut world, None, O, size());
        for id in all_sprite_ids() {
            assert!(!sprite(&world, id).visible, "{id:?} survived the hide");
        }
        for id in all_label_ids() {
            assert!(!label(&world, id).visible, "{id:?} survived the hide");
        }
    }
}
