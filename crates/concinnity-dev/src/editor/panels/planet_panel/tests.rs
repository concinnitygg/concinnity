use super::*;
use concinnity_engine::components::{Sprite, TextInput, TextLabel};
use serde_json::json;

use crate::editor::planet_tools::{Brush, Tools};

const IDLE: Tools = Tools {
    active: None,
    brush: Brush {
        radius: 8.0,
        strength: 0.5,
        falloff: 0.5,
    },
    layer: 0,
    cover: 0,
};

fn strip() -> f32 {
    planet_tool_strip::height(&IDLE)
}

fn planet_rows() -> Vec<Row> {
    let args = json!({
        "seed": 3,
        "sea": {"level": 0.0, "water": "sea"},
        "covers": [{"material": "sand"}]
    });
    planet::rows(Some(&planet::working(args.as_object().unwrap())))
}

fn view(rows: &[Row], scroll: usize) -> PlanetView<'_> {
    PlanetView {
        rows,
        scroll,
        focus: None,
        status: None,
        dirty: false,
        mouse: [0.0, 0.0],
        strip: StripView {
            tools: &IDLE,
            pick: "",
            sliding: None,
            mouse: [0.0, 0.0],
        },
    }
}

fn position(rows: &[Row], key: &str) -> usize {
    rows.iter()
        .position(|r| r.field().is_some_and(|f| f.key == key))
        .unwrap()
}

// Inside slot `j` of the window, clear of its buttons.
fn in_slot(o: [f32; 2], j: usize) -> (f32, f32) {
    let r = slot_rect(o, PLANET_W, strip(), j);
    (r[0] + 5.0, r[1] + 5.0)
}

#[test]
fn hit_test_resolves_each_control() {
    let rows = planet_rows();
    let v = view(&rows, 0);
    let o = [30.0, 30.0];
    let s = size(rows.len(), strip());
    let a = apply_rect(o, PLANET_W);
    assert_eq!(
        hit_test(&v, a[0] + 2.0, a[1] + 2.0, o, s),
        Some(PlanetAction::Apply)
    );
    let radius = position(&rows, "radius");
    let (x, y) = in_slot(o, radius);
    assert_eq!(hit_test(&v, x, y, o, s), Some(PlanetAction::Focus(radius)));
    let seed = position(&rows, "seed");
    let b = button_rect(slot_rect(o, PLANET_W, strip(), seed), 0);
    assert_eq!(
        hit_test(&v, b[0] + 2.0, b[1] + 2.0, o, s),
        Some(PlanetAction::Button(Button::Reroll))
    );
    let water = position(&rows, "sea.water");
    let (x, y) = in_slot(o, water);
    assert_eq!(
        hit_test(&v, x, y, o, s),
        Some(PlanetAction::CycleRef(water))
    );
    let (x, y) = in_slot(o, 0);
    assert_eq!(
        hit_test(&v, x, y, o, s),
        Some(PlanetAction::Consume),
        "a heading"
    );
    assert_eq!(hit_test(&v, 5000.0, 5000.0, o, s), None);
}

// The window starts at the scroll: slot 0 is the first row scrolled to.
#[test]
fn hit_test_follows_the_scroll() {
    let rows = planet_rows();
    let gravity = position(&rows, "gravity");
    let scroll = max_scroll(rows.len());
    assert!(scroll > 0, "a planet with a cover overflows the window");
    let v = view(&rows, scroll);
    let o = [0.0, 0.0];
    let (x, y) = in_slot(o, gravity - scroll);
    assert_eq!(
        hit_test(&v, x, y, o, size(rows.len(), strip())),
        Some(PlanetAction::Focus(gravity))
    );
}

#[test]
fn a_world_without_a_planet_offers_to_add_one() {
    let rows = planet::rows(None);
    let v = view(&rows, 0);
    let o = [0.0, 0.0];
    let (x, y) = in_slot(o, 0);
    assert_eq!(
        hit_test(&v, x, y, o, size(rows.len(), strip())),
        Some(PlanetAction::AddPlanet)
    );
}

#[test]
fn the_window_holds_at_most_its_rows() {
    assert_eq!(size(3, strip())[1], size(1, strip())[1] + 2.0 * ROW_H);
    assert_eq!(size(500, strip()), size(VISIBLE_ROWS, strip()));
    assert_eq!(max_scroll(5), 0);
}

#[test]
fn place_shows_the_window_and_hides_the_rest() {
    let mut world = ids().test_world();
    let rows = planet_rows();
    let o = [20.0, 20.0];
    place(&mut world, &view(&rows, 0), o, size(rows.len(), strip()));
    let title = world.get_by_id::<TextLabel>(TITLE_LABEL).unwrap();
    assert!(title.visible && title.content == "Planet");
    assert_eq!(
        world.get_by_id::<TextLabel>(row_label(0)).unwrap().content,
        "Shape"
    );
    let radius = position(&rows, "radius");
    assert!(world.get_by_id::<TextInput>(input(radius)).unwrap().visible);
    let water = position(&rows, "sea.water");
    assert!(world.get_by_id::<Sprite>(picker_bg(water)).unwrap().visible);
    assert!(!world.get_by_id::<TextInput>(input(water)).unwrap().visible);
    // A row past the window keeps its input hidden.
    let gravity = position(&rows, "gravity");
    assert!(gravity >= VISIBLE_ROWS);
    assert!(
        !world
            .get_by_id::<TextInput>(input(gravity))
            .unwrap()
            .visible
    );
    assert!(!world.get_by_id::<TextLabel>(STATUS_LABEL).unwrap().visible);

    // Scrolled to the end, gravity shows and the first rows are gone.
    let scroll = max_scroll(rows.len());
    place(
        &mut world,
        &view(&rows, scroll),
        o,
        size(rows.len(), strip()),
    );
    assert!(
        world
            .get_by_id::<TextInput>(input(gravity))
            .unwrap()
            .visible
    );
    assert!(!world.get_by_id::<TextInput>(input(radius)).unwrap().visible);
}

#[test]
fn a_short_row_list_blanks_the_unused_slots() {
    let mut world = ids().test_world();
    let rows = planet::rows(None);
    place(
        &mut world,
        &view(&rows, 0),
        [0.0, 0.0],
        size(rows.len(), strip()),
    );
    assert_eq!(
        world.get_by_id::<TextLabel>(row_label(0)).unwrap().content,
        "+ Add Planet"
    );
    assert!(!world.get_by_id::<Sprite>(row_bg(1)).unwrap().visible);
}
