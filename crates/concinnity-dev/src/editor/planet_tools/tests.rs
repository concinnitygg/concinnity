use super::*;
use concinnity_engine::components::KeyMods;

fn press(key: InputKey) -> KeyPress {
    KeyPress::new(key, KeyMods::default())
}

#[test]
fn every_tool_has_its_own_key() {
    for tool in Tool::ALL {
        assert_eq!(Tool::for_key(tool.key()), Some(tool));
    }
    assert_eq!(Tool::for_key(InputKey::T), None);
}

// A tool's key picks it, again puts it down, and another tool's key swaps.
#[test]
fn keys_pick_swap_and_put_down_tools() {
    let mut t = Tools::default();
    assert!(t.key(press(InputKey::U)));
    assert_eq!(t.active, Some(Tool::Raise));
    assert!(t.key(press(InputKey::G)));
    assert_eq!(t.active, Some(Tool::PaintGrass));
    assert!(t.key(press(InputKey::G)));
    assert_eq!(t.active, None);
}

#[test]
fn modified_repeated_and_foreign_keys_are_left_alone() {
    let mut t = Tools::default();
    let ctrl = KeyPress::new(InputKey::C, KeyMods::CTRL);
    assert!(!t.key(ctrl));
    assert!(!t.key(press(InputKey::T)));
    let mut held = press(InputKey::U);
    held.repeat = true;
    t.toggle(Tool::Raise);
    assert!(t.key(held));
    assert_eq!(t.active, Some(Tool::Raise), "a held key does not flicker");
}

#[test]
fn brackets_size_the_brush_only_while_a_tool_is_picked() {
    let mut t = Tools::default();
    let start = t.brush.radius;
    assert!(!t.key(press(InputKey::RightBracket)));
    assert_eq!(t.brush.radius, start);
    t.toggle(Tool::Smooth);
    assert!(t.key(press(InputKey::RightBracket)));
    assert!(t.brush.radius > start);
    for _ in 0..100 {
        t.key(press(InputKey::LeftBracket));
    }
    assert_eq!(t.brush.radius, MIN_RADIUS);
}

// The radius slides on a log scale from its smallest to its largest, and
// every setting comes back from its slider position.
#[test]
fn slider_positions_round_trip() {
    let mut b = Brush::default();
    b.set_position(Setting::Radius, 0.0);
    assert_eq!(b.radius, MIN_RADIUS);
    b.set_position(Setting::Radius, 1.0);
    assert!((b.radius - MAX_RADIUS).abs() < 1e-3);
    b.set_position(Setting::Radius, 0.5);
    let mid = (MIN_RADIUS * MAX_RADIUS).sqrt();
    assert!((b.radius - mid).abs() < 1e-3, "{}", b.radius);
    for s in Setting::ALL {
        b.set_position(s, 0.3);
        assert!((b.position(s) - 0.3).abs() < 1e-5);
        b.set_position(s, 7.0);
        assert!((b.position(s) - 1.0).abs() < 1e-5);
    }
}

#[test]
fn only_the_paint_tools_pick_from_a_list() {
    assert_eq!(Tool::PaintGrass.list(), Some(List::Layers));
    assert_eq!(Tool::PaintCover.list(), Some(List::Covers));
    let others = Tool::ALL.iter().filter(|t| t.list().is_none()).count();
    assert_eq!(others, 6);
}

#[test]
fn picks_cycle_through_their_list_and_survive_a_shrink() {
    let mut t = Tools::default();
    t.cycle(List::Layers, 3);
    t.cycle(List::Layers, 3);
    assert_eq!(t.pick(List::Layers), 2);
    t.cycle(List::Layers, 3);
    assert_eq!(t.pick(List::Layers), 0);
    t.layer = 5;
    t.cycle(List::Layers, 2);
    assert_eq!(t.pick(List::Layers), 0);
    t.cycle(List::Covers, 0);
    assert_eq!(t.pick(List::Covers), 0);
}
