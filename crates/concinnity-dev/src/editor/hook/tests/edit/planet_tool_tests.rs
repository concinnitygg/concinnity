//! The Planet panel's tools at work (`hook/edit/planet_tool.rs`): a picked
//! tool takes the viewport's presses from selection, its drag lays a stroke
//! of points along the ground, the brush ring follows the cursor, and the
//! tool keys and Escape pick and put it down.

use concinnity_engine::components::{FrameInput, InputKey, KeyEvent, KeyMods, KeyPress};
use concinnity_engine::ecs::{World, WorldLines};
use concinnity_engine::planet::{Continents, LocalFrame, PlanetFrame, PlanetShape};
use concinnity_host::thread::asset_id;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{
    active, click_at, drag_to, hook, pick_world, release_at, set_input,
};
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::planet_tools::Tool;
use crate::editor::planet_tools::ring::RING_POINTS;
use crate::frame_hook::FrameHook;

// Below the view's center, where the ray from a camera 10 m up lands on the
// ground about 15 m ahead, on the box standing there.
const ON_BOX: [f32; 2] = [300.0, 600.0];

// A camera 10 m over a smooth planet whose ground is at y = 0, a box on the
// ground ahead, and an open Planet panel.
fn rig() -> (EditorHook, World) {
    asset_id::reset_interner();
    let id = asset_id::intern("crate");
    let mut world = pick_world(
        [0.0, 10.0, 0.0],
        vec![(id, [-30.0, -1.0, -16.0], [30.0, 1.0, -14.0])],
    );
    world.insert_resource(PlanetFrame {
        shape: PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 0.0,
            feature_size: 2_000.0,
            octaves: 1,
            seed: 1,
            continents: Continents::NONE,
            edits: None,
        },
        frame: LocalFrame::AUTHORED,
    });
    let mut h = hook(vec![
        serde_json::json!({"type": "Prop", "args": {"$id": "crate"}}),
        serde_json::json!({"type": "Planet", "args": {"layers": [{"grass": "meadow"}]}}),
    ]);
    registry::panel(PanelKey::Planet).toggle(&mut h, &mut world);
    (h, world)
}

fn key_tick(world: &mut World, h: &mut EditorHook, key: InputKey) {
    set_input(
        world,
        FrameInput {
            viewport: [1280.0, 720.0],
            key_events: vec![KeyEvent::Press(KeyPress::new(key, KeyMods::default()))],
            ..Default::default()
        },
    );
    h.tick(world);
}

fn lines(world: &World) -> usize {
    world.resource::<WorldLines>().map_or(0, |l| l.0.len())
}

#[test]
fn without_a_tool_the_press_selects() {
    let (mut h, mut world) = rig();
    click_at(&mut world, &mut h, ON_BOX);
    assert_eq!(active(&h).as_deref(), Some("crate"));
    assert!(h.planet_tool.stroke.is_none());
}

// With a tool picked the press starts a stroke instead of a pick, its first
// point the ground under the cursor; a drag lays more, and the release ends
// it.
#[test]
fn a_picked_tool_takes_the_press_and_lays_a_stroke() {
    let (mut h, mut world) = rig();
    h.planet_tool.tools.toggle(Tool::Raise);
    h.planet_tool.tools.brush.radius = 2.0;
    click_at(&mut world, &mut h, ON_BOX);
    assert_eq!(active(&h), None, "nothing is selected");
    assert!(h.gizmo_drag.is_none() && h.marquee.is_none());
    let stroke = h.planet_tool.stroke.expect("the press starts a stroke");
    assert_eq!(stroke.tool, Tool::Raise);
    assert_eq!(h.planet_tool.points.len(), 1);
    let first = h.planet_tool.points[0];
    assert!(first.height_above_ground.abs() < 1e-3);
    assert!(first.latitude > 0.0, "ahead is north");

    // Dragging sideways lays points half a meter apart.
    drag_to(&mut world, &mut h, [ON_BOX[0] + 40.0, ON_BOX[1]]);
    let laid = h.planet_tool.points.len();
    assert!(laid >= 2, "{laid}");
    release_at(&mut world, &mut h, [ON_BOX[0] + 40.0, ON_BOX[1]]);
    assert!(h.planet_tool.stroke.is_none());
    assert!(!h.dirty, "the tools change nothing yet");
}

// Off the planet, a stroke lays nothing until it is back on the ground.
#[test]
fn a_stroke_off_the_ground_lays_nothing() {
    let (mut h, mut world) = rig();
    h.planet_tool.tools.toggle(Tool::Smooth);
    // Above the horizon.
    click_at(&mut world, &mut h, [300.0, 200.0]);
    assert!(
        h.planet_tool.stroke.is_some(),
        "the press is still the tool's"
    );
    assert!(h.planet_tool.points.is_empty());
    drag_to(&mut world, &mut h, ON_BOX);
    assert_eq!(h.planet_tool.points.len(), 1);
}

#[test]
fn the_ring_follows_the_cursor_only_while_a_tool_is_picked() {
    let (mut h, mut world) = rig();
    release_at(&mut world, &mut h, ON_BOX);
    let without = lines(&world);
    h.planet_tool.tools.toggle(Tool::Flatten);
    release_at(&mut world, &mut h, ON_BOX);
    let ring = RING_POINTS * 2 + 1;
    assert_eq!(lines(&world), without + ring);
    assert!(h.planet_tool.hover.is_some());
    // Over the sky there is no ground to ring.
    release_at(&mut world, &mut h, [300.0, 200.0]);
    assert_eq!(lines(&world), without);
}

#[test]
fn the_tool_keys_pick_and_escape_puts_down() {
    let (mut h, mut world) = rig();
    key_tick(&mut world, &mut h, InputKey::G);
    assert_eq!(h.planet_tool.tools.active, Some(Tool::PaintGrass));
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            escape: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(h.planet_tool.tools.active, None);
}

// A box of the panel's being typed into keeps its letters.
#[test]
fn the_tool_keys_stand_down_while_a_box_is_typed_in() {
    let (mut h, mut world) = rig();
    h.planet.focus = Some(0);
    key_tick(&mut world, &mut h, InputKey::U);
    assert_eq!(h.planet_tool.tools.active, None);
}

#[test]
fn closing_the_panel_puts_the_tool_down() {
    let (mut h, mut world) = rig();
    h.planet_tool.tools.toggle(Tool::Erase);
    registry::panel(PanelKey::Planet).close(&mut h, &mut world);
    assert_eq!(h.planet_tool.tools.active, None);
}
