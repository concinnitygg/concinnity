// src/editor/hook/tests/drag/content_tests.rs
//
// Drag-out placement from the Content panel (`hook/drag/content.rs`): the entry
// a release commits where the ghost landed, the rotation that aligns a drop to
// the struck face, the still press that places nothing, and the material drag
// that assigns to the prop under the cursor instead.

use concinnity_core::components::Transform;
use concinnity_host::thread::asset_id;

use crate::debug_hook::DebugHook;

use crate::editor::hook::tests::fixtures::{click_at, drag_input, hook, pick_world, set_input};
use crate::editor::hook::{entry_name, entry_type};

use crate::editor::panels::registry::PanelKey;

// Dragging a mesh out of the Content grid places a Prop where the ghost
// lands: press a cell, pull into the viewport (ghost follows the surface
// below the cursor), release commits ONE undoable entry and selects it.
#[test]
fn drag_out_places_a_prop_where_the_ghost_lands() {
    asset_id::reset_interner();
    let ground = asset_id::intern("ground");
    let mut world = pick_world(
        [0.0; 3],
        vec![(ground, [-20.0, -1.0, -20.0], [20.0, 0.0, 20.0])],
    );
    let mut h = hook(vec![
        serde_json::json!({
            "name": "demo_ball", "type": "ProceduralMesh",
            "args": { "generator": "sphere" }
        }),
        serde_json::json!({
            "name": "ground", "type": "Prop",
            "args": { "mesh": "demo_ball", "position": [0.0, -0.5, 0.0] }
        }),
    ]);
    h.content_open = true;
    h.tree_stale = true;
    h.refresh_tree_if_needed();

    // Press the first grid cell: selects and arms the drag.
    let o = h.origin(PanelKey::Content, [1280.0, 720.0]);
    let cell = crate::editor::panels::content_panel::cell_rect(o, 0);
    click_at(&mut world, &mut h, [cell[0] + 10.0, cell[1] + 10.0]);
    assert!(h.content_drag.is_some(), "the cell press arms a drag");
    assert_eq!(h.selection.active(), Some("demo_ball"));
    let before = h.entries.len();

    // Pull into the viewport: the ghost lands on the ground box below.
    set_input(&mut world, drag_input([640.0, 500.0], true));
    h.tick(&mut world);
    let pose = h
        .content_ghost_pose()
        .expect("the ghost has a landing point");
    assert!(
        pose.position[1].abs() < 1e-3,
        "landed on the ground top: {pose:?}"
    );

    // Release commits one entry through the shared path.
    set_input(&mut world, drag_input([640.0, 500.0], false));
    h.tick(&mut world);
    assert!(h.content_drag.is_none());
    assert_eq!(h.entries.len(), before + 1);
    let placed = &h.entries[before];
    assert_eq!(entry_type(placed), Some("Prop"));
    assert_eq!(placed["args"]["mesh"], "demo_ball");
    assert!(
        placed["args"]["position"][1].as_f64().unwrap().abs() < 1e-3,
        "{placed}"
    );
    assert_eq!(h.selection.active(), entry_name(placed));
    assert!(h.dirty);
    h.undo(&mut world);
    assert_eq!(h.entries.len(), before, "one undo removes the placement");
}

// The align table really carries local +Y onto each AABB face normal under
// the engine's Euler convention: the model matrix's +Y column must equal the
// normal the face reports.
#[test]
fn align_rotation_maps_up_onto_every_face_normal() {
    for (axis, sign) in [
        (0, 1.0f32),
        (0, -1.0),
        (1, 1.0),
        (1, -1.0),
        (2, 1.0),
        (2, -1.0),
    ] {
        let rotation_deg = crate::editor::hook::drag::content::align_rotation(axis, sign);
        let m = Transform {
            position: [0.0; 3],
            rotation_deg,
            scale: [1.0; 3],
        }
        .model_matrix();
        // Column 1 is the rotated local +Y (column-major storage).
        let up = [m[1][0], m[1][1], m[1][2]];
        let mut normal = [0.0f32; 3];
        normal[axis] = sign;
        for c in 0..3 {
            assert!(
                (up[c] - normal[c]).abs() < 1e-5,
                "axis {axis} sign {sign}: +Y maps to {up:?}, want {normal:?}"
            );
        }
    }
}

// With align-to-surface on, dropping against a box side orients the placed
// prop to that face: a straight-on hit of the +Z face carries +Y onto +Z.
#[test]
fn aligned_drag_out_orients_the_drop_to_the_struck_face() {
    asset_id::reset_interner();
    let wall = asset_id::intern("wall");
    let mut world = pick_world([0.0; 3], vec![(wall, [-2.0, -2.0, -6.0], [2.0, 2.0, -4.0])]);
    let mut h = hook(vec![
        serde_json::json!({
            "name": "demo_ball", "type": "ProceduralMesh",
            "args": { "generator": "sphere" }
        }),
        serde_json::json!({
            "name": "wall", "type": "Prop",
            "args": { "mesh": "demo_ball", "position": [0.0, 0.0, -5.0] }
        }),
    ]);
    h.content_open = true;
    h.align_to_surface = true;
    h.tree_stale = true;
    h.refresh_tree_if_needed();

    let o = h.origin(PanelKey::Content, [1280.0, 720.0]);
    let cell = crate::editor::panels::content_panel::cell_rect(o, 0);
    click_at(&mut world, &mut h, [cell[0] + 10.0, cell[1] + 10.0]);
    let before = h.entries.len();

    // Straight down the camera axis into the box's near +Z face.
    set_input(&mut world, drag_input([640.0, 360.0], true));
    h.tick(&mut world);
    let pose = h.content_ghost_pose().expect("ghost up");
    assert_eq!(pose.normal, [0.0, 0.0, 1.0], "the +Z face was struck");
    assert_eq!(pose.rotation_deg, [90.0, 0.0, 0.0]);

    set_input(&mut world, drag_input([640.0, 360.0], false));
    h.tick(&mut world);
    let placed = &h.entries[before];
    assert_eq!(
        placed["args"]["rotation_deg"],
        serde_json::json!([90.0, 0.0, 0.0]),
        "{placed}"
    );
    assert!(
        (placed["args"]["position"][2].as_f64().unwrap() - (-4.0)).abs() < 1e-3,
        "landed on the face: {placed}"
    );

    // With the toggle off the same drop stays axis-aligned.
    h.undo(&mut world);
    h.align_to_surface = false;
    click_at(&mut world, &mut h, [cell[0] + 10.0, cell[1] + 10.0]);
    set_input(&mut world, drag_input([640.0, 360.0], true));
    h.tick(&mut world);
    set_input(&mut world, drag_input([640.0, 360.0], false));
    h.tick(&mut world);
    let placed = &h.entries[before];
    assert!(
        placed["args"].get("rotation_deg").is_none(),
        "no rotation written while align is off: {placed}"
    );
}

// A press that never travels past the slop is just the selecting click.
#[test]
fn a_still_cell_press_places_nothing() {
    asset_id::reset_interner();
    let mut world = pick_world([0.0; 3], vec![]);
    let mut h = hook(vec![serde_json::json!({
        "name": "demo_ball", "type": "ProceduralMesh", "args": { "generator": "box" }
    })]);
    h.content_open = true;
    h.tree_stale = true;
    h.refresh_tree_if_needed();
    let o = h.origin(PanelKey::Content, [1280.0, 720.0]);
    let cell = crate::editor::panels::content_panel::cell_rect(o, 0);
    let at = [cell[0] + 10.0, cell[1] + 10.0];
    click_at(&mut world, &mut h, at);
    set_input(&mut world, drag_input(at, false));
    h.tick(&mut world);
    assert!(h.content_drag.is_none());
    assert_eq!(h.entries.len(), 1, "no placement from a plain click");
    assert!(!h.dirty);
    assert_eq!(h.selection.active(), Some("demo_ball"), "still selected");
}

// Dragging a Material onto a Prop assigns it instead of placing anything.
#[test]
fn material_drag_assigns_to_the_prop_under_the_cursor() {
    asset_id::reset_interner();
    let crate_id = asset_id::intern("crate_prop");
    let mut world = pick_world(
        [0.0; 3],
        vec![(crate_id, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])],
    );
    let mut h = hook(vec![
        serde_json::json!({
            "name": "wood", "type": "Material", "args": { "roughness": 0.7 }
        }),
        serde_json::json!({
            "name": "crate_prop", "type": "Prop",
            "args": { "mesh": "demo", "position": [0.0, 0.0, -5.0] }
        }),
    ]);
    h.content_open = true;
    h.arm_content_drag("wood".to_string(), "Material".to_string(), [1000.0, 300.0]);

    set_input(&mut world, drag_input([640.0, 360.0], true));
    h.tick(&mut world);
    set_input(&mut world, drag_input([640.0, 360.0], false));
    h.tick(&mut world);
    assert_eq!(
        h.entries[1]["args"]["material"], "wood",
        "the hovered Prop gains the material"
    );
    assert_eq!(h.entries.len(), 2, "nothing was placed");
    assert!(h.dirty);
    h.undo(&mut world);
    assert!(h.entries[1]["args"].get("material").is_none());
}
