//! Dropping the selection to the floor (`hook/drop_floor.rs`): the surface
//! below each member, and the one undo step the landing commits.

use concinnity_core::components::Transform;
use concinnity_host::thread::asset_id;

use super::fixtures::{hook, pick_world};
use crate::editor::hook::tests::fixtures::select;

// Drop-to-floor lands an indexed member's bounds on the surface below it,
// rests a bounds-less member's origin on the ground-plane fallback, and
// commits the batch as one undo step.
#[test]
fn drop_to_floor_lands_the_selection_on_the_surface_below() {
    asset_id::reset_interner();
    let a = asset_id::intern("box_a");
    let _g = asset_id::intern("ground");
    let lamp_id = asset_id::intern("lamp");
    let mut world = pick_world(
        [0.0; 3],
        vec![
            (a, [-1.0, 4.0, -6.0], [1.0, 6.0, -4.0]),
            (_g, [-10.0, -1.0, -10.0], [10.0, 0.0, 10.0]),
        ],
    );
    let box_e = world.push(Transform {
        position: [0.0, 5.0, -5.0],
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    // The lamp has no pick-index bounds and sits clear of the ground box, so
    // it exercises both the position-as-foot path and the y=0 fallback.
    let lamp_e = world.push(Transform {
        position: [100.0, 3.0, 0.0],
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    world.identify(box_e, a);
    world.identify(lamp_e, lamp_id);

    let mut h = hook(vec![
        serde_json::json!({
            "type": "Prop", "args": { "$id": "box_a", "position": [0.0, 5.0, -5.0] }
        }),
        serde_json::json!({
            "type": "Prop", "args": { "$id": "ground", "position": [0.0, 0.0, 0.0] }
        }),
        serde_json::json!({
            "type": "PointLight", "args": { "$id": "lamp", "position": [100.0, 3.0, 0.0] }
        }),
    ]);
    select(&mut h, &["box_a", "lamp"]);

    h.run_console_line(&mut world, "/floor");
    assert_eq!(
        h.entries[0]["args"]["position"],
        serde_json::json!([0.0, 1.0, -5.0]),
        "the box bottom (1 below its position) rests on the ground top"
    );
    assert_eq!(
        h.entries[2]["args"]["position"],
        serde_json::json!([100.0, 0.0, 0.0]),
        "nothing below the lamp: its origin lands on the y=0 fallback"
    );
    let live = world.get::<Transform>(box_e).unwrap();
    assert_eq!(
        live.position,
        [0.0, 1.0, -5.0],
        "the live transform follows"
    );
    let lines = h.console_sink.window(0, 16);
    assert!(lines.iter().any(|l| l.text == "dropped 2"));

    h.undo(&mut world);
    assert_eq!(
        h.entries[0]["args"]["position"],
        serde_json::json!([0.0, 5.0, -5.0])
    );
    assert!(!h.can_undo(), "the whole drop was one step");

    // A selection with no eligible member (no live entity) drops nothing and
    // records no undo step.
    select(&mut h, &["ground"]);
    h.run_console_line(&mut world, "/floor");
    assert!(!h.can_undo(), "a no-op drop records nothing");
}
