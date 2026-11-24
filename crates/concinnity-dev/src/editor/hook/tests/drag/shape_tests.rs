//! The character-shape slider drag (`hook/drag/shape.rs`): the values a drag
//! writes as it is held, the single undo step letting go commits, and the
//! start state Escape restores.

use concinnity_engine::components::{FrameInput, UiEvent, UiValue};
use concinnity_engine::ecs::World;

use crate::frame_hook::FrameHook;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{hook, select, set_input, shape_world_entries};

use crate::editor::install;

use crate::editor::panels::registry::PanelKey;

fn send(world: &mut World, h: &mut EditorHook, name: &str, value: f32) {
    world.events_mut::<UiEvent>().send(UiEvent {
        name: name.into(),
        element: None,
        value: Some(UiValue::Float(value)),
    });
    h.read_ui_events(world);
}

fn opened() -> (World, EditorHook) {
    let mut world = World::new();
    install::editor_ui(&mut world);
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            ..Default::default()
        },
    );
    let mut h = hook(shape_world_entries());
    h.open[PanelKey::CharacterShape] = true;
    select(&mut h, &["body"]);
    h.tick(&mut world);
    (world, h)
}

// The full slider loop: changes while held, then the release. No rebuild is
// requested while the slider is held (the preview re-resolves the live pose
// instead); the release commits the value to the shape entry as ONE undo
// step, and undo restores the pre-drag args.
#[test]
fn shape_slider_drag_commits_one_undo_step() {
    let (mut world, mut h) = opened();
    let data = h.shape_data(&world);
    assert_eq!(
        data.binding.as_ref().and_then(|b| b.shape_idx),
        Some(1),
        "the selected mesh binds its shape"
    );
    let names: Vec<&str> = data
        .derived
        .sliders
        .iter()
        .map(|r| r.name.as_str())
        .collect();
    assert_eq!(names, ["jaw", "muscle", "leg_length"]);
    assert!(
        !data.presets.is_empty(),
        "a plain SkinnedMesh gets the bundled humanoid schema"
    );

    send(&mut world, &mut h, "editor.shape.change.0", 0.5);
    assert!(h.shape.drag.is_some(), "the first change starts a drag");
    assert!(
        !h.dirty && !h.rebuild_preview,
        "no entry change until release"
    );
    let shown = |h: &EditorHook| h.make_shape_view().unwrap().values[0];
    assert!((shown(&h) - 0.5).abs() < 1e-3, "the working value shows");

    send(&mut world, &mut h, "editor.shape.change.0", -0.25);
    assert!(!h.rebuild_preview, "dragging never rebuilds the preview");
    assert!(!h.can_undo(), "nothing recorded mid-drag");

    send(&mut world, &mut h, "editor.shape.commit.0", -0.25);
    assert!(h.shape.drag.is_none(), "release ends the drag");
    assert!(
        h.dirty && h.rebuild_preview,
        "the release is one committed edit"
    );
    let sliders = h.entries[1]["args"]["sliders"].as_array().unwrap().clone();
    let jaw = sliders
        .iter()
        .find(|s| s["name"] == "jaw")
        .expect("jaw written");
    assert!((jaw["value"].as_f64().unwrap() + 0.25).abs() < 1e-6);
    assert!(
        sliders.iter().any(|s| s["name"] == "muscle"),
        "the untouched slider rides along"
    );
    assert!(h.can_undo());

    h.undo();
    assert!(!h.can_undo(), "the whole drag was one step");
    let sliders = h.entries[1]["args"]["sliders"].as_array().unwrap();
    assert_eq!(sliders.len(), 1);
    assert_eq!(sliders[0]["name"], "muscle");
}

// Escape mid-drag shows the start state again, and the release that follows
// commits nothing.
#[test]
fn escape_cancels_a_shape_slider_drag() {
    let (mut world, mut h) = opened();
    send(&mut world, &mut h, "editor.shape.change.0", 0.75);
    h.drive_shape_drag(
        &FrameInput {
            escape: true,
            ..Default::default()
        },
        &mut world,
    );
    send(&mut world, &mut h, "editor.shape.change.0", 0.5);
    send(&mut world, &mut h, "editor.shape.commit.0", 0.5);
    assert!(h.shape.drag.is_none());
    assert!(
        !h.dirty && !h.can_undo(),
        "a cancelled drag commits nothing"
    );
}
