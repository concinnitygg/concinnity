// src/editor/hook/tests/drag/shape_tests.rs
//
// The character-shape slider drag (`hook/drag/shape.rs`): the values a drag
// writes as it is held, and the single undo step the release commits.

use concinnity_core::components::FrameInput;
use concinnity_core::ecs::World;

use crate::debug_hook::DebugHook;

use crate::editor::hook::tests::fixtures::{
    click_at, drag_to, hook, release_at, set_input, shape_world_entries,
};

use crate::editor::inject;

use crate::editor::panels::registry::PanelKey;

use crate::editor::widget_slider;

// The full slider loop: select the mesh, press a slider, drag, release. No
// rebuild is requested while the button is held (the preview re-resolves the
// live pose instead); the release commits the value to the shape entry as
// ONE undo step, and undo restores the pre-drag args.
#[test]
fn shape_slider_drag_commits_one_undo_step() {
    use crate::editor::panels::character_shape_panel as sp;
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let vp = [1280.0, 720.0];
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            ..Default::default()
        },
    );
    let mut h = hook(shape_world_entries());
    h.shape_open = true;
    h.focus_panel(PanelKey::CharacterShape);
    h.selection.set(vec!["body".to_string()]);
    h.tick(&mut world);
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
    let presets = data.presets.len();
    assert!(
        presets > 0,
        "a plain SkinnedMesh gets the bundled humanoid schema"
    );
    assert_eq!(
        h.shape_rows,
        presets + 1 + 6,
        "the preset rows, three headers + three sliders"
    );

    // The jaw slider follows the Face header, after the preset rows.
    let jaw_row = presets + 2;
    let o = h.origin(PanelKey::CharacterShape, vp);
    let rect = sp::slider_rect(sp::row_rect(o, sp::SHAPE_W, jaw_row));
    let bipolar = (-1.0, 1.0);
    let y = rect[1] + rect[3] * 0.5;
    let x_half = widget_slider::handle_x(rect, 0.5, bipolar);
    click_at(&mut world, &mut h, [x_half, y]);
    assert!(h.shape_drag.is_some(), "the press starts a drag");
    assert!(
        !h.dirty && !h.rebuild_preview,
        "no entry change until release"
    );
    let data = h.shape_data(&world);
    assert!(
        (data.values[0] - 0.5).abs() < 1e-3,
        "the working value follows"
    );

    let x_neg = widget_slider::handle_x(rect, -0.25, bipolar);
    drag_to(&mut world, &mut h, [x_neg, y]);
    assert!(!h.rebuild_preview, "dragging never rebuilds the preview");
    assert!(!h.can_undo(), "nothing recorded mid-drag");

    release_at(&mut world, &mut h, [x_neg, y]);
    assert!(h.shape_drag.is_none(), "release ends the drag");
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

    h.undo(&mut world);
    assert!(!h.can_undo(), "the whole drag was one step");
    let sliders = h.entries[1]["args"]["sliders"].as_array().unwrap();
    assert_eq!(sliders.len(), 1);
    assert_eq!(sliders[0]["name"], "muscle");
}
