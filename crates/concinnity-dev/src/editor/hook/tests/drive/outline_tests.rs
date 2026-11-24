//! The extent-outline drive (`hook/drive/outline.rs`): the lines a selected
//! `TriggerVolume` publishes for its own shape.

use concinnity_engine::components::TriggerVolume;
use concinnity_engine::protocol::WorldLines;
use concinnity_host::thread::asset_id;

use crate::frame_hook::FrameHook;

use crate::editor::hook::tests::fixtures::{
    active, click_at, hook, marks, pick_world, release_at, select,
};

use crate::editor::{outlines, view_menu};

use crate::editor::viewport::billboards;

// Selecting a trigger volume publishes its collider outline through the line
// pass (12 box edges alongside the 6 axis runs), with no dotted outline over
// the viewport (that belongs to the drag ghost), and clearing the selection
// leaves only the axes.
#[test]
fn selected_trigger_volume_publishes_its_line_outline() {
    asset_id::begin_build();
    let mut world = pick_world([0.0; 3], Vec::new());
    let entity = world.push(TriggerVolume {
        position: [0.0, 0.0, -6.0],
        ..Default::default()
    });
    let id = asset_id::intern("zone");
    world.identify(entity, id);
    let mut h = hook(vec![serde_json::json!({
        "type": "TriggerVolume",
        "args": {"$id": "zone", "position": [0.0, 0.0, -6.0]}
    })]);

    // Click the volume's projected icon (viewport center): it selects and its
    // outline comes up in the published line buffer.
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(active(&h).as_deref(), Some("zone"));
    let published = world.resource::<WorldLines>().unwrap().0.len();
    assert_eq!(
        published,
        6 + outlines::shapes::BOX_EDGES,
        "axes plus the volume's box edges"
    );
    let tint = billboards::tint("TriggerVolume");
    assert!(
        marks(&world).iter().filter(|m| m.fill == tint).count() == 0,
        "no dotted outline draws; the volume shows only as its icon"
    );

    // Clearing the selection leaves only the axes (the button is released
    // first, so the tick does not re-pick the icon under the cursor).
    release_at(&mut world, &mut h, [640.0, 360.0]);
    h.selection.clear();
    h.tick(&mut world);
    let published = world.resource::<WorldLines>().unwrap().0.len();
    assert_eq!(published, 6, "no outline without a selected volume");

    // Clearing the Lines show flag publishes nothing at all, selection and
    // axes included: the pass it would feed is masked for the frame anyway.
    select(&mut h, &["zone"]);
    h.show_flags = h.show_flags.toggled(view_menu::ShowFlags::LINES);
    h.tick(&mut world);
    let published = world.resource::<WorldLines>().unwrap().0.len();
    assert_eq!(published, 0, "the Lines flag gates axes and outlines alike");
}
