// src/editor/hook/tests/pick_tests.rs
//
// Picking in the viewport (`hook/pick.rs`): the nearest hit a click resolves
// to, the cycle a repeat click walks, the clear on empty space, the ring the
// selection draws, and the rows the session's hide and lock flags make
// unpickable.

use concinnity_core::components::FrameInput;
use concinnity_core::components::Sprite;
use concinnity_core::ecs::World;
use concinnity_host::thread::asset_id;

use super::fixtures::{click_at, entry, hook, pick_world, release_at, set_input};
use crate::debug_hook::DebugHook;
use crate::editor::hook::FormTarget;

use crate::editor::panels::asset_tree::{self, TreeGroup};

use crate::editor::panels::panel;

use crate::editor::sim;
use crate::editor::viewport::highlight;
use crate::test_support::isolate_state_dir;

// A click through the scene picks the nearest hit and nothing more: no
// panel or form opens on a viewport click, so clicking around the scene never
// spawns UI.
#[test]
fn viewport_click_picks_the_nearest_prop_without_opening_a_form() {
    asset_id::reset_interner();
    let near = asset_id::intern("box_near");
    let far = asset_id::intern("box_far");
    let mut world = pick_world(
        [0.0; 3],
        vec![
            // Both boxes straddle the -Z axis; the near one at z ~ -5.
            (far, [-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]),
            (near, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0]),
        ],
    );
    let mut h = hook(vec![
        entry("box_near", "Sprite"),
        entry("box_far", "Sprite"),
    ]);

    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("box_near"), "nearest hit wins");
    assert!(!h.panel_open, "a viewport click opens no panel");
    assert!(!h.form_open(), "a viewport click opens no form");

    // An already-open form follows the pick instead.
    h.open_asset_form("box_far", &mut world);
    assert_eq!(h.form_target, FormTarget::Entry(1));
    // Off the repeat-click slop, so this is a fresh pick and not a cycle.
    click_at(&mut world, &mut h, [650.0, 370.0]);
    assert_eq!(
        h.form_target,
        FormTarget::Entry(0),
        "the open form retargets to the picked entry"
    );
    assert_eq!(h.selected_type.as_deref(), Some("Sprite"));
}

// A second click on the same spot cycles to the occluded hit; a click away
// from any box clears the selection and the cycle. The boxes sit down-left of
// the camera axis so the clicks land in the screen region no default panel
// covers (the first pick opens the edit form, which claims center presses).
#[test]
fn repeat_viewport_clicks_cycle_and_empty_space_clears() {
    asset_id::reset_interner();
    let near = asset_id::intern("box_near");
    let far = asset_id::intern("box_far");
    // The ray through pixel [200, 600] (fov 90, 1280x720) passes ~[-6.1, -3.3]
    // at depth 5 and ~[-12.2, -6.7] at depth 10; both boxes straddle it.
    let mut world = pick_world(
        [0.0; 3],
        vec![
            (near, [-7.1, -4.3, -6.0], [-5.1, -2.3, -4.0]),
            (far, [-13.2, -7.7, -11.0], [-11.2, -5.7, -9.0]),
        ],
    );
    let mut h = hook(vec![
        entry("box_near", "Sprite"),
        entry("box_far", "Sprite"),
    ]);

    click_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(h.selection.active(), Some("box_near"));
    click_at(&mut world, &mut h, [201.0, 601.0]);
    assert_eq!(
        h.selection.active(),
        Some("box_far"),
        "a repeat click reaches the occluded box"
    );
    assert_eq!(
        h.selection.iter().count(),
        1,
        "a plain click replaces, never accumulates"
    );
    click_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(
        h.selection.active(),
        Some("box_near"),
        "the cycle wraps back to the front"
    );

    // Aim into the gutter between the Preview panel and the edit form, well
    // away from both boxes: the press arms a marquee, and the still release
    // clears.
    click_at(&mut world, &mut h, [250.0, 450.0]);
    assert_eq!(
        h.selection.active(),
        Some("box_near"),
        "the selection survives until the release decides click vs marquee"
    );
    release_at(&mut world, &mut h, [250.0, 450.0]);
    assert_eq!(
        h.selection.active(),
        None,
        "empty space clears the selection"
    );
}

// Picking an asset the world does not declare opens no form until the tree
// knows how (if at all) it can be promoted: an unknown name is selectable only.
#[test]
fn viewport_click_on_an_unknown_asset_selects_without_a_form() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    asset_id::reset_interner();
    let generated = asset_id::intern("some_generated_asset");
    let mut world = pick_world(
        [0.0; 3],
        vec![(generated, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])],
    );
    let mut h = hook(vec![entry("box_near", "Sprite")]);

    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("some_generated_asset"));
    assert!(!h.panel_open, "no panel opens on a viewport click");
    assert_eq!(h.form_target, FormTarget::New);
    assert!(!h.form_open());
}

// Undo/redo invalidates the pick state along with the other entry-indexed UI.
#[test]
fn history_jumps_clear_the_pick_selection() {
    asset_id::reset_interner();
    let id = asset_id::intern("box_near");
    let mut world = pick_world([0.0; 3], vec![(id, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])]);
    let mut h = hook(vec![entry("box_near", "Sprite")]);
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("box_near"));

    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    h.undo(&mut world);
    assert_eq!(
        h.selection.active(),
        None,
        "a history jump drops the selection"
    );
}

// The selection ring follows the picked asset's projected bounds and hides
// outside edit mode.
#[test]
fn selection_ring_tracks_the_picked_asset() {
    asset_id::reset_interner();
    let id = asset_id::intern("box_near");
    let mut world = pick_world([0.0; 3], vec![(id, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])]);
    let mut h = hook(vec![entry("box_near", "Sprite")]);
    click_at(&mut world, &mut h, [640.0, 360.0]);

    let ring = |world: &World| {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == highlight::all_sprite_ids()[0])
            .cloned()
            .expect("outline sprite injected")
    };
    let s = ring(&world);
    assert!(s.visible, "the ring shows on pick");
    let (cx, cy) = (s.x + s.width * 0.5, s.y + s.height * 0.5);
    assert!(
        (cx - 640.0).abs() < 2.0 && (cy - 360.0).abs() < 2.0,
        "ring centered on the box, got ({cx}, {cy})"
    );
    assert!(s.border_width > 0.0 && s.tint[3] == 0.0, "border-only ring");

    // Play mode hides the ring; returning to edit mode restores it.
    h.sim.state = sim::SimState::Playing;
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(!ring(&world).visible, "hidden in play mode");
    h.sim.state = sim::SimState::Stopped;
    h.tick(&mut world);
    assert!(ring(&world).visible, "back in edit mode it returns");
}

// A locked asset is skipped by viewport picking: the click passes through to
// empty space (arming the marquee) instead of selecting it.
#[test]
fn locked_assets_are_skipped_by_viewport_picking() {
    asset_id::reset_interner();
    let near = asset_id::intern("box_near");
    let mut world = pick_world([0.0; 3], vec![(near, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0])]);
    let mut h = hook(vec![entry("box_near", "Sprite")]);
    h.locked_assets.insert("box_near".to_string());

    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), None, "the locked box is not picked");
    assert!(h.marquee.is_some(), "the click fell through to empty space");
}

// A viewport pick unfolds the picked asset's group and scrolls its row into
// the tree's window.
#[test]
fn viewport_pick_reveals_the_tree_row() {
    let world = World::new();
    let mut h = hook(Vec::new());
    h.panel_open = true;
    h.tree_stale = false;
    h.tree_groups = vec![TreeGroup {
        label: asset_tree::WORLD_GROUP.to_string(),
        assets: (0..30)
            .map(|i| asset_tree::TreeAsset {
                name: format!("a{i:02}"),
                asset_type: "Sprite".to_string(),
                badge: asset_tree::Badge::Authored,
                promote: None,
            })
            .collect(),
    }];

    h.reveal_in_tree("a25", &world);
    assert_eq!(h.tree_unfolded, vec![0], "the group unfolds");
    // Rows: header at 0, a25 at 26; the scroll clamps to the last window.
    assert_eq!(h.tree_scroll, 31 - panel::ROW_POOL);

    // A revealed row already inside the window leaves the scroll alone.
    h.reveal_in_tree("a20", &world);
    assert_eq!(h.tree_scroll, 31 - panel::ROW_POOL);
}
