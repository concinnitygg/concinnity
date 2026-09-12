// src/editor/hook/tests/drag/marquee_tests.rs
//
// Box-select (`hook/drag/marquee.rs`): the assets a swept rectangle replaces
// the selection with, the rings every member draws, and the shift-click that
// toggles one member in or out.

use concinnity_core::components::Sprite;
use concinnity_core::ecs::World;

use crate::editor::hook::tests::fixtures::{
    SIDE_A, SIDE_B, click_at, click_at_mod, drag_to, release_at, two_prop_rig,
};

use crate::editor::viewport::highlight;
use crate::editor::viewport::marquee;

// Shift-click toggles membership without disturbing the rest of the set.
#[test]
fn shift_click_toggles_selection_membership() {
    let (mut world, _, _, mut h) = two_prop_rig(SIDE_A, SIDE_B, 1.0);

    click_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(h.selection.active(), Some("box_a"));
    click_at_mod(&mut world, &mut h, [424.0, 598.0], true);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box_a", "box_b"],
        "shift-click adds the second box"
    );
    assert_eq!(h.selection.active(), Some("box_b"), "the newest is active");
    assert!(!h.form_open(), "a closed form stays closed");

    click_at_mod(&mut world, &mut h, [424.0, 598.0], true);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box_a"],
        "a second shift-click removes it again"
    );
}

// A drag from empty space boxes both props; shift-drag adds instead of
// replacing; a sub-slop release is the plain clearing click.
#[test]
fn marquee_drag_selects_the_boxed_assets() {
    let (mut world, _, _, mut h) = two_prop_rig(SIDE_A, SIDE_B, 1.0);

    // Press empty space (the ray misses both boxes), drag across both
    // projections, release.
    click_at(&mut world, &mut h, [80.0, 450.0]);
    assert!(h.marquee.is_some(), "an empty-space press arms the marquee");
    drag_to(&mut world, &mut h, [560.0, 700.0]);
    let rect = world
        .query::<Sprite>()
        .find(|s| s.asset_id == marquee::RECT)
        .cloned()
        .expect("marquee sprite injected");
    assert!(rect.visible, "the rect shows once the drag clears the slop");
    assert_eq!(
        (rect.x, rect.y, rect.width, rect.height),
        (80.0, 450.0, 480.0, 250.0)
    );
    release_at(&mut world, &mut h, [560.0, 700.0]);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box_a", "box_b"],
        "both boxed props are selected"
    );
    assert!(h.marquee.is_none(), "release ends the marquee");
    assert!(
        !world
            .query::<Sprite>()
            .find(|s| s.asset_id == marquee::RECT)
            .unwrap()
            .visible,
        "the rect hides after release"
    );

    // A fresh plain click on box_a, then a shift-drag over box_b only: added.
    click_at(&mut world, &mut h, [200.0, 600.0]);
    release_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(h.selection.iter().collect::<Vec<_>>(), ["box_a"]);
    // Starts in the Preview / edit-form gutter, right of box_a's projection so
    // the box encloses box_b alone.
    click_at_mod(&mut world, &mut h, [250.0, 450.0], true);
    assert!(h.marquee.is_some());
    drag_to(&mut world, &mut h, [560.0, 700.0]);
    release_at(&mut world, &mut h, [560.0, 700.0]);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box_a", "box_b"],
        "shift-drag adds without replacing"
    );

    // A still empty-space click clears; a still shift-click does not.
    click_at_mod(&mut world, &mut h, [80.0, 450.0], true);
    release_at(&mut world, &mut h, [81.0, 450.0]);
    assert_eq!(h.selection.iter().count(), 2, "shift keeps the selection");
    click_at(&mut world, &mut h, [80.0, 450.0]);
    release_at(&mut world, &mut h, [81.0, 450.0]);
    assert_eq!(h.selection.active(), None, "plain still release clears");
}

// Every selection member gets a ring; the active member's is brighter.
#[test]
fn selection_rings_cover_every_member() {
    let (mut world, _, _, mut h) = two_prop_rig(SIDE_A, SIDE_B, 1.0);
    click_at(&mut world, &mut h, [200.0, 600.0]);
    click_at_mod(&mut world, &mut h, [424.0, 598.0], true);

    let ids = highlight::all_sprite_ids();
    let ring = |world: &World, i: usize| {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == ids[i])
            .cloned()
            .expect("ring pool injected")
    };
    let (r0, r1) = (ring(&world, 0), ring(&world, 1));
    assert!(r0.visible && r1.visible, "one ring per member");
    assert!(
        !ring(&world, 2).visible,
        "the rest of the pool stays hidden"
    );
    assert_ne!(
        r0.border_color, r1.border_color,
        "the active member's ring is distinguished"
    );
    let center = |r: &Sprite| (r.x + r.width * 0.5, r.y + r.height * 0.5);
    let (c0x, _) = center(&r0);
    let (c1x, _) = center(&r1);
    assert!(
        c0x < c1x,
        "rings follow selection order: box_a left of box_b"
    );
}
