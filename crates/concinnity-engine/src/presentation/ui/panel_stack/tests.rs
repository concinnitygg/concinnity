use super::*;
use crate::components::{LayoutChild, Sprite};
use crate::ecs::World;
use crate::protocol::ScreenStack;
use alloc::vec;

// A world of two overlapping panels, each holding one sprite; the second
// holds its sprite through a nested layout. `ranks` are the panels' ranks.
fn world(ranks: [u32; 2]) -> (World, [Entity; 2], [Entity; 2]) {
    let mut world = World::new();
    let a_sprite = world.push_identified(AssetId(10), Sprite::default());
    let b_sprite = world.push_identified(AssetId(20), Sprite::default());
    world.push_identified(
        AssetId(21),
        Layout {
            children: vec![child(20)],
            ..Default::default()
        },
    );
    let a = world.push_identified(AssetId(1), panel([0.0, 0.0], ranks[0], 10));
    let b = world.push_identified(AssetId(2), panel([50.0, 50.0], ranks[1], 21));
    (world, [a, b], [a_sprite, b_sprite])
}

fn child(id: u32) -> LayoutChild {
    LayoutChild {
        item: Some(Ref::new(AssetId(id))),
        ..Default::default()
    }
}

fn panel(at: [f32; 2], rank: u32, holds: u32) -> Layout {
    Layout {
        movable: true,
        children: vec![child(holds)],
        rect: Some([at[0], at[1], 100.0, 100.0]),
        rank,
        ..Default::default()
    }
}

// The pointer at `(x, y)` in every screen's space.
fn at(x: f32, y: f32) -> impl Fn(Option<AssetId>) -> (f32, f32) {
    move |_| (x, y)
}

fn stack(world: &mut World) -> PanelStack {
    let mut stack = PanelStack::default();
    stack.rebuild(&world.context());
    stack
}

#[test]
fn a_panels_elements_take_its_rank_and_the_rest_none() {
    let (mut world, [a, b], [a_sprite, b_sprite]) = world([0, 0]);
    let loose = world.push(Sprite::default());
    let stack = stack(&mut world);
    assert_eq!(stack.rank_of(a), 1);
    assert_eq!(stack.rank_of(a_sprite), 1);
    assert_eq!(stack.rank_of(b), 2, "equal ranks keep the order added");
    assert_eq!(stack.rank_of(b_sprite), 2, "through a nested layout");
    assert_eq!(stack.rank_of(loose), 0);
}

#[test]
fn a_higher_rank_stacks_in_front() {
    let (mut world, [a, b], _) = world([5, 1]);
    let stack = stack(&mut world);
    assert!(stack.rank_of(a) > stack.rank_of(b));
    assert_eq!(stack.root_over(None, at(75.0, 75.0)), Some(a));
    assert_eq!(stack.raised_rank(b), Some(6));
    assert_eq!(stack.raised_rank(a), None, "already in front");
}

#[test]
fn only_the_frontmost_panel_under_the_pointer_is_reached() {
    let (mut world, [_, b], [a_sprite, b_sprite]) = world([0, 0]);
    let loose = world.push(Sprite::default());
    let stack = stack(&mut world);
    assert_eq!(stack.root_over(None, at(75.0, 75.0)), Some(b));
    assert!(stack.reaches(None, b_sprite, at(75.0, 75.0)));
    assert!(
        !stack.reaches(None, a_sprite, at(75.0, 75.0)),
        "covered by b"
    );
    assert!(!stack.reaches(None, loose, at(75.0, 75.0)));
    assert!(stack.reaches(None, a_sprite, at(10.0, 10.0)));
    assert!(
        stack.reaches(None, loose, at(400.0, 400.0)),
        "no panel there"
    );
    assert!(
        !stack.reaches(Some(AssetId(9)), a_sprite, at(75.0, 75.0)),
        "a panel covers a screen on its layer too"
    );
}

#[test]
fn a_hidden_or_fixed_root_is_no_panel() {
    let mut world = World::new();
    world.push(Layout {
        rect: Some([0.0, 0.0, 10.0, 10.0]),
        ..Default::default()
    });
    world.push(Layout {
        movable: true,
        visible: false,
        rect: Some([0.0, 0.0, 10.0, 10.0]),
        ..Default::default()
    });
    let stack = stack(&mut world);
    assert_eq!(stack.root_over(None, at(5.0, 5.0)), None);
    assert!(stack.panels.is_empty());
}

#[test]
fn a_hidden_layout_hides_everything_in_it() {
    let (mut world, [a, _], [a_sprite, b_sprite]) = world([0, 0]);
    world.get_mut::<Layout>(a).unwrap().visible = false;
    let stack = stack(&mut world);
    assert!(stack.hides(a_sprite));
    assert!(!stack.hides(b_sprite));
    assert!(
        !stack.reaches(None, a_sprite, at(10.0, 10.0)),
        "a hidden element takes no input"
    );
}

// A screen stack where screen 9 sits at `layer` and the HUD (no screen) at 0.
fn layered(world: &mut World, layer: i32) {
    world.insert_resource(ScreenStack {
        layers: alloc::collections::BTreeMap::from([(AssetId(9), layer)]),
        open: vec![AssetId(9)],
        ..Default::default()
    });
}

#[test]
fn a_panel_covers_lower_screens_but_not_higher_ones() {
    let (mut world, _, [a_sprite, _]) = world([0, 0]);
    let loose = world.push(Sprite::default());
    layered(&mut world, 40);
    let higher = stack(&mut world);
    assert!(
        higher.reaches(Some(AssetId(9)), loose, at(10.0, 10.0)),
        "an element on a higher screen is not covered"
    );
    assert!(!higher.reaches(None, loose, at(10.0, 10.0)));
    assert!(higher.reaches(None, a_sprite, at(10.0, 10.0)));
    layered(&mut world, -40);
    let lower = stack(&mut world);
    assert!(
        !lower.reaches(Some(AssetId(9)), loose, at(10.0, 10.0)),
        "an element on a lower screen is covered"
    );
}

#[test]
fn each_screen_is_tested_in_its_own_space() {
    let (mut world, _, [a_sprite, _]) = world([0, 0]);
    let loose = world.push(Sprite::default());
    let stack = stack(&mut world);
    // The panels sit in the HUD's space at (10, 10); the element's own screen
    // sees the pointer elsewhere.
    let split = |s: Option<AssetId>| {
        if s.is_none() {
            (10.0, 10.0)
        } else {
            (500.0, 500.0)
        }
    };
    assert!(!stack.reaches(Some(AssetId(9)), loose, split));
    assert!(stack.reaches(None, a_sprite, split));
}

#[test]
fn a_nested_movable_layout_is_no_panel() {
    let mut world = World::new();
    world.push_identified(
        AssetId(5),
        Layout {
            movable: true,
            rect: Some([0.0, 0.0, 10.0, 10.0]),
            ..Default::default()
        },
    );
    world.push(Layout {
        children: vec![child(5)],
        ..Default::default()
    });
    let stack = stack(&mut world);
    assert_eq!(stack.root_over(None, at(5.0, 5.0)), None);
}

// A modal root stacks in front of every panel whatever their ranks, covers
// the whole screen so nothing else is reached, and leaves a rank between
// itself and the panels behind for its dim.
#[test]
fn a_modal_root_takes_the_screen() {
    let (mut world, [a, b], [a_sprite, _]) = world([9, 3]);
    let held = world.push_identified(AssetId(30), Sprite::default());
    let dialog = world.push_identified(
        AssetId(3),
        Layout {
            modal: true,
            children: vec![child(30)],
            rect: Some([500.0, 500.0, 50.0, 50.0]),
            ..Default::default()
        },
    );
    let shown = stack(&mut world);
    assert_eq!(
        shown.root_over(None, at(10.0, 10.0)),
        Some(dialog),
        "everywhere"
    );
    assert!(!shown.reaches(None, a_sprite, at(10.0, 10.0)));
    assert!(shown.reaches(None, held, at(510.0, 510.0)));
    assert!(shown.shuts(a_sprite, None) && !shown.shuts(held, None));
    let dims: Vec<_> = shown.modals().collect();
    assert_eq!(dims.len(), 1);
    let (root, dim) = dims[0];
    assert_eq!(root, dialog);
    assert!(dim > shown.rank_of(a).max(shown.rank_of(b)));
    assert!(shown.rank_of(held) > dim, "the dim draws under the dialog");

    if let Some(l) = world.get_mut::<Layout>(dialog) {
        l.visible = false;
    }
    assert!(
        !stack(&mut world).shuts(a_sprite, None),
        "a hidden one shuts nothing"
    );
}

// Of two modal roots the front one shuts the other out, each dims under itself
// at its own rank, and a screen layered above them both is not shut.
#[test]
fn the_front_modal_root_shuts_the_rest_and_each_dims_alone() {
    let mut world = World::new();
    let modal = |holds: u32| Layout {
        modal: true,
        children: vec![child(holds)],
        rect: Some([0.0, 0.0, 10.0, 10.0]),
        ..Default::default()
    };
    let back_held = world.push_identified(AssetId(10), Sprite::default());
    let front_held = world.push_identified(AssetId(20), Sprite::default());
    let above = world.push_identified(AssetId(30), Sprite::default());
    world.push_identified(AssetId(1), modal(10));
    world.push_identified(AssetId(2), modal(20));
    world.insert_resource(ScreenStack {
        layers: alloc::collections::BTreeMap::from([(AssetId(99), 5)]),
        open: vec![AssetId(99)],
        ..Default::default()
    });
    let stack = stack(&mut world);
    assert!(stack.shuts(back_held, None), "the back one is shut out");
    assert!(!stack.shuts(front_held, None));
    assert!(!stack.shuts(above, Some(AssetId(99))), "a higher screen");
    let dims: Vec<u32> = stack.modals().map(|(_, rank)| rank).collect();
    assert_eq!(dims.len(), 2);
    assert!(dims[0] < stack.rank_of(back_held));
    assert!(
        stack.rank_of(back_held) < dims[1],
        "the front dim covers the back root"
    );
    assert!(dims[1] < stack.rank_of(front_held));
}

// A front root stays in front of every panel, however high their ranks run,
// covers them where it lies, and a press never raises a panel over it or
// raises it.
#[test]
fn a_front_root_stays_in_front_of_the_panels() {
    let (mut world, [a, b], _) = world([3, 900]);
    world.push_identified(AssetId(30), Sprite::default());
    let bar = world.push_identified(
        AssetId(3),
        Layout {
            front: true,
            children: vec![child(30)],
            rect: Some([0.0, 0.0, 400.0, 20.0]),
            ..Default::default()
        },
    );
    let stack = stack(&mut world);
    assert!(stack.rank_of(bar) > stack.rank_of(b));
    assert!(stack.rank_of(b) > stack.rank_of(a));
    assert_eq!(
        stack.root_over(None, at(60.0, 10.0)),
        Some(bar),
        "covers them"
    );
    assert_eq!(
        stack.raised_rank(a),
        Some(901),
        "raised over b, not the bar"
    );
    assert_eq!(stack.raised_rank(bar), None, "a press never raises it");
}
