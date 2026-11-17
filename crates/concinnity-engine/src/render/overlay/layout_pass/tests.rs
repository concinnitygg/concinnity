use super::*;
use crate::components::{
    LayoutAnchor, LayoutChild, LayoutDirection, LayoutSize, Screen, ScreenSpace, TextAlign,
};
use crate::ecs::{Ref, World};
use crate::render::overlay::test_fonts::{FONT, loaded_fonts};
use alloc::string::ToString;
use alloc::vec;

// An unpadded chip in the test font: 10px a character wide, 12px tall, its
// text origin 2px below the top of its box.
fn chip(content: &str) -> TextLabel {
    TextLabel {
        font: Some(FONT),
        content: content.to_string(),
        ..Default::default()
    }
}

fn run(world: &mut World, viewport: [f32; 2]) {
    let mut scratch = LayoutScratch::default();
    apply_layouts(
        &mut world.context(),
        &loaded_fonts(),
        viewport,
        &mut scratch,
    );
}

fn label_at(world: &World, id: u32) -> (f32, f32) {
    let l = world.get_by_id::<TextLabel>(AssetId(id)).unwrap();
    (l.x, l.y)
}

// A column anchored top-right stacks its labels flush right, measured at their
// live text, in window pixels.
#[test]
fn a_root_column_places_labels_by_their_measured_text() {
    let mut world = World::new();
    world.push_identified(AssetId(1), chip("ab"));
    world.push_identified(AssetId(2), chip("abcd"));
    world.push(Layout {
        children: vec![LayoutChild::fit(AssetId(1)), LayoutChild::fit(AssetId(2))],
        gap: 6.0,
        align: crate::components::LayoutAlign::End,
        anchor: LayoutAnchor::TopRight,
        margin: [10.0, 10.0],
        ..Layout::default()
    });
    run(&mut world, [400.0, 300.0]);
    let (x1, y1) = label_at(&world, 1);
    let (x2, y2) = label_at(&world, 2);
    // The widest chip ends at 390; the narrower one is flush right with it.
    assert_eq!(x2 + 40.0, 390.0);
    assert_eq!(x1 + 20.0, 390.0);
    assert_eq!(y2 - y1, 12.0 + 6.0);
}

// A sprite takes the whole rect the layout gives it; a canvas screen's root is
// placed on the reference canvas whatever the window.
#[test]
fn a_canvas_root_sizes_a_filling_sprite_on_the_reference_canvas() {
    let mut world = World::new();
    let screen = AssetId(9);
    world.push_identified(
        AssetId(1),
        Sprite {
            visible: true,
            ..Default::default()
        },
    );
    world.push(Layout {
        screen: Some(Ref::new(screen)),
        direction: LayoutDirection::Row,
        children: vec![LayoutChild {
            item: Some(Ref::new(AssetId(1))),
            width: LayoutSize::Fill(1.0),
            height: LayoutSize::Fixed(20.0),
        }],
        width: LayoutSize::Fill(1.0),
        margin: [40.0, 0.0],
        anchor: LayoutAnchor::Bottom,
        ..Layout::default()
    });
    run(&mut world, [3000.0, 2000.0]);
    let s = world.get_by_id::<Sprite>(AssetId(1)).unwrap();
    assert_eq!((s.x, s.y, s.width, s.height), (40.0, 700.0, 1200.0, 20.0));

    // On a window-space screen the same root spans the window.
    let mut world = World::new();
    world.push_identified(
        screen,
        Screen {
            space: ScreenSpace::Window,
            ..Default::default()
        },
    );
    world.insert_resource(ScreenSpaces {
        window: vec![screen],
    });
    world.push_identified(AssetId(1), Sprite::default());
    world.push(Layout {
        screen: Some(Ref::new(screen)),
        children: vec![LayoutChild {
            item: Some(Ref::new(AssetId(1))),
            width: LayoutSize::Fill(1.0),
            height: LayoutSize::Fixed(20.0),
        }],
        width: LayoutSize::Fill(1.0),
        ..Layout::default()
    });
    run(&mut world, [3000.0, 2000.0]);
    let s = world.get_by_id::<Sprite>(AssetId(1)).unwrap();
    assert_eq!((s.x, s.width), (0.0, 3000.0));
}

// A layout another holds is placed by its parent, not as a root of its own.
#[test]
fn a_nested_layout_is_placed_by_its_parent() {
    let mut world = World::new();
    world.push_identified(AssetId(1), chip("ab"));
    world.push_identified(AssetId(2), chip("abc"));
    world.push_identified(
        AssetId(10),
        Layout {
            direction: LayoutDirection::Row,
            children: vec![LayoutChild::fit(AssetId(1)), LayoutChild::fit(AssetId(2))],
            gap: 4.0,
            ..Layout::default()
        },
    );
    world.push(Layout {
        children: vec![LayoutChild::fit(AssetId(10))],
        margin: [10.0, 10.0],
        ..Layout::default()
    });
    run(&mut world, [400.0, 300.0]);
    let (x1, y1) = label_at(&world, 1);
    let (x2, y2) = label_at(&world, 2);
    assert_eq!(x1, 10.0);
    assert_eq!(x2, 10.0 + 20.0 + 4.0);
    assert_eq!(y1, y2);
}

// A label with no text takes no space, so its neighbor closes up.
#[test]
fn an_empty_label_takes_no_space() {
    let mut world = World::new();
    world.push_identified(AssetId(1), chip(""));
    world.push_identified(AssetId(2), chip("ab"));
    world.push(Layout {
        direction: LayoutDirection::Row,
        children: vec![LayoutChild::fit(AssetId(1)), LayoutChild::fit(AssetId(2))],
        gap: 4.0,
        ..Layout::default()
    });
    run(&mut world, [400.0, 300.0]);
    assert_eq!(label_at(&world, 2).0, 0.0);
}

// A label anchors by its alignment: a centered one at its box's middle, a
// right-aligned one at its right edge, so its text lands in the same box.
#[test]
fn an_aligned_label_is_anchored_where_its_alignment_reads() {
    let mut world = World::new();
    let aligned = |content: &str, align| TextLabel {
        align,
        ..chip(content)
    };
    world.push_identified(AssetId(1), aligned("ab", TextAlign::Center));
    world.push_identified(AssetId(2), aligned("ab", TextAlign::Right));
    world.push(Layout {
        direction: LayoutDirection::Row,
        children: vec![LayoutChild::fit(AssetId(1)), LayoutChild::fit(AssetId(2))],
        gap: 4.0,
        margin: [10.0, 10.0],
        ..Layout::default()
    });
    run(&mut world, [400.0, 300.0]);
    assert_eq!(label_at(&world, 1).0, 10.0 + 10.0);
    assert_eq!(label_at(&world, 2).0, 10.0 + 20.0 + 4.0 + 20.0);
}
