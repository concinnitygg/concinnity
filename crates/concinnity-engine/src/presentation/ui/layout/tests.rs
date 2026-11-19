use super::*;
use alloc::vec;

// Layouts and element sizes by id, as the overlay hands them to the solver.
#[derive(Default)]
struct Src {
    layouts: HashMap<AssetId, Layout>,
    elements: HashMap<AssetId, [f32; 2]>,
}

impl LayoutSource for Src {
    fn layout(&self, id: AssetId) -> Option<&Layout> {
        self.layouts.get(&id)
    }
    fn element(&self, id: AssetId) -> Option<[f32; 2]> {
        self.elements.get(&id).copied()
    }
}

fn child(id: u32, width: LayoutSize, height: LayoutSize) -> LayoutChild {
    LayoutChild {
        item: Some(Ref::new(AssetId(id))),
        width,
        height,
    }
}

fn fit(id: u32) -> LayoutChild {
    LayoutChild::fit(AssetId(id))
}

fn solve(src: &Src, root: &Layout, area: [f32; 2]) -> HashMap<u32, [f32; 4]> {
    let mut solve = LayoutSolve::default();
    solve.solve_root(src, None, root, area);
    solve.placed.iter().map(|&(id, r)| (id.0, r)).collect()
}

fn elements(sizes: &[(u32, [f32; 2])]) -> Src {
    Src {
        elements: sizes.iter().map(|&(id, s)| (AssetId(id), s)).collect(),
        ..Src::default()
    }
}

#[test]
fn a_column_stacks_its_children_with_the_gap_from_the_margin() {
    let src = elements(&[(1, [40.0, 10.0]), (2, [20.0, 30.0])]);
    let root = Layout {
        children: vec![fit(1), fit(2)],
        gap: 5.0,
        margin: [10.0, 20.0],
        ..Layout::default()
    };
    let placed = solve(&src, &root, [500.0, 500.0]);
    assert_eq!(placed[&1], [10.0, 20.0, 40.0, 10.0]);
    assert_eq!(placed[&2], [10.0, 35.0, 20.0, 30.0]);
}

#[test]
fn an_anchor_places_a_root_against_its_edges() {
    let src = elements(&[(1, [40.0, 10.0])]);
    let root = |anchor| Layout {
        children: vec![fit(1)],
        anchor,
        margin: [10.0, 10.0],
        ..Layout::default()
    };
    let at = |anchor| solve(&src, &root(anchor), [200.0, 100.0])[&1];
    assert_eq!(at(LayoutAnchor::TopRight), [150.0, 10.0, 40.0, 10.0]);
    assert_eq!(at(LayoutAnchor::Center), [80.0, 45.0, 40.0, 10.0]);
    assert_eq!(at(LayoutAnchor::BottomLeft), [10.0, 80.0, 40.0, 10.0]);
}

#[test]
fn fills_share_the_space_their_siblings_leave_by_weight() {
    let src = elements(&[(1, [40.0, 10.0]), (2, [0.0, 10.0]), (3, [0.0, 10.0])]);
    let root = Layout {
        direction: LayoutDirection::Row,
        children: vec![
            fit(1),
            child(2, LayoutSize::Fill(1.0), LayoutSize::Fit),
            child(3, LayoutSize::Fill(3.0), LayoutSize::Fill(1.0)),
        ],
        width: LayoutSize::Fixed(140.0),
        height: LayoutSize::Fixed(30.0),
        ..Layout::default()
    };
    let placed = solve(&src, &root, [500.0, 500.0]);
    assert_eq!(placed[&1], [0.0, 0.0, 40.0, 10.0]);
    assert_eq!(placed[&2], [40.0, 0.0, 25.0, 10.0]);
    assert_eq!(placed[&3], [65.0, 0.0, 75.0, 30.0], "fills across too");
}

#[test]
fn justify_and_align_place_children_in_the_space_over() {
    let src = elements(&[(1, [10.0, 10.0]), (2, [10.0, 20.0])]);
    let row = |justify, align| Layout {
        direction: LayoutDirection::Row,
        children: vec![fit(1), fit(2)],
        justify,
        align,
        width: LayoutSize::Fixed(100.0),
        ..Layout::default()
    };
    let placed = solve(
        &src,
        &row(LayoutAlign::End, LayoutAlign::Center),
        [500.0, 500.0],
    );
    assert_eq!(placed[&1], [80.0, 5.0, 10.0, 10.0]);
    assert_eq!(placed[&2], [90.0, 0.0, 10.0, 20.0]);
    let spread = solve(
        &src,
        &row(LayoutAlign::SpaceBetween, LayoutAlign::End),
        [500.0, 500.0],
    );
    assert_eq!(spread[&1], [0.0, 10.0, 10.0, 10.0]);
    assert_eq!(spread[&2], [90.0, 0.0, 10.0, 20.0]);
}

#[test]
fn absent_children_take_no_space_or_gap() {
    // Element 2 is absent (hidden or empty), so 3 closes up against 1.
    let src = elements(&[(1, [10.0, 10.0]), (3, [10.0, 10.0])]);
    let root = Layout {
        children: vec![fit(1), fit(2), fit(3)],
        gap: 4.0,
        ..Layout::default()
    };
    let placed = solve(&src, &root, [500.0, 500.0]);
    assert!(!placed.contains_key(&2));
    assert_eq!(placed[&3], [0.0, 14.0, 10.0, 10.0]);
}

#[test]
fn nested_layouts_size_to_their_content_with_padding() {
    let mut src = elements(&[
        (1, [10.0, 10.0]),
        (2, [30.0, 10.0]),
        (3, [5.0, 50.0]),
        (4, [8.0, 8.0]),
    ]);
    src.layouts.insert(
        AssetId(10),
        Layout {
            direction: LayoutDirection::Row,
            children: vec![fit(1), fit(2)],
            gap: 2.0,
            padding: 3.0,
            ..Layout::default()
        },
    );
    src.layouts.insert(
        AssetId(11),
        Layout {
            visible: false,
            children: vec![fit(3)],
            ..Layout::default()
        },
    );
    let root = Layout {
        children: vec![fit(10), fit(11), fit(4)],
        ..Layout::default()
    };
    let placed = solve(&src, &root, [500.0, 500.0]);
    assert_eq!(placed[&1], [3.0, 3.0, 10.0, 10.0]);
    assert_eq!(placed[&2], [15.0, 3.0, 30.0, 10.0]);
    assert!(!placed.contains_key(&3), "a hidden layout places nothing");
    // The row is 16 tall with its padding, and the hidden layout takes nothing.
    assert_eq!(placed[&4], [0.0, 16.0, 8.0, 8.0]);
}

#[test]
fn a_stack_places_each_child_in_the_whole_layout() {
    let src = elements(&[(1, [10.0, 10.0]), (2, [0.0, 0.0])]);
    let root = Layout {
        direction: LayoutDirection::Stack,
        children: vec![
            fit(1),
            child(2, LayoutSize::Fill(1.0), LayoutSize::Fill(1.0)),
        ],
        justify: LayoutAlign::Center,
        align: LayoutAlign::End,
        width: LayoutSize::Fixed(50.0),
        height: LayoutSize::Fixed(40.0),
        ..Layout::default()
    };
    let placed = solve(&src, &root, [500.0, 500.0]);
    assert_eq!(placed[&1], [20.0, 30.0, 10.0, 10.0]);
    assert_eq!(placed[&2], [0.0, 0.0, 50.0, 40.0]);
}

#[test]
fn a_layout_holding_itself_is_sized_and_placed_once() {
    let mut src = elements(&[(1, [10.0, 10.0])]);
    let looped = Layout {
        children: vec![fit(10), fit(10), fit(1)],
        ..Layout::default()
    };
    src.layouts.insert(AssetId(10), looped.clone());
    let mut solve = LayoutSolve::default();
    solve.solve_root(&src, Some(AssetId(10)), &looped, [500.0, 500.0]);
    // The self-references take no space, and the label is placed once.
    assert_eq!(solve.placed, vec![(AssetId(1), [0.0, 0.0, 10.0, 10.0])]);

    // Held by another root, the loop is placed through it the same way.
    let root = Layout {
        children: vec![fit(10)],
        ..Layout::default()
    };
    let placed = solve_list(&src, &root);
    assert_eq!(placed, vec![(AssetId(1), [0.0, 0.0, 10.0, 10.0])]);
}

fn solve_list(src: &Src, root: &Layout) -> Vec<(AssetId, [f32; 4])> {
    let mut solve = LayoutSolve::default();
    solve.solve_root(src, None, root, [500.0, 500.0]);
    solve.placed
}

#[test]
fn a_root_with_nothing_to_show_places_nothing() {
    let src = Src::default();
    let root = Layout {
        children: vec![fit(1)],
        ..Layout::default()
    };
    assert!(solve(&src, &root, [500.0, 500.0]).is_empty());
}

#[test]
fn sizes_read_as_fit_fixed_or_fill() {
    let read = |json: &str| serde_json::from_str::<LayoutSize>(json).unwrap();
    assert_eq!(read(r#""fit""#), LayoutSize::Fit);
    assert_eq!(read(r#"{"fixed": 120}"#), LayoutSize::Fixed(120.0));
    assert_eq!(read(r#"{"fill": 2}"#), LayoutSize::Fill(2.0));
}

// A scrolling column takes the height it is given, lays its children out at
// their full height moved up by its scroll and short of its scrollbar, and
// reports how tall they are.
#[test]
fn a_scrolling_column_moves_its_children_and_asks_no_height() {
    let mut src = elements(&[(1, [40.0, 30.0]), (2, [40.0, 30.0]), (3, [40.0, 30.0])]);
    src.layouts.insert(
        AssetId(10),
        Layout {
            children: vec![fit(1), fit(2), fit(3)],
            scroll: true,
            scrolled: 25.0,
            ..Layout::default()
        },
    );
    let root = Layout {
        children: vec![child(10, LayoutSize::Fixed(100.0), LayoutSize::Fixed(50.0))],
        ..Layout::default()
    };
    let mut solve = LayoutSolve::default();
    let rect = solve.solve_root(&src, None, &root, [500.0, 500.0]);
    assert_eq!(
        rect,
        Some([0.0, 0.0, 100.0, 50.0]),
        "sized by its fixed child"
    );
    let placed: HashMap<u32, [f32; 4]> = solve.placed.iter().map(|&(id, r)| (id.0, r)).collect();
    assert_eq!(placed[&1][1], -25.0, "scrolled up");
    assert_eq!(placed[&3][1], 35.0);
    assert_eq!(
        solve.scrolls,
        vec![(AssetId(10), [0.0, 0.0, 100.0, 50.0], 90.0)]
    );

    // Its parent never grows to its children's height.
    let fitted = Layout {
        children: vec![fit(10)],
        ..Layout::default()
    };
    let rect = LayoutSolve::default().solve_root(&src, None, &fitted, [500.0, 500.0]);
    assert_eq!(rect.map(|r| r[3]), Some(0.0));
}

// A resizable root laid out on screen says which of its edges a press would
// size; a fixed one, a hidden one, or a press in its body says none.
#[test]
fn a_resizable_root_reports_the_edge_a_press_would_size() {
    let mut root = Layout {
        resizable: true,
        rect: Some([100.0, 100.0, 400.0, 300.0]),
        ..Layout::default()
    };
    let edges = |right, bottom| Some(ResizeEdges { right, bottom });
    assert_eq!(root.resize_grip(498.0, 200.0), edges(true, false));
    assert_eq!(root.resize_grip(300.0, 398.0), edges(false, true));
    assert_eq!(root.resize_grip(498.0, 398.0), edges(true, true));
    assert_eq!(root.resize_grip(300.0, 200.0), None, "the body");
    assert_eq!(root.resize_grip(600.0, 200.0), None, "outside it");
    root.resizable = false;
    assert_eq!(root.resize_grip(498.0, 200.0), None);
    root.resizable = true;
    root.visible = false;
    assert_eq!(root.resize_grip(498.0, 200.0), None);
}
