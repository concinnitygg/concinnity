use super::*;
use alloc::vec::Vec;

fn tile(level: u8, x: u32, y: u32) -> TileId {
    TileId {
        face: CubeFace(2),
        level,
        x,
        y,
    }
}

// A page whose height change is `e(fs, ft)` across it and whose weight is 1.
fn page(e: impl Fn(f64, f64) -> f64) -> EditPage {
    let n = PAGE_CELLS as f64;
    let es: Vec<f64> = (0..PAGE_SAMPLES)
        .map(|k| e((k % PAGE_SIDE) as f64 / n, (k / PAGE_SIDE) as f64 / n))
        .collect();
    EditPage::from_samples(&es, &[1.0; PAGE_SAMPLES]).unwrap()
}

fn flat(e: f64) -> EditPage {
    page(|_, _| e)
}

// The face coordinates of `(fs, ft)` across `t`.
fn face_at(t: TileId, fs: f64, ft: f64) -> (f64, f64) {
    let (s0, t0, span) = t.span();
    (s0 + fs * span, t0 + ft * span)
}

// A point reads the deepest page over it, never a sum of levels; a point no
// page lies over reads nothing.
#[test]
fn the_deepest_page_wins() {
    let coarse = tile(4, 7, 9);
    let fine = coarse.children()[3].children()[0];
    let pages = EditPages::new([(coarse, flat(2.0)), (fine, flat(-5.0))]);
    let read = |t: TileId, fs: f64, ft: f64| {
        let (s, u) = face_at(t, fs, ft);
        pages.at(CubeFace(2), s, u).map(|h| (h.tile, h.sample.e))
    };
    let (_, e) = read(fine, 0.5, 0.5).unwrap();
    assert!((e + 5.0).abs() < 1e-9, "{e}");
    assert_eq!(read(fine, 0.5, 0.5).unwrap().0, fine);
    let (at, e) = read(coarse, 0.1, 0.1).unwrap();
    assert_eq!(at, coarse);
    assert!((e - 2.0).abs() < 1e-9);
    assert!(read(tile(4, 2, 2), 0.5, 0.5).is_none());
    assert_eq!(pages.covering(fine.children()[1]).unwrap().0, fine);
    assert_eq!(pages.covering(coarse.children()[0]).unwrap().0, coarse);
    assert!(pages.covering(TileId::root(CubeFace(3))).is_none());
}

// The edits over one tile read what the planet-wide lookup reads, at every
// point within the tile, whatever pages lie above or within it. (On a tile's
// edge the tile reads its own pages, the planet-wide lookup its neighbor's.)
#[test]
fn a_tile_s_edits_read_as_the_planet_s() {
    let coarse = tile(3, 4, 3);
    let mid = coarse.children()[2];
    let fine = mid.children()[1].children()[3];
    let pages = EditPages::new([
        (coarse, page(|s, t| 4.0 * s - t)),
        (fine, page(|s, t| s * t * 9.0)),
    ]);
    for over in [coarse, mid, mid.children()[1], fine, fine.children()[0]] {
        let view = pages.over(over);
        assert!(!view.is_empty());
        for (fs, ft) in [(0.001, 0.001), (0.33, 0.71), (0.999, 0.999), (0.5, 0.02)] {
            let (s, t) = face_at(over, fs, ft);
            assert_eq!(
                view.at(s, t),
                pages.at(CubeFace(2), s, t),
                "{over:?} {fs} {ft}"
            );
        }
    }
    assert!(pages.over(tile(3, 0, 0)).is_empty());
}

// Two pages refined from one coarse page meet along their shared side, and
// each meets the coarse page around it: the ground is continuous across
// every page edge.
#[test]
fn heights_are_continuous_across_page_edges() {
    let coarse = tile(6, 20, 41);
    let pages = EditPages::new([(coarse, page(|s, t| 10.0 * s * s - 6.0 * t + 3.0 * s * t))]);
    let [a, b, ..] = coarse.children();
    let mut refined = pages.clone();
    refined.insert(a, Arc::new(pages.refined(a)));
    refined.insert(b, Arc::new(pages.refined(b)));
    let read = |set: &EditPages, s: f64, t: f64| set.at(CubeFace(2), s, t).unwrap().sample.e;
    for k in 0..=64 {
        let ft = f64::from(k) / 64.0;
        // The side `a` and `b` share, read from each.
        let (s, t) = face_at(a, 1.0, ft);
        let from_a = refined.over(a).at(s, t).unwrap().sample.e;
        let from_b = refined.over(b).at(s, t).unwrap().sample.e;
        assert!((from_a - from_b).abs() < 1e-3, "{ft}: {from_a} {from_b}");
        // The side `a` shares with the coarse page's unrefined quarter.
        let (s, t) = face_at(a, ft, 1.0);
        let below = face_at(a, ft, 1.0 + 1e-9);
        let inside = read(&refined, s, t - 1e-12);
        let outside = read(&refined, below.0, below.1);
        assert!((inside - outside).abs() < 1e-3, "{ft}: {inside} {outside}");
        // Refining changed nothing anywhere.
        assert!((read(&refined, s, t) - read(&pages, s, t)).abs() < 1e-3);
    }
}

#[test]
fn refining_where_no_page_lies_changes_nothing() {
    let pages = EditPages::default();
    let fine = tile(9, 3, 4);
    assert_eq!(pages.refined(fine), EditPage::neutral());
    let coarse = EditPages::new([(tile(2, 0, 1), flat(1.5))]);
    let under = tile(2, 0, 1).children()[2];
    let r = coarse.refined(under);
    assert!((r.e_at(7, 30) - 1.5).abs() < 1e-6);
}

// The figures bound every page; inserting only ever widens them.
#[test]
fn the_figures_bound_every_page() {
    let mut pages = EditPages::new([
        (tile(5, 1, 1), page(|s, _| 8.0 * s)),
        (tile(7, 0, 0), page(|_, t| -3.0 * t)),
    ]);
    assert!((pages.highest_change() - 8.0).abs() < 1e-3);
    assert!((pages.lowest_change() + 3.0).abs() < 1e-3);
    let [e5, _] = pages.level_steps(5);
    assert!((e5 - 8.0 / 32.0).abs() < 1e-3);
    assert_eq!(pages.level_steps(6), [0.0, 0.0]);
    pages.insert(tile(5, 1, 1), Arc::new(flat(0.0)));
    assert!((pages.highest_change() - 8.0).abs() < 1e-3, "widened only");
    assert!(!pages.insert(tile(2, 9, 0), Arc::new(flat(1.0))));
    assert_eq!(pages.len(), 2);
}

#[test]
fn a_child_is_found_under_any_point_of_its_parent() {
    let t = tile(5, 6, 30);
    for (fs, ft, k) in [(0.1, 0.1, 0), (0.9, 0.1, 1), (0.1, 0.9, 2), (0.9, 0.9, 3)] {
        let (s, u) = face_at(t, fs, ft);
        assert_eq!(child_at(t, s, u), t.children()[k]);
    }
    // A point past the tile's edge reads its nearest child.
    let (s, u) = face_at(t, 1.2, -0.3);
    assert_eq!(child_at(t, s, u), t.children()[1]);
}
