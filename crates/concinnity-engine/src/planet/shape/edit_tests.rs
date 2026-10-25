// The ground with edits stored over it: as generated where none lie, the
// edits' height change and weighted hills where they do, and every bound the
// planet keeps widened to hold them.

use super::*;
use crate::planet::coast::fibonacci_direction;
use crate::planet::{EditPage, PAGE_SAMPLES, PAGE_SIDE, TILE_CELLS, tile_bounds, tile_mesh};
use alloc::vec::Vec;

fn generated() -> PlanetShape {
    PlanetShape {
        center: [0.0, -50_000.0, 0.0],
        radius: 50_000.0,
        amplitude: 40.0,
        feature_size: 2_000.0,
        octaves: 8,
        seed: 7,
        continents: Continents::NONE,
        edits: None,
    }
}

fn coastal() -> PlanetShape {
    PlanetShape {
        continents: Continents::new(50_000.0, 7, 20_000.0, 0.4, 60.0, 60.0),
        ..generated()
    }
}

fn page_tile() -> TileId {
    TileId {
        face: CubeFace(2),
        level: 8,
        x: 128,
        y: 127,
    }
}

// A page of height change `e(fs, ft)` and weight `w(fs, ft)` across it.
fn page(e: impl Fn(f64, f64) -> f64, w: impl Fn(f64, f64) -> f64) -> EditPage {
    let at = |k: usize| ((k % PAGE_SIDE) as f64 / 32.0, (k / PAGE_SIDE) as f64 / 32.0);
    let es: Vec<f64> = (0..PAGE_SAMPLES).map(|k| e(at(k).0, at(k).1)).collect();
    let ws: Vec<f64> = (0..PAGE_SAMPLES).map(|k| w(at(k).0, at(k).1)).collect();
    EditPage::from_samples(&es, &ws).unwrap()
}

// A raised ring and a flattened, smoothed disc on one page.
fn ring_and_disc() -> EditPage {
    let ring = |s: f64, t: f64| {
        let r = libm::sqrt((s - 0.5) * (s - 0.5) + (t - 0.5) * (t - 0.5));
        8.0 * libm::exp(-((r - 0.3) * (r - 0.3)) / 0.005)
    };
    let disc = |s: f64, t: f64| {
        let r = libm::sqrt((s - 0.5) * (s - 0.5) + (t - 0.5) * (t - 0.5));
        (r / 0.2).clamp(0.0, 1.0)
    };
    page(ring, disc)
}

fn edited(shape: &PlanetShape, pages: impl IntoIterator<Item = (TileId, EditPage)>) -> PlanetShape {
    shape.with_edits(Some(Arc::new(EditPages::new(pages))))
}

// Directions within the page's tile, an `n` by `n` grid of them, every one
// off the page's sample lines.
fn over_page(n: u32) -> impl Iterator<Item = DVec3> {
    let t = page_tile();
    let at = move |k: u32| (f64::from(k) + 0.37) / f64::from(n);
    (0..n).flat_map(move |j| (0..n).map(move |i| t.direction(at(i), at(j))))
}

// With no edits, or none over it, the ground is the generated ground bit for
// bit: heights, normals, tile meshes and bounds, and every bound the planet
// keeps. A tile's corners carry full detail and no correction.
#[test]
fn ground_without_edits_over_it_is_unchanged() {
    for base in [generated(), coastal()] {
        let empty = base.with_edits(Some(Arc::new(EditPages::default())));
        let elsewhere = edited(
            &base,
            [(TileId::root(CubeFace(3)), page(|_, _| 50.0, |_, _| 0.0))],
        );
        for other in [&empty, &elsewhere] {
            for i in 0..500 {
                let dir = fibonacci_direction(i, 500);
                if dir[1] < -0.5 {
                    continue;
                }
                assert_eq!(base.height(dir).to_bits(), other.height(dir).to_bits());
                assert_eq!(base.surface_and_normal(dir), other.surface_and_normal(dir));
            }
            for t in [page_tile(), TileId::root(CubeFace(0))] {
                let (a, b) = (tile_mesh(&base, t), tile_mesh(other, t));
                assert_eq!(a.origin, b.origin);
                for (va, vb) in a.vertices.iter().zip(&b.vertices) {
                    assert_eq!(
                        (va.pos, va.normal, va.tangent),
                        (vb.pos, vb.normal, vb.tangent)
                    );
                    assert_eq!(va.color, [1.0, 0.0, 0.0]);
                    assert_eq!(vb.color, [1.0, 0.0, 0.0]);
                }
            }
        }
        for t in [page_tile(), TileId::root(CubeFace(0))] {
            assert_eq!(tile_bounds(&base, t), tile_bounds(&empty, t));
        }
        assert_eq!(base.max_slope(), empty.max_slope());
        assert_eq!(base.lowest_height(), empty.lowest_height());
        assert_eq!(base.highest_height(), empty.highest_height());
    }
}

// Where a page lies, its height change adds to the hills its weight scales:
// at weight 0 the ground is the coast's smooth profile plus the change.
#[test]
fn a_page_changes_the_height_by_its_change_and_weight() {
    for base in [generated(), coastal()] {
        let raised = edited(&base, [(page_tile(), page(|_, _| 5.0, |_, _| 1.0))]);
        let smooth = edited(&base, [(page_tile(), page(|_, _| -2.0, |_, _| 0.0))]);
        for dir in over_page(9) {
            let lift = raised.height(dir) - base.height(dir);
            assert!((lift - 5.0).abs() < 1e-9, "{lift}");
            let bare = if base.continents.divides() {
                base.coast_height(base.continents.inland(base.radius, dir).0)
            } else {
                0.0
            };
            assert!((smooth.height(dir) - (bare - 2.0)).abs() < 1e-9);
        }
    }
}

// The gradient the edited ground reports is the height's own slope, inside
// pages and across a planet of seas.
#[test]
fn the_edited_gradient_matches_the_height_s_slope() {
    for base in [generated(), coastal()] {
        let s = edited(&base, [(page_tile(), ring_and_disc())]);
        let e = 1e-8;
        // Off the sample lines, where a bilinear cell's slope is its own.
        for dir in over_page(11) {
            let (_, g) = s.height_gradient(dir);
            for k in 0..2 {
                let axis = edit_frame_axis(dir, k);
                let at = |m: f64| {
                    s.height(dvec3::normalize_or(
                        dvec3::add(dir, dvec3::scale(axis, m)),
                        dir,
                    ))
                };
                let slope = (at(e) - at(-e)) / (2.0 * e);
                let want = dvec3::dot(g, axis);
                assert!(
                    (slope - want).abs() < 1e-3 * (1.0 + want.abs()),
                    "{slope} {want}"
                );
            }
        }
    }
}

// The edits widen the lowest and highest ground and the steepest slope by
// as much as they reach: no two nearby points differ by more than the
// slope's bound, and every tile's bounds hold its edited corners.
#[test]
fn the_edits_widen_the_bounds_and_the_slope() {
    let base = generated();
    let deep = page_tile().children()[1].children()[2].children()[0];
    // A steep bump on the deep page, refined from the coarse one so the two
    // meet along its edges.
    let coarse = EditPages::new([(page_tile(), ring_and_disc())]);
    let refined = coarse.refined(deep);
    let bump = |i: usize, j: usize| {
        let (s, t) = (i as f64 / 32.0, j as f64 / 32.0);
        let lift = 30.0
            * libm::sin(core::f64::consts::PI * s)
            * libm::sin(core::f64::consts::PI * t)
            * libm::cos(core::f64::consts::TAU * s);
        let edges = libm::sin(core::f64::consts::PI * s) * libm::sin(core::f64::consts::PI * t);
        (
            refined.e_at(i, j) + lift - 12.0 * s * (1.0 - s) * edges,
            refined.w_at(i, j) * (1.0 - 0.5 * edges),
        )
    };
    let samples: Vec<(f64, f64)> = (0..PAGE_SAMPLES)
        .map(|k| bump(k % PAGE_SIDE, k / PAGE_SIDE))
        .collect();
    let e: Vec<f64> = samples.iter().map(|p| p.0).collect();
    let w: Vec<f64> = samples.iter().map(|p| p.1).collect();
    let pages = [
        (page_tile(), ring_and_disc()),
        (deep, EditPage::from_samples(&e, &w).unwrap()),
    ];
    let s = edited(&base, pages);
    assert!(s.highest_height() > base.highest_height() + 5.0);
    assert!(s.lowest_height() < base.lowest_height() - 20.0);
    let bound = s.max_slope();
    assert!(bound > base.max_slope());
    let mut steepest: f64 = 0.0;
    let face = CubeFace(2);
    let (s0, t0, span) = deep.span();
    for i in 0..3_000 {
        let fs = s0 + span * (f64::from(i % 60) / 50.0 - 0.1);
        let ft = t0 + span * (f64::from(i / 60) / 40.0 - 0.1);
        let a = face.direction(fs, ft);
        let b = face.direction(fs + 3e-7, ft - 2e-7);
        let run = 50_000.0 * libm::acos(dvec3::dot(a, b).min(1.0));
        steepest = steepest.max((s.height(a) - s.height(b)).abs() / run);
        let h = s.height(a);
        assert!(h >= s.lowest_height() && h <= s.highest_height());
    }
    assert!(steepest > base.max_slope(), "{steepest}");
    assert!(steepest <= bound, "{steepest} > {bound}");
    let side = TILE_CELLS as usize + 1;
    for t in [
        page_tile(),
        deep.parent().unwrap(),
        deep,
        deep.children()[3],
    ] {
        let b = tile_bounds(&s, t);
        let mesh = tile_mesh(&s, t);
        for v in &mesh.vertices[..side * side] {
            let p = dvec3::add(mesh.origin, dvec3::from_f32(v.pos));
            let gap = dvec3::length(dvec3::sub(p, b.center)) - b.radius;
            assert!(gap <= 1e-3 * (1.0 + b.radius * 1e-4), "{t:?} {gap}");
        }
    }
}

// The normals of an edited tile are the exact ones of its ground. The page
// is linear across, so its bilinear cells meet without a crease and the
// normal at a corner on a sample line is the same from either side.
#[test]
fn edited_tile_normals_are_exact() {
    let linear = page(|s, t| 6.0 * s - 3.0 * t, |s, _| 0.3 + 0.5 * s);
    let s = edited(&coastal(), [(page_tile(), linear)]);
    let t = page_tile().children()[3];
    let mesh = tile_mesh(&s, t);
    let n = TILE_CELLS as usize;
    for (k, v) in mesh.vertices.iter().enumerate() {
        let dir = t.direction(
            (k % (n + 1)) as f64 / n as f64,
            (k / (n + 1)) as f64 / n as f64,
        );
        let want = dvec3::to_f32(s.normal(dir));
        assert!(concinnity_core::math::vec3::dot(v.normal, want) > 1.0 - 1e-6);
    }
}

// The cube face a direction passes through gives it back, and its tangents
// are the direction's own rates along the face.
#[test]
fn face_coordinates_and_tangents_invert_and_differentiate_the_face() {
    for face in CubeFace::ALL {
        for (s, t) in [(0.0, 0.0), (0.7, -0.2), (-0.95, 0.95), (0.3, 0.999)] {
            let dir = face.direction(s, t);
            let (f, gs, gt) = CubeFace::of_direction(dir);
            assert_eq!(f, face);
            assert!(
                (gs - s).abs() < 1e-12 && (gt - t).abs() < 1e-12,
                "{s} {t}: {gs} {gt}"
            );
            let h = 1e-6;
            let [ds, dt] = face.tangents(s, t);
            let fs = dvec3::scale(
                dvec3::sub(face.direction(s + h, t), face.direction(s - h, t)),
                0.5 / h,
            );
            let ft = dvec3::scale(
                dvec3::sub(face.direction(s, t + h), face.direction(s, t - h)),
                0.5 / h,
            );
            assert!(dvec3::length(dvec3::sub(ds, fs)) < 1e-7);
            assert!(dvec3::length(dvec3::sub(dt, ft)) < 1e-7);
            assert!(smallest_singular([ds, dt]) >= min_face_rate());
        }
    }
}

// A ray meets the edited ground: a raised page lifts where it lands.
#[test]
fn a_ray_meets_the_edited_ground() {
    let base = generated();
    let s = edited(&base, [(page_tile(), page(|_, _| 12.0, |_, _| 1.0))]);
    let dir = page_tile().direction(0.5, 0.5);
    let from = dvec3::add(s.center, dvec3::scale(dir, s.radius + 500.0));
    let down = dvec3::scale(dir, -1.0);
    let hit = s.ray_hit(from, down, 1_000.0).unwrap();
    let plain = base.ray_hit(from, down, 1_000.0).unwrap();
    assert!((plain.distance - hit.distance - 12.0).abs() < 0.01);
    assert_eq!(hit.normal, s.normal(s.up_at(hit.point)));
}

// Axis `k` of a frame along the ground at `dir`.
fn edit_frame_axis(dir: DVec3, k: usize) -> DVec3 {
    let side = dvec3::normalize_or(dvec3::cross(dir, [0.3, 0.5, 0.8]), [1.0, 0.0, 0.0]);
    if k == 0 {
        side
    } else {
        dvec3::cross(dir, side)
    }
}
