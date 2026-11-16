use super::facts::{Sighting, SubjectKind, VIEW_SUBJECTS, ViewFacts};
use super::measure::{MAX_MEASURED_AGE, MeasureView, ViewMeasure};
use super::screen::ScreenRect;
use super::subjects::{Hit, PLANET_SLOT, PROP_SLOT, PlanetSubject, ViewSubjects};
use crate::gfx::frustum::Frustum;
use crate::gfx::render_types::{
    GroundBoundGpu, LevelLayerGpu, LevelSurfacesGpu, ViewSampleCell, ViewSampleGrid,
};
use crate::render::depth::{DEPTH_FAR, camera_projection};
use crate::render::post::reflection_layer::NO_LEVEL_SURFACES;
use crate::render::shadow::far_anchor::GroundSphere;
use crate::transform::{Mat4, mat4_inverse, mat4_mul, mat4_transform};
use alloc::vec::Vec;
use concinnity_core::math::vec3::{normalize, sub};

const DOWN: [f32; 3] = [0.0, -1.0, 0.0];
const UP: [f32; 3] = [0.0, 1.0, 0.0];

// A water rectangle `height` meters under the camera at `eye`, spanning
// `x_range` along x and 200 m along z about the camera.
fn rectangle(eye: [f32; 3], height: f32, x_range: [f32; 2]) -> LevelLayerGpu {
    let [x0, x1] = x_range;
    LevelLayerGpu {
        shape: [(x0 + x1) * 0.5, eye[1] - height, eye[2], 0.0],
        extent: [(x1 - x0) * 0.5, 100.0, height, 1.0],
        ..NO_LEVEL_SURFACES.layers[0]
    }
}

// A sea shell of `radius` about a center straight under the camera, which
// stands `height` over it.
fn shell(eye: [f32; 3], radius: f32, height: f32) -> LevelLayerGpu {
    LevelLayerGpu {
        shape: [eye[0], eye[1] - radius - height, eye[2], radius],
        extent: [0.0, 0.0, height, 1.0],
        ..NO_LEVEL_SURFACES.layers[0]
    }
}

fn with_water(layers: &[LevelLayerGpu]) -> LevelSurfacesGpu {
    let mut level = NO_LEVEL_SURFACES;
    level.layers[..layers.len()].copy_from_slice(layers);
    level
}

// A planet of `radius` whose ground lies `height` under the camera at `eye`,
// and stays under `relief` meters above that.
fn planet(eye: [f32; 3], radius: f32, height: f32, relief: f32) -> PlanetSubject {
    let center = [eye[0], eye[1] - radius - height, eye[2]];
    PlanetSubject {
        ground: GroundSphere { center, radius },
        bound: GroundBoundGpu {
            sphere: [center[0], center[1], center[2], radius + relief],
            camera: [height - relief, 0.0, 0.0, 0.0],
        },
    }
}

#[test]
fn a_ray_down_meets_the_sea_at_the_camera_s_height_and_a_ray_up_the_sky() {
    let eye = [0.0, 20.0, 0.0];
    let subjects = ViewSubjects::new(
        None,
        &GroundBoundGpu::default(),
        &with_water(&[shell(eye, 6.0e6, 20.0)]),
        None,
    );
    match subjects.hit(eye, DOWN, None) {
        Hit::Subject(slot, t) => {
            assert_eq!(slot, 1);
            assert!((t - 20.0).abs() < 0.05, "{t}");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(subjects.hit(eye, UP, None), Hit::Sky);
    assert_eq!(subjects.sea_height(), Some(20.0));
}

#[test]
fn whichever_of_water_and_an_opaque_surface_is_nearer_is_seen() {
    let eye = [0.0, 10.0, 0.0];
    let subjects = ViewSubjects::new(
        None,
        &GroundBoundGpu::default(),
        &with_water(&[rectangle(eye, 10.0, [-50.0, 50.0])]),
        None,
    );
    // A floor under the water shows the water.
    assert!(matches!(
        subjects.hit(eye, DOWN, Some([0.0, -12.0, 0.0])),
        Hit::Subject(1, t) if (t - 10.0).abs() < 1e-3
    ));
    // A post standing out of it shows the post.
    assert_eq!(
        subjects.hit(eye, DOWN, Some([0.0, -4.0, 0.0])),
        Hit::Other(4.0)
    );
}

#[test]
fn a_water_rectangle_shows_only_inside_its_extent() {
    let eye = [0.0, 10.0, 0.0];
    let subjects = ViewSubjects::new(
        None,
        &GroundBoundGpu::default(),
        &with_water(&[rectangle(eye, 10.0, [5.0, 50.0])]),
        None,
    );
    assert_eq!(subjects.hit(eye, DOWN, None), Hit::Sky);
    let toward = normalize([20.0, -10.0, 0.0]);
    assert!(matches!(
        subjects.hit(eye, toward, None),
        Hit::Subject(1, _)
    ));
    // No sea among them.
    assert_eq!(subjects.sea_height(), None);
}

#[test]
fn ground_and_what_stands_on_it_belong_to_the_planet_and_the_prop_wins_its_box() {
    let eye = [0.0, 30.0, 0.0];
    let mut subjects = ViewSubjects {
        planet: Some(planet(eye, 1.0e6, 30.0, 10.0)),
        ..ViewSubjects::default()
    };
    // The ground under the camera.
    assert_eq!(
        subjects.hit(eye, DOWN, Some([0.0, -30.0, 0.0])),
        Hit::Subject(PLANET_SLOT, 30.0)
    );
    // A rock 8 m tall still lies under the bound.
    assert_eq!(
        subjects.hit(eye, DOWN, Some([0.0, -22.0, 0.0])),
        Hit::Subject(PLANET_SLOT, 22.0)
    );
    // A balloon over the bound does not.
    assert_eq!(
        subjects.hit(eye, DOWN, Some([0.0, -5.0, 0.0])),
        Hit::Other(5.0)
    );
    subjects.prop = Some(([-1.0, -0.5, -1.0], [1.0, 1.5, 1.0]));
    assert_eq!(
        subjects.hit(eye, DOWN, Some([0.0, -28.5, 0.0])),
        Hit::Subject(PROP_SLOT, 28.5)
    );
}

#[test]
fn each_subject_carries_its_size_and_distances() {
    let eye = [0.0, 30.0, 0.0];
    let subjects = ViewSubjects {
        planet: Some(planet(eye, 1.0e6, 30.0, 10.0)),
        water: with_water(&[shell(eye, 1.0e6 + 2.0, 28.0)]).layers,
        prop: Some(([3.0, 0.0, 0.0], [5.0, 2.0, 2.0])),
    };
    let (view, _) = looking_down(eye);
    let proj = camera_projection(core::f32::consts::FRAC_PI_2, 1.0, 0.1);
    let vp = mat4_mul(proj, view);
    let all = subjects.subjects(eye, &Frustum::from_camera(vp, None), &vp);
    assert!(all.iter().flatten().all(|s| s.in_view));
    let planet = all[PLANET_SLOT].unwrap();
    assert_eq!(planet.kind, SubjectKind::Planet);
    assert_eq!(planet.size, 2.0e6);
    assert!((planet.distance - 30.0).abs() < 0.2);
    let sea = all[1].unwrap();
    assert_eq!((sea.kind, sea.distance), (SubjectKind::Water, 28.0));
    let prop = all[PROP_SLOT].unwrap();
    assert_eq!(prop.kind, SubjectKind::Prop);
    assert!((prop.distance - (9.0f32 + 28.0 * 28.0).sqrt()).abs() < 1e-3);
    assert!(all[2].is_none());
    // Resting on the ground under the sea's surface, the prop stands clear of
    // nothing; the planet and the water carry their own relief.
    assert!(prop.clearance < 0.2, "{}", prop.clearance);
    assert_eq!((planet.clearance, sea.clearance), (0.0, 0.0));
    // Looking up from over the prop, it is out of view; the ground is not,
    // being all around the horizon.
    let up = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 30.0, 1.0],
    ];
    let up_vp = mat4_mul(proj, up);
    let skyward = subjects.subjects(eye, &Frustum::from_camera(up_vp, None), &up_vp);
    assert!(!skyward[PROP_SLOT].unwrap().in_view);
}

#[test]
fn a_subject_spans_its_angular_size_and_never_more_than_half_a_turn() {
    let facts = ViewFacts {
        focal_px: 1000.0,
        ..ViewFacts::default()
    };
    let near = super::Subject {
        kind: SubjectKind::Prop,
        size: 2.0,
        distance: 99.0,
        center_distance: 100.0,
        clearance: 0.0,
        in_view: true,
        rect: ScreenRect::FULL,
        seen: Sighting::default(),
    };
    assert!((near.extent_px(facts.focal_px) - 20.0).abs() < 0.01);
    assert!((facts.pixels(2.0, 100.0) - 20.0).abs() < 1e-4);
    let inside = super::Subject {
        center_distance: 0.0,
        ..near
    };
    assert!((inside.extent_px(1000.0) - core::f32::consts::PI * 1000.0).abs() < 0.1);
    // At its own radius the sphere fills half a turn too.
    let touching = super::Subject {
        center_distance: 1.0,
        ..near
    };
    assert!((touching.extent_px(1000.0) - core::f32::consts::PI * 1000.0).abs() < 0.1);
}

// A camera at `eye` looking straight down, with a 90 degree field of view.
fn looking_down(eye: [f32; 3]) -> (Mat4, Mat4) {
    // Right = +x, up = -z, forward = -y.
    let view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [-eye[0], eye[2], -eye[1], 1.0],
    ];
    let proj = camera_projection(core::f32::consts::FRAC_PI_2, 1.0, 0.1);
    let vp = mat4_mul(proj, view);
    (view, mat4_inverse(vp))
}

// The cells a GPU sampling a depth of floor at height `floor` would write, for
// a `size` square depth seen through `inv_vp` from `eye`.
fn floor_cells(
    eye: [f32; 3],
    inv_vp: &Mat4,
    size: u32,
    floor: f32,
    serial: u32,
) -> Vec<ViewSampleCell> {
    let vp = mat4_inverse(*inv_vp);
    let depth_at = |x: u32, y: u32| {
        let ndc = [
            (x as f32 + 0.5) / size as f32 * 2.0 - 1.0,
            1.0 - (y as f32 + 0.5) / size as f32 * 2.0,
        ];
        let far = mat4_transform(inv_vp, [ndc[0], ndc[1], DEPTH_FAR, 1.0]);
        let dir = normalize(sub([far[0], far[1], far[2]], eye.map(|e| e * far[3])));
        let t = (eye[1] - floor) / -dir[1];
        let p = [eye[0] + dir[0] * t, floor, eye[2] + dir[2] * t];
        let clip = mat4_transform(&vp, [p[0], p[1], p[2], 1.0]);
        clip[2] / clip[3]
    };
    let grid = ViewSampleGrid::over(size, size);
    (0..grid.cells())
        .map(|i| {
            let [cx, cy] = grid.sample(i, grid.spot(serial), size, size);
            let index = u32::try_from(i).unwrap();
            let x0 = (index % grid.columns) * grid.cell;
            let y0 = (index / grid.columns) * grid.cell;
            let mut best = (DEPTH_FAR, [cx, cy]);
            for y in y0..(y0 + grid.cell).min(size) {
                for x in x0..(x0 + grid.cell).min(size) {
                    let d = depth_at(x, y);
                    if d > best.0 {
                        best = (d, [x, y]);
                    }
                }
            }
            ViewSampleCell {
                depth: depth_at(cx, cy),
                nearest: best.0,
                texel: (best.1[1] << 16) | best.1[0],
                serial,
            }
        })
        .collect()
}

#[test]
fn a_floor_half_under_water_measures_as_half_water_and_half_floor() {
    const SIZE: u32 = 128;
    let eye = [0.0, 10.0, 0.0];
    let (_, inv_vp) = looking_down(eye);
    let subjects = ViewSubjects::new(
        None,
        &GroundBoundGpu::default(),
        &with_water(&[rectangle(eye, 10.0, [-100.0, 0.0])]),
        None,
    );
    let mut measure = ViewMeasure::default();
    let params = measure.begin(&MeasureView {
        depth_inv_vp: inv_vp,
        eye,
        size: [SIZE, SIZE],
        subjects,
    });
    assert_eq!((params.width, params.height), (SIZE, SIZE));
    measure.accept(&floor_cells(eye, &inv_vp, SIZE, -1.0, params.serial));
    let (age, m) = measure.latest().expect("measured");
    assert_eq!(age, 1);
    assert_eq!(m.sky, 0.0);
    assert!((m.subjects[1].coverage - 0.5).abs() < 0.05, "{m:?}");
    assert!((m.other.coverage - 0.5).abs() < 0.05, "{m:?}");
    // The water's nearest part is straight under the camera, the floor's
    // just past the water's edge.
    assert!((m.subjects[1].nearest.unwrap() - 10.0).abs() < 0.1, "{m:?}");
    assert!((m.other.nearest.unwrap() - 11.0).abs() < 0.2, "{m:?}");
    assert_eq!(m.subjects[PLANET_SLOT], Sighting::default());
}

#[test]
fn an_empty_depth_is_all_sky() {
    let eye = [0.0, 10.0, 0.0];
    let (_, inv_vp) = looking_down(eye);
    let mut measure = ViewMeasure::default();
    let params = measure.begin(&MeasureView {
        depth_inv_vp: inv_vp,
        eye,
        size: [64, 64],
        subjects: ViewSubjects::default(),
    });
    let cells = alloc::vec![
        ViewSampleCell {
            serial: params.serial,
            ..ViewSampleCell::default()
        };
        ViewSampleGrid::over(64, 64).cells()
    ];
    measure.accept(&cells);
    let seen = measure.seen().unwrap();
    assert_eq!((seen.sky, seen.other), (1.0, Sighting::default()));
}

// A readback is taken only when it is the newest, every cell carries its
// number, and its view is still known.
#[test]
fn only_a_whole_newer_readback_of_a_known_view_is_taken() {
    let eye = [0.0, 10.0, 0.0];
    let (_, inv_vp) = looking_down(eye);
    let view = MeasureView {
        depth_inv_vp: inv_vp,
        eye,
        size: [64, 64],
        subjects: ViewSubjects::default(),
    };
    let cells = |serial: u32, depth: f32| {
        alloc::vec![
            ViewSampleCell {
                depth,
                serial,
                ..ViewSampleCell::default()
            };
            ViewSampleGrid::over(64, 64).cells()
        ]
    };
    let mut measure = ViewMeasure::default();
    let first = measure.begin(&view).serial;
    let second = measure.begin(&view).serial;
    // Nothing was written to the slot.
    measure.accept(&cells(0, 0.0));
    assert!(measure.latest().is_none());
    measure.accept(&cells(second, 0.0));
    assert_eq!(measure.latest().unwrap().0, 1);
    // An older readback arriving late is dropped.
    measure.accept(&cells(first, 0.5));
    assert_eq!(measure.seen().unwrap().sky, 1.0);
    // A readback with a cell from another measurement is dropped whole.
    let third = measure.begin(&view).serial;
    let mut torn = cells(third, 0.5);
    torn[3].serial = second;
    measure.accept(&torn);
    assert_eq!(measure.latest().unwrap().0, 2);
    // Too short for its grid.
    measure.accept(&cells(third, 0.5)[..10]);
    assert_eq!(measure.latest().unwrap().0, 2);
    // A view that does not continue the last forgets everything in flight.
    measure.reset();
    measure.accept(&cells(third, 0.5));
    assert!(measure.latest().is_none());
}

#[test]
fn a_measurement_stands_only_while_it_is_recent() {
    let view = MeasureView {
        depth_inv_vp: looking_down([0.0; 3]).1,
        eye: [0.0; 3],
        size: [8, 8],
        subjects: ViewSubjects::default(),
    };
    let mut measure = ViewMeasure::default();
    let serial = measure.begin(&view).serial;
    let cells = alloc::vec![
        ViewSampleCell {
            serial,
            ..ViewSampleCell::default()
        };
        ViewSampleGrid::over(8, 8).cells()
    ];
    measure.accept(&cells);
    for _ in 1..MAX_MEASURED_AGE {
        measure.begin(&view);
    }
    assert_eq!(measure.latest().unwrap().0, MAX_MEASURED_AGE);
    measure.begin(&view);
    assert!(measure.latest().is_none());
}

#[test]
fn the_facts_sum_each_kind_s_share_and_take_the_nearest_of_everything() {
    let water = |coverage, nearest| super::Subject {
        kind: SubjectKind::Water,
        size: 10.0,
        distance: 1.0,
        center_distance: 5.0,
        clearance: 0.0,
        in_view: true,
        rect: ScreenRect::FULL,
        seen: Sighting { coverage, nearest },
    };
    let mut facts = ViewFacts {
        seen: Some(super::Seen {
            age: 1,
            drift_px: 0.0,
            sky: 0.25,
            other: Sighting {
                coverage: 0.25,
                nearest: Some(30.0),
            },
        }),
        ..ViewFacts::default()
    };
    facts.subjects[1] = Some(water(0.25, Some(12.0)));
    facts.subjects[2] = Some(water(0.25, None));
    assert_eq!(facts.share(SubjectKind::Water), 0.5);
    assert_eq!(facts.share(SubjectKind::Planet), 0.0);
    assert_eq!(facts.nearest(), Some(12.0));
    facts.subjects[PROP_SLOT] = Some(super::Subject {
        kind: SubjectKind::Prop,
        ..water(0.1, Some(2.0))
    });
    assert_eq!(facts.nearest(), Some(2.0));
    assert_eq!(facts.nearest_still(), Some(12.0));
    assert_eq!(facts.subjects().count(), 3);
    assert_eq!(VIEW_SUBJECTS, facts.subjects.len());
    facts.seen = None;
    assert_eq!(facts.nearest(), None);
}

#[test]
fn a_prop_stands_clear_by_its_gap_to_the_highest_ground_or_the_nearest_water() {
    let eye = [0.0, 30.0, 0.0];
    let vp = mat4_mul(
        camera_projection(core::f32::consts::FRAC_PI_2, 1.0, 0.1),
        looking_down(eye).0,
    );
    let frustum = Frustum::from_camera(vp, None);
    let clearance = |subjects: &ViewSubjects| {
        subjects.subjects(eye, &frustum, &vp)[PROP_SLOT]
            .unwrap()
            .clearance
    };
    let at = |y: f32| Some(([-1.0, y, -1.0], [1.0, y + 2.0, 1.0]));
    // Ground 30 m under the camera, rising at most 2 m above that.
    let mut subjects = ViewSubjects {
        planet: Some(planet(eye, 1.0e5, 30.0, 2.0)),
        water: NO_LEVEL_SURFACES.layers,
        prop: at(20.0),
    };
    let over_ground = clearance(&subjects);
    assert!((over_ground - 18.0).abs() < 0.05, "{over_ground}");
    // Anywhere within the ground's relief, it may be touching it.
    subjects.prop = at(1.0);
    assert_eq!(clearance(&subjects), 0.0);
    // A sea 5 m over the ground lies nearer.
    subjects.prop = at(20.0);
    subjects.water = with_water(&[shell(eye, 1.0e5 + 5.0, 25.0)]).layers;
    let over_sea = clearance(&subjects);
    assert!((over_sea - 15.0).abs() < 0.05, "{over_sea}");
    // Deep under it, the sea's surface lies above.
    subjects.planet = None;
    subjects.prop = at(-12.0);
    let under_sea = clearance(&subjects);
    assert!((under_sea - 15.0).abs() < 0.05, "{under_sea}");
    // A water rectangle off to the side counts by the gap across to it.
    subjects.prop = at(20.0);
    subjects.water = with_water(&[rectangle(eye, 30.0, [10.0, 20.0])]).layers;
    let beside = clearance(&subjects);
    assert!((beside - (81.0f32 + 400.0).sqrt()).abs() < 1e-3, "{beside}");
    // With neither ground nor water, nothing says it floats.
    subjects.water = NO_LEVEL_SURFACES.layers;
    assert_eq!(clearance(&subjects), 0.0);
}

// Formed about the camera, the gap resolves a tenth of a meter over a planet
// the size of the Earth, finer than its center distance can.
#[test]
fn an_earth_sized_planet_resolves_a_prop_s_clearance_to_centimeters() {
    let eye = [0.0, 3.0, 0.0];
    let subjects = ViewSubjects {
        planet: Some(planet(eye, 6.371e6, 3.0, 0.5)),
        water: NO_LEVEL_SURFACES.layers,
        prop: Some(([-0.5, 0.6, 2.0], [0.5, 1.6, 3.0])),
    };
    let vp = mat4_mul(
        camera_projection(core::f32::consts::FRAC_PI_2, 1.0, 0.1),
        looking_down(eye).0,
    );
    let frustum = Frustum::from_camera(vp, None);
    let clearance = subjects.subjects(eye, &frustum, &vp)[PROP_SLOT]
        .unwrap()
        .clearance;
    assert!((clearance - 0.1).abs() < 0.02, "{clearance}");
}

// Looking straight down from 30 m with a 90 degree view, a 2 m box on the
// ground 4 m east covers a strip right of center; a box reaching the camera
// covers the whole screen.
#[test]
fn a_box_covers_the_part_of_the_screen_its_corners_project_to() {
    let eye = [0.0, 30.0, 0.0];
    let vp = mat4_mul(
        camera_projection(core::f32::consts::FRAC_PI_2, 1.0, 0.1),
        looking_down(eye).0,
    );
    let rect = ScreenRect::of_box(&vp, ([3.0, 0.0, -1.0], [5.0, 2.0, 1.0]));
    // Its west edge spans least at its foot, 30 m down, and its east edge
    // and its sides most at its top, 28 m down.
    let expect = ScreenRect {
        min: [0.5 + 3.0 / 60.0, 0.5 - 1.0 / 56.0],
        max: [0.5 + 5.0 / 56.0, 0.5 + 1.0 / 56.0],
    };
    for k in 0..2 {
        assert!((rect.min[k] - expect.min[k]).abs() < 1e-4, "{rect:?}");
        assert!((rect.max[k] - expect.max[k]).abs() < 1e-4, "{rect:?}");
    }
    let around = ScreenRect::of_box(&vp, ([-1.0, 29.0, -1.0], [1.0, 31.0, 1.0]));
    assert_eq!(around, ScreenRect::FULL);
    // Off to the side it clamps to the screen, and past it holds nothing.
    let edge = ScreenRect::of_box(&vp, ([20.0, 0.0, -1.0], [40.0, 2.0, 1.0]));
    assert_eq!(edge.max[0], 1.0);
    let past = ScreenRect::of_box(&vp, ([60.0, 0.0, -1.0], [80.0, 2.0, 1.0]));
    assert!(past.is_empty(), "{past:?}");
    assert!(ScreenRect::EMPTY.is_empty() && !ScreenRect::FULL.is_empty());
    assert_eq!(ScreenRect::EMPTY.union(rect), rect);
}

// The prop carries the part of the screen its box covers; the planet and the
// water can fill the screen.
#[test]
fn the_prop_carries_its_part_of_the_screen() {
    let eye = [0.0, 30.0, 0.0];
    let subjects = ViewSubjects {
        planet: Some(planet(eye, 1.0e6, 30.0, 10.0)),
        water: with_water(&[shell(eye, 1.0e6 + 2.0, 28.0)]).layers,
        prop: Some(([3.0, 0.0, -1.0], [5.0, 2.0, 1.0])),
    };
    let vp = mat4_mul(
        camera_projection(core::f32::consts::FRAC_PI_2, 1.0, 0.1),
        looking_down(eye).0,
    );
    let all = subjects.subjects(eye, &Frustum::from_camera(vp, None), &vp);
    assert_eq!(all[PLANET_SLOT].unwrap().rect, ScreenRect::FULL);
    assert_eq!(all[1].unwrap().rect, ScreenRect::FULL);
    let prop = all[PROP_SLOT].unwrap().rect;
    assert_eq!(prop, ScreenRect::of_box(&vp, subjects.prop.unwrap()));
    assert!(prop.max[0] - prop.min[0] < 0.05, "{prop:?}");
}

// A measurement that misses a subject an earlier one saw keeps its nearest
// until a sweep's worth in a row have missed it; seeing it again restarts
// the count, and a reset forgets it.
#[test]
fn a_missed_subject_keeps_its_nearest_for_a_sweep() {
    const SIZE: u32 = 128;
    let eye = [0.0, 10.0, 0.0];
    let (_, inv_vp) = looking_down(eye);
    let view = |water: &[LevelLayerGpu]| MeasureView {
        depth_inv_vp: inv_vp,
        eye,
        size: [SIZE, SIZE],
        subjects: ViewSubjects::new(None, &GroundBoundGpu::default(), &with_water(water), None),
    };
    let sea = [rectangle(eye, 10.0, [-100.0, 0.0])];
    let mut measure = ViewMeasure::default();
    let take = |measure: &mut ViewMeasure, water: &[LevelLayerGpu]| {
        let serial = measure.begin(&view(water)).serial;
        measure.accept(&floor_cells(eye, &inv_vp, SIZE, -1.0, serial));
        measure.latest().unwrap().1.subjects[1].nearest
    };
    let seen = take(&mut measure, &sea).expect("the sea is seen");
    let sweep = ViewSampleGrid::over(SIZE, SIZE).sweep();
    for missed in 1..sweep {
        assert_eq!(take(&mut measure, &[]), Some(seen), "{missed}");
    }
    assert_eq!(take(&mut measure, &[]), None);
    let again = take(&mut measure, &sea).expect("seen again");
    assert_eq!(take(&mut measure, &[]), Some(again));
    measure.reset();
    assert_eq!(take(&mut measure, &[]), None);
}
