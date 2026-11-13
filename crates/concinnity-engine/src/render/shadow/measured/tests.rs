use super::*;
use crate::gfx::projection::look_at;
use crate::gfx::render_types::RECEIVER_EMPTY;
use crate::render::depth::camera_projection;
use crate::transform::{mat4_inverse, mat4_mul};
use concinnity_core::math::quat_from_axis_angle;
use concinnity_core::math::vec3::normalize_clamped;

const FOV: f32 = 1.0;
const ASPECT: f32 = 16.0 / 9.0;
const NEAR: f32 = 0.1;
const BANDS: [[f32; 2]; NUM_SHADOW_CASCADES] = [[0.0, 3.5], [3.0, 8.4], [7.6, 22.0], [19.9, 80.0]];

fn to_light() -> [f32; 3] {
    normalize_clamped([0.3, 0.8, 0.2], 1e-6)
}

// A camera 2 m up at `eye`, looking toward `target`.
fn view_at(eye: [f32; 3], target: [f32; 3], to_light: [f32; 3]) -> (MeasureView, Mat4) {
    let view = look_at(eye, target, [0.0, 1.0, 0.0]);
    let vp = mat4_mul(camera_projection(FOV, ASPECT, NEAR), view);
    (
        MeasureView {
            view,
            eye,
            depth_inv_vp: mat4_inverse(vp),
            fov_y: FOV,
            aspect: ASPECT,
            to_light,
            bands: BANDS,
            size: [320, 180],
        },
        vp,
    )
}

fn looking_ahead() -> (MeasureView, Mat4) {
    view_at([0.0, 2.0, 0.0], [0.0, 1.0, -10.0], to_light())
}

fn clip(m: &Mat4, p: [f32; 3]) -> [f32; 4] {
    core::array::from_fn(|r| m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r])
}

// What the kernel computes for the texel at NDC `ndc` holding device depth
// `d`: the light-space point and view depth, or `None` for no surface.
fn kernel_point(params: &ReceiverBoundsParams, ndc: [f32; 2], d: f32) -> Option<[f32; 4]> {
    let at = [ndc[0], ndc[1], d, 1.0];
    let w: f32 = (0..4).map(|k| params.w_row[k] * at[k]).sum();
    if d <= 0.0 || w <= 1e-7 {
        return None;
    }
    let m = &params.to_light_space;
    Some(core::array::from_fn(|r| {
        (0..4).map(|k| m[k][r] * at[k]).sum::<f32>() / w
    }))
}

/// The kernel's reduction under `params` over a 64x36 grid of texels seeing
/// the ground plane y = 0 from `eye` through `vp`.
pub(crate) fn measure_ground(
    params: &ReceiverBoundsParams,
    eye: [f32; 3],
    vp: &Mat4,
) -> ReceiverBounds {
    let inverse = mat4_inverse(*vp);
    let mut bands = [ReceiverBand::EMPTY; NUM_SHADOW_CASCADES];
    for j in 0..36 {
        for i in 0..64 {
            let ndc = [(i as f32 + 0.5) / 32.0 - 1.0, 1.0 - (j as f32 + 0.5) / 18.0];
            let far = clip(&inverse, [ndc[0], ndc[1], 0.0]);
            let dir = sub([far[0], far[1], far[2]], scale(eye, far[3]));
            if dir[1] >= 0.0 {
                continue;
            }
            let hit = add(eye, scale(dir, -eye[1] / dir[1]));
            let c = clip(vp, hit);
            let Some(q) = kernel_point(params, ndc, c[2] / c[3]) else {
                continue;
            };
            for (b, band) in bands.iter_mut().enumerate() {
                if q[3] >= params.band_near[b] && q[3] <= params.band_far[b] {
                    for (k, v) in q.iter().enumerate() {
                        band.lo[k] = band.lo[k].min(*v);
                        band.hi[k] = band.hi[k].max(*v);
                    }
                }
            }
        }
    }
    ReceiverBounds {
        bands,
        serial: params.serial,
        _pad: [0; 3],
    }
}

fn ground_of(view: &MeasureView, vp: &Mat4, serial: u32) -> ReceiverBounds {
    measure_ground(&view.params(serial), view.eye, vp)
}

// The kernel's matrix takes a texel's NDC and device depth to the point's
// light-space coordinates and view depth, and bands not in use match nothing.
#[test]
fn params_take_a_depth_texel_to_light_space_and_view_depth() {
    let (mut view, vp) = looking_ahead();
    view.bands[3] = [1.0, 0.0];
    let params = view.params(7);
    assert_eq!(params.serial, 7);
    assert_eq!(
        [params.width, params.height, params.groups_x],
        [320, 180, 5]
    );
    assert!(params.band_near[3] > params.band_far[3]);
    let axes = LightBasis::axes(view.to_light);
    let forward = camera_basis(&view.view)[2];
    for p in [[0.5, 0.0, -6.0], [-3.0, 1.5, -20.0], [4.0, -0.5, -60.0]] {
        let c = clip(&vp, p);
        let ndc = [c[0] / c[3], c[1] / c[3]];
        let q = kernel_point(&params, ndc, c[2] / c[3]).unwrap();
        let want = [
            dot(axes[0], p),
            dot(axes[1], p),
            dot(axes[2], p),
            dot(forward, sub(p, view.eye)),
        ];
        for k in 0..4 {
            assert!(
                (q[k] - want[k]).abs() < 1e-2 * want[k].abs().max(1.0),
                "{p:?} {q:?} {want:?}"
            );
        }
    }
    assert!(
        kernel_point(&params, [0.0, 0.0], 0.0).is_none(),
        "a cleared texel"
    );
}

// Taken again through the same view, a measurement hands back what it
// measured, unpadded, and a band it found empty stays unmeasured.
#[test]
fn a_measurement_stands_for_the_view_it_was_taken_through() {
    let (view, vp) = looking_ahead();
    let mut measure = ReceiverMeasure::default();
    let params = measure.begin(&view, MovedSince::Nothing);
    let mut bounds = ground_of(&view, &vp, params.serial);
    bounds.bands[3] = ReceiverBand::EMPTY;
    measure.accept(&bounds);
    let bands = measure.camera_bands(&view, [100.0; NUM_SHADOW_CASCADES], MovedSince::Nothing);
    for c in 0..3 {
        let band = bands.band(c).unwrap();
        let raw = bounds.bands[c];
        for k in 0..3 {
            assert!((band.ranges[k][0] - raw.lo[k]).abs() < 1e-3, "{c} {k}");
            assert!((band.ranges[k][1] - raw.hi[k]).abs() < 1e-3, "{c} {k}");
        }
        assert_eq!(band.depths, [raw.lo[3], raw.hi[3]]);
    }
    assert!(bands.band(3).is_none());
}

// A measurement is taken only with the stamp it was numbered with, and an
// older one never replaces a newer.
#[test]
fn only_a_stamped_newer_measurement_is_taken() {
    let (view, vp) = looking_ahead();
    let mut measure = ReceiverMeasure::default();
    let first = measure.begin(&view, MovedSince::Nothing).serial;
    let second = measure.begin(&view, MovedSince::Nothing).serial;
    let unknown = ground_of(&view, &vp, second + 40);
    measure.accept(&unknown);
    assert!(measure.held.is_none());
    measure.accept(&ground_of(&view, &vp, second));
    measure.accept(&ground_of(&view, &vp, first));
    assert_eq!(measure.held.map(|h| h.stamp.serial), Some(second));
    measure.reset();
    measure.accept(&ground_of(&view, &vp, second));
    assert!(measure.held.is_none(), "a reset forgets every stamp");
}

fn measured(view: &MeasureView, vp: &Mat4) -> ReceiverMeasure {
    let mut measure = ReceiverMeasure::default();
    let serial = measure.begin(view, MovedSince::Nothing).serial;
    measure.accept(&ground_of(view, vp, serial));
    measure
}

// Past its age, after a turn wider than allowed or once the bands split
// differently, a measurement no longer stands for the view.
#[test]
fn an_old_turned_or_resplit_measurement_is_dropped() {
    let (view, vp) = looking_ahead();
    let reach = [100.0; NUM_SHADOW_CASCADES];
    let mut old = measured(&view, &vp);
    for _ in 0..MAX_MEASURED_AGE - 1 {
        old.begin(&view, MovedSince::Nothing);
    }
    assert!(
        old.camera_bands(&view, reach, MovedSince::Nothing)
            .band(0)
            .is_some()
    );
    old.begin(&view, MovedSince::Nothing);
    assert!(
        old.camera_bands(&view, reach, MovedSince::Nothing)
            .band(0)
            .is_none()
    );

    let measure = measured(&view, &vp);
    let turn = |angle: f32| {
        let target = [10.0 * angle.sin(), 1.0, -10.0 * angle.cos()];
        view_at(view.eye, target, view.to_light).0
    };
    let small = measure.camera_bands(&turn(0.02), reach, MovedSince::Nothing);
    let band = small.band(2).unwrap();
    let exact = measure
        .camera_bands(&view, reach, MovedSince::Nothing)
        .band(2)
        .unwrap();
    // Padded by the turn over the band's reach.
    let pad = band.ranges[0][1] - exact.ranges[0][1];
    assert!((pad - 0.02 * 100.0).abs() < 0.05, "{pad}");
    let under = MAX_MEASURED_TURN * 0.95;
    assert!(
        measure
            .camera_bands(&turn(under), reach, MovedSince::Nothing)
            .band(2)
            .is_some()
    );
    assert!(
        measure
            .camera_bands(&turn(MAX_MEASURED_TURN * 1.1), reach, MovedSince::Nothing)
            .band(2)
            .is_none()
    );

    let mut resplit = view;
    resplit.bands[2] = [7.6, 30.0];
    assert!(
        measure
            .camera_bands(&resplit, reach, MovedSince::Nothing)
            .band(0)
            .is_none()
    );
}

// A camera that moved since pads every band by how far it went.
#[test]
fn a_moved_camera_pads_by_its_move() {
    let (view, vp) = looking_ahead();
    let measure = measured(&view, &vp);
    let reach = [100.0; NUM_SHADOW_CASCADES];
    let exact = measure
        .camera_bands(&view, reach, MovedSince::Nothing)
        .band(1)
        .unwrap();
    let (moved, _) = view_at([0.0, 2.0, -1.5], [0.0, 1.0, -11.5], view.to_light);
    let band = measure
        .camera_bands(&moved, reach, MovedSince::Nothing)
        .band(1)
        .unwrap();
    for k in 0..3 {
        assert!((exact.ranges[k][0] - band.ranges[k][0] - 1.5).abs() < 1e-3);
        assert!((band.ranges[k][1] - exact.ranges[k][1] - 1.5).abs() < 1e-3);
    }
}

// Carried through a rebase, a measurement says what the same measurement taken
// in the new frame does.
#[test]
fn a_rebase_carries_the_measurement() {
    let rebase = Rebase {
        rotation: quat_from_axis_angle([0.0, 1.0, 0.0], 0.4),
        translation: [-300.0, 0.0, 120.0],
    };
    let (view, vp) = looking_ahead();
    let mut measure = measured(&view, &vp);
    measure.rebase(&rebase);
    let carried = MeasureView {
        view: rebase.reproject(view.view),
        eye: rebase.apply_point(view.eye),
        depth_inv_vp: rebase.apply_matrix(view.depth_inv_vp),
        to_light: rebase.apply_vector(view.to_light),
        ..view
    };
    let reach = [100.0; NUM_SHADOW_CASCADES];
    let got = measure.camera_bands(&carried, reach, MovedSince::Nothing);
    let new_vp = rebase.reproject(vp);
    let want = measured(&carried, &new_vp).camera_bands(&carried, reach, MovedSince::Nothing);
    for c in 0..NUM_SHADOW_CASCADES {
        let (g, w) = (got.band(c).unwrap(), want.band(c).unwrap());
        for k in 0..3 {
            for e in 0..2 {
                let (a, b) = (g.ranges[k][e], w.ranges[k][e]);
                assert!((a - b).abs() < 2e-2 * b.abs().max(1.0), "{c} {k} {a} {b}");
            }
        }
    }
}

// A caster that moved since the measurement, in a band's depths, widens the
// band to hold it; one with no box to say where drops the measurement.
#[test]
fn casters_moved_since_widen_the_band() {
    let (view, vp) = looking_ahead();
    let reach = [100.0; NUM_SHADOW_CASCADES];
    let mut measure = measured(&view, &vp);
    let flyer = ([1.0, 9.0, -12.0], [2.0, 10.0, -11.0]);
    measure.begin(&view, MovedSince::Within(flyer));
    let band = measure
        .camera_bands(&view, reach, MovedSince::Nothing)
        .band(2)
        .unwrap();
    let corners = box_corners(flyer);
    let axes = LightBasis::axes(view.to_light);
    for (axis, range) in axes.iter().zip(band.ranges) {
        let [lo, hi] = span(&corners, *axis);
        assert!(range[0] <= lo && hi <= range[1], "{axis:?}");
    }
    let near = measure
        .camera_bands(&view, reach, MovedSince::Nothing)
        .band(0)
        .unwrap();
    let alone = measured(&view, &vp).camera_bands(&view, reach, MovedSince::Nothing);
    assert_eq!(Some(near), alone.band(0), "a band the flyer is not in");
    assert!(
        measure
            .camera_bands(&view, reach, MovedSince::Anywhere)
            .band(0)
            .is_none()
    );
}

#[test]
fn moved_since_merges_boxes() {
    let a = ([0.0; 3], [1.0; 3]);
    let b = ([-1.0, 2.0, 0.5], [0.5, 3.0, 4.0]);
    assert_eq!(MovedSince::of(false, []), MovedSince::Nothing);
    assert_eq!(
        MovedSince::of(false, [a, b]),
        MovedSince::Within(([-1.0, 0.0, 0.0], [1.0, 3.0, 4.0]))
    );
    assert_eq!(MovedSince::of(true, [a]), MovedSince::Anywhere);
    assert!(newer(0, u32::MAX) && !newer(u32::MAX, 0) && !newer(3, 3));
    assert_eq!(ReceiverBand::EMPTY.lo[0], RECEIVER_EMPTY);
}

// A change of the field of view or aspect widens or narrows what the camera
// sees, so the measurement no longer stands for it.
#[test]
fn a_new_projection_drops_the_measurement() {
    let (view, vp) = looking_ahead();
    let measure = measured(&view, &vp);
    let reach = [100.0; NUM_SHADOW_CASCADES];
    assert!(
        measure
            .camera_bands(&view, reach, MovedSince::Nothing)
            .band(0)
            .is_some()
    );
    for changed in [
        MeasureView {
            fov_y: FOV * 1.2,
            ..view
        },
        MeasureView {
            aspect: 4.0 / 3.0,
            ..view
        },
    ] {
        assert!(
            measure
                .camera_bands(&changed, reach, MovedSince::Nothing)
                .band(0)
                .is_none()
        );
    }
}

// Serial 0 is never stamped: a readback slot nothing was written to reads back
// as zeros and is never taken for a measurement.
#[test]
fn an_unwritten_readback_is_never_taken() {
    let (view, _) = looking_ahead();
    let mut measure = ReceiverMeasure::default();
    assert_eq!(measure.begin(&view, MovedSince::Nothing).serial, 1);
    let unwritten: ReceiverBounds = bytemuck::Zeroable::zeroed();
    measure.accept(&unwritten);
    assert!(measure.held.is_none());
}

// Numbering wraps past u32::MAX to 1: a measurement numbered after the wrap is
// newer, an older one does not replace it, ages count across it, and the
// casters moved since are found across it.
#[test]
fn serials_wrap_past_zero() {
    let (view, vp) = looking_ahead();
    let reach = [100.0; NUM_SHADOW_CASCADES];
    let mut measure = ReceiverMeasure {
        next: u32::MAX - 1,
        ..ReceiverMeasure::default()
    };
    let before = measure.begin(&view, MovedSince::Nothing).serial;
    let last = measure.begin(&view, MovedSince::Nothing).serial;
    let flyer = ([1.0, 9.0, -12.0], [2.0, 10.0, -11.0]);
    let wrapped = measure.begin(&view, MovedSince::Within(flyer)).serial;
    assert_eq!([before, last, wrapped], [u32::MAX - 1, u32::MAX, 1]);
    assert!(newer(wrapped, last) && !newer(last, wrapped));
    assert_eq!(steps(before, measure.next), 3);

    measure.accept(&ground_of(&view, &vp, before));
    let band = measure
        .camera_bands(&view, reach, MovedSince::Nothing)
        .band(2)
        .unwrap();
    // The flyer stands above all the ground, toward the light.
    let [lo, hi] = span(&box_corners(flyer), LightBasis::axes(view.to_light)[2]);
    let ground = ground_of(&view, &vp, before).bands[2];
    assert!(hi > ground.hi[2]);
    assert!(band.ranges[2][0] <= lo && hi <= band.ranges[2][1]);

    measure.accept(&ground_of(&view, &vp, wrapped));
    measure.accept(&ground_of(&view, &vp, last));
    assert_eq!(measure.held.map(|h| h.stamp.serial), Some(wrapped));
}
