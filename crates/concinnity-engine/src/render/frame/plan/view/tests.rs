use super::*;
use crate::gfx::render_types::{GroundBoundGpu, ViewSampleGrid};
use crate::render::depth::camera_projection;
use crate::render::post::reflection_layer::NO_LEVEL_SURFACES;
use crate::render::shadow::far_anchor::GroundSphere;
use crate::render::view::SubjectKind;
use crate::transform::{mat4_inverse, mat4_mul};
use concinnity_core::math::{cos, sin};

const FOV: f32 = core::f32::consts::FRAC_PI_2;

// A camera at `eye` turned `yaw` radians about +y from looking down -z.
fn view_at(eye: [f32; 3], yaw: f32) -> Mat4 {
    let (s, c) = (sin(yaw), cos(yaw));
    // Rows: right, up, and the view's +z (backward).
    let right = [c, 0.0, -s];
    let back = [s, 0.0, c];
    let up = [0.0, 1.0, 0.0];
    let t = |r: [f32; 3]| -(r[0] * eye[0] + r[1] * eye[1] + r[2] * eye[2]);
    [
        [right[0], up[0], back[0], 0.0],
        [right[1], up[1], back[1], 0.0],
        [right[2], up[2], back[2], 0.0],
        [t(right), t(up), t(back), 1.0],
    ]
}

fn inputs(eye: [f32; 3], yaw: f32) -> ViewInputs {
    let view = view_at(eye, yaw);
    let proj = camera_projection(FOV, 2.0, 0.1);
    ViewInputs {
        view,
        eye,
        frustum: Frustum::from_camera(mat4_mul(proj, view), None),
        fov_y: FOV,
        depth_inv_vp: mat4_inverse(mat4_mul(proj, view)),
        render_size: [200, 100],
        subjects: ViewSubjects::default(),
        sun: Sun {
            direction: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0],
        },
    }
}

#[test]
fn the_focal_length_turns_sizes_into_pixels() {
    let mut plan = ViewPlan::default();
    plan.refresh(&inputs([0.0; 3], 0.0));
    // 100 pixels tall over a 90 degree view: 50 pixels per unit of tan.
    let facts = plan.facts();
    assert!((facts.focal_px - 50.0).abs() < 1e-3);
    assert_eq!(facts.render_size, [200, 100]);
    assert!((facts.pixels(1.0, 10.0) - 5.0).abs() < 1e-4);
}

#[test]
fn a_sampled_frame_hands_the_pass_its_grid_and_an_unsampled_one_none() {
    let mut plan = ViewPlan::default();
    let frame = inputs([0.0; 3], 0.0);
    plan.refresh(&frame);
    plan.stamp(&frame, true);
    let params = plan.samples.expect("sampled");
    let grid = ViewSampleGrid::over(200, 100);
    assert_eq!(
        (params.width, params.height, params.cell, params.columns),
        (200, 100, grid.cell, grid.columns)
    );
    plan.refresh(&frame);
    plan.stamp(&frame, false);
    assert!(plan.samples.is_none());
}

#[test]
fn a_turn_moves_the_view_by_its_angle_in_pixels_and_a_reset_starts_still() {
    let mut plan = ViewPlan::default();
    plan.refresh(&inputs([0.0; 3], 0.0));
    assert_eq!(plan.facts().motion_px, 0.0);
    plan.refresh(&inputs([0.0; 3], 0.1));
    assert!(
        (plan.facts().motion_px - 5.0).abs() < 0.01,
        "{}",
        plan.facts().motion_px
    );
    // Without a measurement a move is not seen against anything.
    plan.refresh(&inputs([3.0, 0.0, 0.0], 0.1));
    assert_eq!(plan.facts().motion_px, 0.0);
    plan.reset();
    plan.refresh(&inputs([0.0; 3], 1.0));
    assert_eq!(plan.facts().motion_px, 0.0);
}

// A plan whose camera at the origin measured a surface 10 m ahead in every
// cell.
fn measured_ten_meters_ahead() -> ViewPlan {
    let mut plan = ViewPlan::default();
    let frame = inputs([0.0; 3], 0.0);
    plan.refresh(&frame);
    plan.stamp(&frame, true);
    let serial = plan.samples.unwrap().serial;
    let proj = camera_projection(FOV, 2.0, 0.1);
    let clip = crate::transform::mat4_transform(&proj, [0.0, 0.0, -10.0, 1.0]);
    let cells = alloc::vec![
        ViewSampleCell {
            depth: clip[2] / clip[3],
            nearest: clip[2] / clip[3],
            texel: (50 << 16) + 100,
            serial,
        };
        ViewSampleGrid::over(200, 100).cells()
    ];
    plan.measured(&cells);
    plan.refresh(&inputs([0.0; 3], 0.0));
    plan
}

// A slow pan still moves the view: a tenth of a milliradian a frame is half
// a pixel at a 4K focal length.
#[test]
fn a_slow_turn_is_not_lost_to_rounding() {
    let mut plan = ViewPlan::default();
    plan.refresh(&inputs([0.0; 3], 0.0));
    plan.refresh(&inputs([0.0; 3], 1e-4));
    let expected = 1e-4 * plan.facts().focal_px;
    assert!(
        (plan.facts().motion_px - expected).abs() < expected * 0.05,
        "{}",
        plan.facts().motion_px
    );
}

#[test]
fn a_move_reads_against_the_nearest_surface_measured() {
    let mut plan = measured_ten_meters_ahead();
    let nearest = plan.facts().nearest().unwrap();
    assert!((nearest - 10.0).abs() < 0.05, "{nearest}");
    plan.refresh(&inputs([1.0, 0.0, 0.0], 0.0));
    // 1 m against a surface 10 m away is a tenth of a radian.
    assert!(
        (plan.facts().motion_px - 5.0).abs() < 0.05,
        "{}",
        plan.facts().motion_px
    );
    assert_eq!(plan.facts().parallax_px, plan.facts().motion_px);
}

// A turn moves every surface alike and uncovers nothing: no parallax.
#[test]
fn a_turn_in_place_has_no_parallax() {
    let mut plan = measured_ten_meters_ahead();
    plan.refresh(&inputs([0.0; 3], 0.2));
    assert!(plan.facts().motion_px > 1.0);
    assert_eq!(plan.facts().parallax_px, 0.0);
}

// A box 10 m ahead, `x` meters to the side.
fn ahead(x: f32) -> Aabb {
    ([x - 0.5, -0.5, -10.5], [x + 0.5, 0.5, -9.5])
}

#[test]
fn a_mover_in_view_reads_its_travel_against_the_camera_s() {
    let mut plan = measured_ten_meters_ahead();
    plan.moved(Some(ahead(0.0)), Some(ahead(1.0)));
    plan.refresh(&inputs([0.0; 3], 0.0));
    // 1 m across, 10 m away, at 50 pixels a radian.
    assert!(
        (plan.facts().mover_px - 5.0).abs() < 0.1,
        "{}",
        plan.facts().mover_px
    );
    // Gone the next frame, once nothing moves.
    plan.refresh(&inputs([0.0; 3], 0.0));
    assert_eq!(plan.facts().mover_px, 0.0);
    // A mover the camera keeps pace with does not move against it.
    plan.moved(Some(ahead(0.0)), Some(ahead(1.0)));
    plan.refresh(&inputs([1.0, 0.0, 0.0], 0.0));
    assert!(plan.facts().mover_px < 1e-3, "{}", plan.facts().mover_px);
}

#[test]
fn a_mover_out_of_view_counts_for_nothing_and_a_reshape_in_view_for_everything() {
    let mut plan = measured_ten_meters_ahead();
    let behind = Some(([-0.5, -0.5, 9.5], [0.5, 0.5, 10.5]));
    let further = Some(([0.5, -0.5, 9.5], [1.5, 0.5, 10.5]));
    plan.moved(behind, further);
    plan.refresh(&inputs([0.0; 3], 0.0));
    assert_eq!(plan.facts().mover_px, 0.0);
    plan.reshaped(ahead(0.0));
    plan.refresh(&inputs([0.0; 3], 0.0));
    assert_eq!(plan.facts().mover_px, f32::INFINITY);
    plan.moved(None, Some(ahead(0.0)));
    plan.refresh(&inputs([0.0; 3], 0.0));
    assert_eq!(plan.facts().mover_px, f32::INFINITY);
}

#[test]
fn a_rebase_carries_where_the_camera_stood() {
    let mut plan = measured_ten_meters_ahead();
    plan.rebase(&Rebase {
        translation: [100.0, 0.0, 0.0],
        ..Rebase::IDENTITY
    });
    plan.refresh(&inputs([100.0, 0.0, 0.0], 0.0));
    assert!(plan.facts().motion_px < 1e-3, "{}", plan.facts().motion_px);
}

#[test]
fn the_light_on_the_ground_follows_the_light_s_height_over_the_horizon() {
    let mut plan = ViewPlan::default();
    let mut frame = inputs([0.0; 3], 0.0);
    plan.refresh(&frame);
    assert!((plan.facts().direct_light - 1.0).abs() < 1e-4);
    frame.sun.direction = [0.0, 0.5, 0.866_025_4];
    plan.refresh(&frame);
    assert!((plan.facts().direct_light - 0.5).abs() < 1e-4);
    frame.sun.direction = [0.0, -1.0, 0.0];
    plan.refresh(&frame);
    assert_eq!(plan.facts().direct_light, 0.0);
}

#[test]
fn a_planet_world_reports_the_camera_s_heights_and_its_subjects() {
    let eye = [0.0, 25.0, 0.0];
    let radius = 1.0e5;
    let center = [0.0, 25.0 - radius - 25.0, 0.0];
    let mut level = NO_LEVEL_SURFACES;
    level.layers[0].shape = [center[0], center[1], center[2], radius + 5.0];
    level.layers[0].extent = [0.0, 0.0, 20.0, 1.0];
    let mut frame = inputs(eye, 0.0);
    frame.subjects = ViewSubjects::new(
        Some(GroundSphere { center, radius }),
        &GroundBoundGpu {
            sphere: [center[0], center[1], center[2], radius + 50.0],
            camera: [-25.0, 0.0, 0.0, 0.0],
        },
        &level,
        None,
    );
    let mut plan = ViewPlan::default();
    plan.refresh(&frame);
    let facts = plan.facts();
    assert!((facts.ground_height.unwrap() - 25.0).abs() < 0.05);
    assert_eq!(facts.sea_height, Some(20.0));
    let kinds: alloc::vec::Vec<SubjectKind> = facts.subjects().map(|s| s.kind).collect();
    assert_eq!(kinds, [SubjectKind::Planet, SubjectKind::Water]);
    assert!(facts.seen.is_none());
    // Without a planet nothing is under the camera.
    plan.refresh(&inputs(eye, 0.0));
    assert_eq!(plan.facts().ground_height, None);
}

// A mover leaving the view uncovers what it hid, so it counts where it was.
#[test]
fn a_mover_leaving_the_view_still_counts() {
    let mut plan = measured_ten_meters_ahead();
    let gone = ([-0.5, -0.5, 9.5], [0.5, 0.5, 10.5]);
    plan.moved(Some(ahead(0.0)), Some(gone));
    plan.refresh(&inputs([0.0; 3], 0.0));
    assert!(plan.facts().mover_px > 1.0, "{}", plan.facts().mover_px);
}

// A measurement taken mid-turn reads as having drifted by the whole turn
// since, even once the camera stops.
#[test]
fn the_drift_since_a_measurement_sums_each_frame_s_motion() {
    let mut plan = measured_ten_meters_ahead();
    assert_eq!(plan.facts().seen.unwrap().drift_px, plan.facts().motion_px);
    let frame = |plan: &mut ViewPlan, yaw| {
        let at = inputs([0.0; 3], yaw);
        plan.stamp(&at, true);
        plan.refresh(&at);
    };
    frame(&mut plan, 0.2);
    let turn = plan.facts().motion_px;
    assert!(turn > 1.0);
    frame(&mut plan, 0.2);
    let seen = plan.facts().seen.unwrap();
    assert_eq!(plan.facts().motion_px, 0.0);
    assert!(
        (seen.drift_px - turn).abs() < 1e-4,
        "{} {turn}",
        seen.drift_px
    );
    plan.reset();
    plan.refresh(&inputs([0.0; 3], 0.2));
    assert!(plan.facts().seen.is_none());
}
