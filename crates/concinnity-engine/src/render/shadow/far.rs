//! The far shadow: one coarse orthographic view of the large casters around
//! the camera, sampled where the cascades fade out, so hills and other big
//! casters still shadow the ground kilometers away.
//!
//! The view is a box centered on the camera rather than fit to its frustum, so
//! turning the camera never changes it, and its center snaps to the map's
//! texel grid on every light axis, so the texels stay anchored in the world
//! from one draw to the next. It re-centers with hysteresis: a camera moving
//! within a small share of the reach of where the map was drawn keeps the map,
//! since the shaded passes sample it through the projection it was drawn with
//! and measure its coverage from where it was centered. It also redraws when
//! the light turns at all, when its reach or map changes, and on a slow
//! timed beat so casters that stream in or move still reach it, never more
//! often than the tier's rate.

use crate::environment::planet::Rebase;
use crate::gfx::projection::look_at;
use crate::gfx::render_types::FarShadowLanes;
use crate::render::depth::shadow_ortho;
use crate::render::shadow::csm::{CascadeView, LightBasis, caster_extent, light_box_corner};
use crate::transform::mat4_mul;
use concinnity_core::math::vec3::{add, cross, dot, length_sq, normalize_clamped, scale, sub};
use concinnity_core::math::{round, sqrt};

// The share of the far shadow's reach, at its far end, over which it fades out.
const FAR_FADE_FRACTION: f32 = 0.15;

// How far the camera may stray from where the far map was drawn before it
// re-centers, as a share of the reach: 200 m at 4 km. Whole coverage then
// still reaches 80% of the reach around the camera.
const FAR_RECENTER_FRACTION: f32 = 0.05;

// The turn of the light, in radians, past which the far map redraws. Small
// enough that a low sun's long shadows move by under a texel: a 500 m caster
// at 2 degrees of elevation shifts its tip about 4 m, one texel of a 2048 map
// over 4 km. The tier's rate, not this, is what bounds the cost.
const FAR_LIGHT_EPSILON: f32 = 1e-5;

// How many far-map texels the depth compare offset spans, as the cascades'.
const FAR_BIAS_TEXELS: f32 = 2.0;

// The longest a far map whose view has not changed keeps, in seconds, before
// it redraws anyway, so casters that stream in, change detail or move still
// reach it.
const FAR_IDLE_REFRESH_SECONDS: f32 = 1.0;

/// What a quality tier spends on the far shadow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FarShadowBudget {
    /// The far map's edge, in texels.
    pub map_size: u32,
    /// The fewest frames between two redraws of the far map.
    pub refresh_frames: u32,
}

impl FarShadowBudget {
    /// The most any tier spends: a 2048-texel map, redrawn whenever it changes.
    pub const MAX: Self = Self {
        map_size: 2048,
        refresh_frames: 1,
    };
}

/// The far shadow a renderer runs, as far as where it takes its casters from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FarShadowReach {
    /// How far from the camera the far shadow covers, in world units; 0 for
    /// none.
    pub distance: f32,
    /// The far map's edge in texels; 0 when the renderer built none.
    pub map_size: u32,
}

impl FarShadowReach {
    /// How far from the camera a caster can lie and still render into the far
    /// view seen by `view`, whatever the light's direction: the farthest corner
    /// of its light box, which spans the coverage across and reaches back
    /// toward the light past it, from a center up to the re-centering distance
    /// away. 0 when no far shadow runs.
    pub fn caster_reach(&self, view: CascadeView) -> f32 {
        let reach = view
            .view_distance
            .map_or(self.distance, |d| self.distance.min(d));
        if self.map_size == 0 || reach <= 0.0 {
            return 0.0;
        }
        // The center snaps up to half a texel off the camera on three axes.
        let texel = 2.0 * reach / self.map_size as f32;
        let drift = FAR_RECENTER_FRACTION * reach + 0.5 * texel * sqrt(3.0);
        drift + light_box_corner(reach, caster_extent(reach))
    }
}

/// The far view's grid: lit along `to_light`, covering `reach` around its
/// center with `map_size` texels across.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FarGrid {
    to_light: [f32; 3],
    // Across the map, then toward the light.
    axes: [[f32; 3]; 3],
    up_hint: [f32; 3],
    reach: f32,
    map_size: u32,
}

impl FarGrid {
    /// The grid of a far view lit from `dir_to_source`, or `None` when it
    /// covers nothing or has no map.
    pub(crate) fn new(dir_to_source: [f32; 3], reach: f32, map_size: u32) -> Option<Self> {
        if reach <= 0.0 || map_size == 0 {
            return None;
        }
        let to_light = normalize_clamped(dir_to_source, 1e-6);
        let basis = LightBasis::toward(to_light);
        Some(Self {
            to_light,
            axes: [basis.right, basis.up, to_light],
            up_hint: basis.up_hint,
            reach,
            map_size,
        })
    }

    fn texel(&self) -> f32 {
        2.0 * self.reach / self.map_size as f32
    }

    // The texel-grid point nearest `p`.
    fn snap(&self, p: [f32; 3]) -> [f32; 3] {
        let texel = self.texel();
        self.axes.iter().fold([0.0; 3], |c, axis| {
            add(c, scale(*axis, round(dot(*axis, p) / texel) * texel))
        })
    }

    // Whether the light has turned from this grid's at all.
    fn turned_from(&self, to_light: [f32; 3]) -> bool {
        dot(self.to_light, to_light) <= 0.0
            || sqrt(length_sq(cross(self.to_light, to_light))) > FAR_LIGHT_EPSILON
    }

    // The lanes of the view centered on the grid point `center`.
    fn lanes(&self, center: [f32; 3]) -> FarShadowLanes {
        let r = self.reach;
        let eye = add(center, scale(self.to_light, r));
        let view = look_at(eye, center, self.up_hint);
        let proj = shadow_ortho(-r, r, -r, r, -caster_extent(r), 2.0 * r);
        let light_vp = mat4_mul(proj, view);
        let depth_per_meter = sqrt(length_sq([light_vp[0][2], light_vp[1][2], light_vp[2][2]]));
        FarShadowLanes {
            light_vp,
            center: [center[0], center[1], center[2], r],
            fade_start: r * (1.0 - FAR_FADE_FRACTION),
            compare_offset: FAR_BIAS_TEXELS * self.texel() * depth_per_meter,
            texel: 1.0 / self.map_size as f32,
            _pad: 0.0,
        }
    }
}

/// What the far view does on one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FarFrame {
    /// No far shadow: the shaded passes see none.
    Off,
    /// The map keeps what it holds, sampled through the lanes it was drawn
    /// with.
    Keep,
    /// The map redraws through these lanes, and is sampled through them.
    Draw(FarShadowLanes),
}

/// One frame's inputs to the far view's schedule.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FarTick {
    /// The view this frame would fit; `None` without a far shadow.
    pub grid: Option<FarGrid>,
    /// Where the camera is.
    pub camera: [f32; 3],
    /// Seconds since the world started.
    pub elapsed: f32,
    /// The fewest frames between two redraws.
    pub refresh_frames: u32,
}

/// When the far view redraws. One per renderer; `next` advances it a frame.
#[derive(Debug, Default)]
pub(crate) struct FarShadowScheduler {
    // The last draw; `None` until the view is primed.
    drawn: Option<DrawnFar>,
    // Frames since the last draw.
    frames: u32,
}

#[derive(Clone, Copy, Debug)]
struct DrawnFar {
    grid: FarGrid,
    center: [f32; 3],
    // When it was drawn, in seconds since the world started.
    at: f32,
}

impl FarShadowScheduler {
    /// Advance one frame.
    pub(crate) fn next(&mut self, tick: FarTick) -> FarFrame {
        let Some(grid) = tick.grid else {
            self.drawn = None;
            return FarFrame::Off;
        };
        self.frames = self.frames.saturating_add(1);
        if !redraw_due(self.drawn.as_ref(), &grid, &tick, self.frames) {
            return FarFrame::Keep;
        }
        let center = grid.snap(tick.camera);
        self.drawn = Some(DrawnFar {
            grid,
            center,
            at: tick.elapsed,
        });
        self.frames = 0;
        FarFrame::Draw(grid.lanes(center))
    }

    /// Redraw on the next frame, as before the view was ever drawn.
    pub(crate) fn reprime(&mut self) {
        self.drawn = None;
    }

    /// Carry the last draw into the frame the world moved to, so the move
    /// alone does not redraw the map.
    pub(crate) fn rebase(&mut self, rebase: &Rebase) {
        let Some(drawn) = &mut self.drawn else {
            return;
        };
        let grid = drawn.grid;
        match FarGrid::new(
            rebase.apply_vector(grid.to_light),
            grid.reach,
            grid.map_size,
        ) {
            Some(carried) => {
                drawn.grid = carried;
                drawn.center = rebase.apply_point(drawn.center);
            }
            None => self.drawn = None,
        }
    }
}

// Whether the far view redraws this frame: always to prime it, and once
// `refresh_frames` have passed since the last draw, when the camera strayed
// past the re-centering distance, the light turned, the coverage or the map
// changed, or the idle beat came round.
fn redraw_due(drawn: Option<&DrawnFar>, fresh: &FarGrid, tick: &FarTick, frames: u32) -> bool {
    let Some(drawn) = drawn else {
        return true;
    };
    if frames < tick.refresh_frames.max(1) {
        return false;
    }
    let grid = &drawn.grid;
    let idle = tick.elapsed - drawn.at;
    let strayed = FAR_RECENTER_FRACTION * grid.reach;
    !(0.0..FAR_IDLE_REFRESH_SECONDS).contains(&idle)
        || grid.reach.to_bits() != fresh.reach.to_bits()
        || grid.map_size != fresh.map_size
        || grid.turned_from(fresh.to_light)
        || length_sq(sub(tick.camera, drawn.center)) > strayed * strayed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::mat4_inverse;
    use alloc::vec::Vec;
    use concinnity_core::math::{cos, quat_from_axis_angle, sin};

    const LIGHTS: [[f32; 3]; 4] = [
        [0.3, 0.8, -0.5],
        [0.0, 1.0, 0.0],
        [-0.9, 0.12, 0.3],
        [0.05, 0.99, 0.1],
    ];

    const VIEW: CascadeView = CascadeView {
        near: 0.1,
        fov_y_rad: 1.2,
        aspect: 16.0 / 9.0,
        view_distance: None,
    };

    // Frames at 60 Hz.
    const DT: f32 = 1.0 / 60.0;

    fn grid(to_light: [f32; 3]) -> FarGrid {
        FarGrid::new(to_light, 4000.0, 2048).unwrap()
    }

    fn tick(grid: Option<FarGrid>, camera: [f32; 3], frame: u32, refresh: u32) -> FarTick {
        FarTick {
            grid,
            camera,
            elapsed: frame as f32 * DT,
            refresh_frames: refresh,
        }
    }

    fn lanes_at(g: &FarGrid, camera: [f32; 3]) -> FarShadowLanes {
        g.lanes(g.snap(camera))
    }

    fn clip(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
        let c: [f32; 4] =
            core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i]);
        [c[0] / c[3], c[1] / c[3], c[2] / c[3]]
    }

    fn center_of(lanes: &FarShadowLanes) -> [f32; 3] {
        [lanes.center[0], lanes.center[1], lanes.center[2]]
    }

    fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
        sqrt(length_sq(sub(a, b)))
    }

    fn is_draw(frame: FarFrame) -> bool {
        matches!(frame, FarFrame::Draw(_))
    }

    fn primed(g: FarGrid, camera: [f32; 3]) -> FarShadowScheduler {
        let mut scheduler = FarShadowScheduler::default();
        assert!(is_draw(scheduler.next(tick(Some(g), camera, 0, 1))));
        scheduler
    }

    // How much of the far shadow remains `distance` from its center: the
    // reference for `far_coverage` in `shadow_cascade.hlsl`.
    fn far_coverage(lanes: &FarShadowLanes, distance: f32) -> f32 {
        let reach = lanes.center[3];
        if reach <= 0.0 {
            return 0.0;
        }
        ((reach - distance) / (reach - lanes.fade_start).max(1e-4)).clamp(0.0, 1.0)
    }

    #[test]
    fn a_far_view_needs_a_reach_and_a_map() {
        assert!(FarGrid::new([0.0, 1.0, 0.0], 0.0, 2048).is_none());
        assert!(FarGrid::new([0.0, 1.0, 0.0], 4000.0, 0).is_none());
        assert!(FarGrid::new([0.0, 1.0, 0.0], 4000.0, 2048).is_some());
    }

    // Every receiver the far shadow covers around the camera lands in the map,
    // and so does a caster up the light from it.
    #[test]
    fn the_view_holds_its_coverage_and_the_casters_above_it() {
        let camera = [137.0, 812.0, -2240.0];
        for light in LIGHTS {
            let g = grid(light);
            let lanes = lanes_at(&g, camera);
            let center = center_of(&lanes);
            assert!(distance(center, camera) <= g.texel());
            for (i, dir) in [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.6, 0.0, 0.8],
                [-0.5, 0.5, 0.7],
            ]
            .into_iter()
            .enumerate()
            {
                let receiver = add(center, scale(normalize_clamped(dir, 1e-6), 3990.0));
                let caster = add(receiver, scale(g.to_light, 3900.0));
                for p in [receiver, caster] {
                    let ndc = clip(lanes.light_vp, p);
                    assert!(
                        ndc[0].abs() <= 1.0 && ndc[1].abs() <= 1.0,
                        "{light:?} {i}: {ndc:?}"
                    );
                    assert!((0.0..=1.0).contains(&ndc[2]), "{light:?} {i}: {ndc:?}");
                }
                let below = clip(lanes.light_vp, receiver)[2];
                let above = clip(lanes.light_vp, caster)[2];
                assert!(above > below, "nearer the light is nearer in the map");
            }
        }
    }

    // A fixed world point keeps its place within a texel wherever the map is
    // centered, so the far shadow's texels never crawl from one draw to the
    // next.
    #[test]
    fn texels_stay_anchored_as_the_camera_moves() {
        let g = grid([-0.4, 0.78, 0.5]);
        let p = [310.0, 4.0, -95.0];
        let texel_of = |camera: [f32; 3]| {
            let ndc = clip(lanes_at(&g, camera).light_vp, p);
            [ndc[0] * 1024.0, ndc[1] * 1024.0]
        };
        let a = texel_of([0.0, 2.0, 0.0]);
        for camera in [[13.7, 2.0, 9.1], [-260.0, 40.0, 75.0], [2.1, 2.0, -0.6]] {
            let b = texel_of(camera);
            for k in 0..2 {
                let d = a[k] - b[k];
                assert!((d - round(d)).abs() < 0.02, "{camera:?}: {d}");
            }
        }
    }

    // Every corner of the light box of a view drawn up to the re-centering
    // distance away lies within the caster reach of the camera, and the reach
    // is tight.
    #[test]
    fn every_box_corner_lies_within_the_caster_reach() {
        let reach = FarShadowReach {
            distance: 4000.0,
            map_size: 2048,
        }
        .caster_reach(VIEW);
        let mut farthest = 0.0_f32;
        for camera in [[0.0, 0.0, 0.0], [1.9, 700.0, -3.3], [-5000.0, 20.0, 12.0]] {
            for light in LIGHTS {
                let g = grid(light);
                for strayed in [[0.0; 3], [199.0, 0.0, 0.0], [0.0, -115.0, 115.0]] {
                    let drawn_from = add(camera, strayed);
                    let inv = mat4_inverse(lanes_at(&g, drawn_from).light_vp);
                    for corner in 0..8 {
                        let ndc = [
                            if corner & 1 == 0 { -1.0 } else { 1.0 },
                            if corner & 2 == 0 { -1.0 } else { 1.0 },
                            if corner & 4 == 0 { 0.0 } else { 1.0 },
                        ];
                        let d = distance(clip(inv, ndc), camera);
                        farthest = farthest.max(d);
                        assert!(d <= reach * 1.0001, "{d} > {reach}: {light:?} {camera:?}");
                    }
                }
            }
        }
        assert!(farthest > 0.98 * reach, "{farthest} of {reach}");
    }

    #[test]
    fn no_far_shadow_takes_no_casters() {
        let none = |distance, map_size| FarShadowReach { distance, map_size }.caster_reach(VIEW);
        assert_eq!(none(0.0, 2048), 0.0);
        assert_eq!(none(4000.0, 0), 0.0);
        assert!(none(4000.0, 1024) > 4000.0);
    }

    // The far view covers no farther than the camera sees, so it takes its
    // casters no farther either.
    #[test]
    fn the_caster_reach_stops_at_the_view_distance() {
        let far = FarShadowReach {
            distance: 4000.0,
            map_size: 2048,
        };
        let short = CascadeView {
            view_distance: Some(1000.0),
            ..VIEW
        };
        let shorter = FarShadowReach {
            distance: 1000.0,
            ..far
        };
        assert_eq!(far.caster_reach(short), shorter.caster_reach(VIEW));
        let beyond = CascadeView {
            view_distance: Some(9000.0),
            ..VIEW
        };
        assert_eq!(far.caster_reach(beyond), far.caster_reach(VIEW));
    }

    // The compare offset spans two texels along the light, in depth units.
    #[test]
    fn the_compare_offset_spans_two_texels() {
        let g = grid(LIGHTS[0]);
        let lanes = lanes_at(&g, [0.0; 3]);
        let center = center_of(&lanes);
        let step = clip(
            lanes.light_vp,
            add(center, scale(g.to_light, 2.0 * g.texel())),
        )[2] - clip(lanes.light_vp, center)[2];
        assert!(
            (step - lanes.compare_offset).abs() < 1e-3 * step,
            "{step} vs {lanes:?}"
        );
        assert!((lanes.texel - 1.0 / 2048.0).abs() < 1e-9);
    }

    #[test]
    fn coverage_is_whole_inside_the_fade_and_none_at_the_reach() {
        let lanes = lanes_at(&grid(LIGHTS[0]), [0.0; 3]);
        let start = lanes.fade_start;
        assert!((start - 3400.0).abs() < 1e-2, "{start}");
        assert_eq!(far_coverage(&lanes, 0.0), 1.0);
        assert_eq!(far_coverage(&lanes, start), 1.0);
        assert_eq!(far_coverage(&lanes, 4000.0), 0.0);
        assert_eq!(far_coverage(&lanes, 9000.0), 0.0);
        let mut last = 1.0;
        for i in 0..=32 {
            let c = far_coverage(&lanes, start + 600.0 * i as f32 / 32.0);
            assert!(c <= last, "{c} after {last}");
            last = c;
        }
        assert_eq!(far_coverage(&FarShadowLanes::OFF, 10.0), 0.0);
    }

    // A camera wandering within the re-centering distance, with the light
    // still, keeps the map until the idle beat, a second after the draw.
    #[test]
    fn a_camera_near_the_drawn_center_keeps_the_map_until_the_idle_beat() {
        let g = grid(LIGHTS[0]);
        let camera = [50.0, 3.0, 20.0];
        let mut scheduler = primed(g, camera);
        let beat = 60;
        for frame in 1..beat {
            let wander = [190.0 * sin(frame as f32 * 0.3), -2.0, 30.0];
            let kept = scheduler.next(tick(Some(g), add(camera, wander), frame, 1));
            assert_eq!(kept, FarFrame::Keep, "{frame}");
        }
        assert!(is_draw(scheduler.next(tick(Some(g), camera, beat, 1))));
    }

    // A clock that runs backward, as after the world restarts, redraws.
    #[test]
    fn a_clock_that_runs_backward_redraws() {
        let g = grid(LIGHTS[0]);
        let mut scheduler = FarShadowScheduler::default();
        scheduler.next(tick(Some(g), [0.0; 3], 120, 1));
        assert!(is_draw(scheduler.next(tick(Some(g), [0.0; 3], 3, 1))));
    }

    // A moving camera re-centers the map only once it strays past 5% of the
    // reach, no sooner than the tier's rate allows, and the map's whole
    // coverage always holds the ground around it.
    #[test]
    fn a_moving_camera_recenters_with_hysteresis() {
        let g = grid(LIGHTS[2]);
        let step = 30.0;
        for refresh in [1_u32, 2, 4, 8] {
            let mut camera = [0.0, 10.0, 0.0];
            let mut scheduler = primed(g, camera);
            let mut center = g.snap(camera);
            let mut draws = Vec::new();
            for frame in 1..=120_u32 {
                camera = add(camera, [step, 0.0, 0.0]);
                if let FarFrame::Draw(lanes) = scheduler.next(tick(Some(g), camera, frame, refresh))
                {
                    draws.push(frame);
                    center = center_of(&lanes);
                }
                let strayed = distance(camera, center);
                assert!(
                    strayed <= 200.0 + step * refresh as f32 + g.texel(),
                    "refresh {refresh}, frame {frame}: {strayed}"
                );
                assert!(4000.0 * (1.0 - FAR_FADE_FRACTION) - strayed > 3000.0);
            }
            // 200 m at 30 m a frame strays past it every 7 frames, unless the
            // tier allows a draw less often than that.
            let every = 7_u32.max(refresh);
            let expected: Vec<u32> = (1..=120).filter(|f| f % every == 0).collect();
            assert_eq!(draws, expected, "refresh {refresh}");
        }
    }

    // Any turn of the light redraws the map, at the tier's rate: at a low
    // sun a turn far under a texel's angle still moves long shadows by
    // several texels.
    #[test]
    fn the_light_turning_redraws_the_view() {
        let g = grid(LIGHTS[2]);
        let tilt = |a: f32| {
            let [x, y, z] = g.to_light;
            [x * cos(a) - y * sin(a), x * sin(a) + y * cos(a), z]
        };
        let mut scheduler = primed(g, [0.0; 3]);
        let still = FarGrid::new(tilt(0.2 * FAR_LIGHT_EPSILON), g.reach, g.map_size);
        assert_eq!(scheduler.next(tick(still, [0.0; 3], 1, 1)), FarFrame::Keep);
        let texel_angle = g.texel() / g.reach;
        let turned = FarGrid::new(tilt(0.05 * texel_angle), g.reach, g.map_size);
        assert!(is_draw(scheduler.next(tick(turned, [0.0; 3], 2, 1))));
        let flipped = FarGrid::new(scale(g.to_light, -1.0), g.reach, g.map_size);
        assert!(is_draw(scheduler.next(tick(flipped, [0.0; 3], 3, 1))));
    }

    // A turning light redraws no more often than the tier allows.
    #[test]
    fn a_turning_light_redraws_at_the_tier_rate() {
        let mut scheduler = primed(grid(LIGHTS[0]), [0.0; 3]);
        let mut draws = Vec::new();
        for frame in 1..=12_u32 {
            let a = frame as f32 * 1e-3;
            let g = FarGrid::new([sin(a), 0.8, cos(a)], 4000.0, 2048);
            if is_draw(scheduler.next(tick(g, [0.0; 3], frame, 4))) {
                draws.push(frame);
            }
        }
        assert_eq!(draws, [4, 8, 12]);
    }

    #[test]
    fn a_new_reach_or_map_redraws_the_view() {
        let mut scheduler = primed(grid(LIGHTS[0]), [0.0; 3]);
        let longer = FarGrid::new(LIGHTS[0], 5000.0, 2048);
        assert!(is_draw(scheduler.next(tick(longer, [0.0; 3], 1, 1))));
        let coarser = FarGrid::new(LIGHTS[0], 5000.0, 1024);
        assert!(is_draw(scheduler.next(tick(coarser, [0.0; 3], 2, 1))));
    }

    // Turning the far shadow off forgets the map, so turning it back on primes
    // it again, as does a reprime.
    #[test]
    fn off_and_reprime_both_prime_the_next_frame() {
        let g = grid(LIGHTS[0]);
        let mut scheduler = primed(g, [0.0; 3]);
        assert_eq!(scheduler.next(tick(None, [0.0; 3], 1, 4)), FarFrame::Off);
        assert!(is_draw(scheduler.next(tick(Some(g), [0.0; 3], 2, 4))));
        assert_eq!(
            scheduler.next(tick(Some(g), [0.0; 3], 3, 4)),
            FarFrame::Keep
        );
        scheduler.reprime();
        assert!(is_draw(scheduler.next(tick(Some(g), [0.0; 3], 4, 4))));
    }

    // A rebase moves the camera and turns the light by the frame's own move,
    // which leaves the drawn map where it was: nothing redraws.
    #[test]
    fn a_rebase_keeps_the_map() {
        let g = grid(LIGHTS[0]);
        let camera = [120.0, 30.0, -40.0];
        let mut scheduler = primed(g, camera);
        let rebase = Rebase {
            rotation: quat_from_axis_angle([0.0, 1.0, 0.0], 0.2),
            translation: [-900.0, 4.0, 10.0],
        };
        scheduler.rebase(&rebase);
        let carried = FarGrid::new(rebase.apply_vector(g.to_light), g.reach, g.map_size);
        let moved = rebase.apply_point(camera);
        for frame in 1..30 {
            assert_eq!(
                scheduler.next(tick(carried, moved, frame, 1)),
                FarFrame::Keep,
                "{frame}"
            );
        }
        let mut stale = primed(g, camera);
        assert!(is_draw(stale.next(tick(carried, moved, 1, 1))));
    }
}
