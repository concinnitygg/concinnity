//! Which shadow slices and views re-render each frame, and the projections
//! they render through.

use crate::components::ShadowUpdate;
use crate::environment::planet::Rebase;
use crate::gfx::frustum::{Aabb, Frustum};
use crate::gfx::render_types::{
    CloudShadowLanes, FarShadowLanes, NUM_SHADOW_CASCADES, ShadowUniforms,
};
use crate::render::backend::init::{ShadowCadence, ShadowMapSizes};
use crate::render::environment::grass::shadow::{
    GrassShadowView, grass_cascades, grass_receiver_depth,
};
use crate::render::planar_reflection::MirrorView;
use crate::render::shadow::csm::{cascade_blend_in, empty_shadow_uniforms};
use crate::render::shadow::far::{FarFrame, FarGrid, FarShadowScheduler, FarTick};
use crate::render::shadow::motion::{MovedCasters, ShadowSweep};
use crate::render::shadow::receivers::{
    ReceiverCorners, ReceiverCut, ShadowReader, ShadowReaders, cast_box,
};
use crate::render::shadow::schedule::{
    CascadeCamera, CascadeLight, ShadowCascadeScheduler, fit_cascades, shadow_distance,
};
use crate::render::shadow::spot::SpotShadowScheduler;
use concinnity_core::math::vec3::normalize_clamped;

/// One frame's inputs to the shadow schedules.
#[derive(Clone, Copy, Debug)]
pub struct ShadowFrame {
    /// The shadow maps the renderer built.
    pub maps: ShadowMapSizes,
    /// Unit vector toward the shadowed light.
    pub dir_to_source: [f32; 3],
    /// The camera the sun's shadows are fit to.
    pub camera: CascadeCamera,
    /// Seconds since the world started.
    pub elapsed: f32,
    /// How many spot lights cast shadows.
    pub spots: u32,
}

/// The cascade, far and spot shadow schedules and the sun shadows' carried
/// uniforms.
pub struct ShadowPlan {
    /// The live schedule.
    pub cadence: ShadowCadence,
    /// The cascade and far projections: a cascade skipped this frame keeps the
    /// one its slice was last rendered with, and the far view the one it was
    /// last drawn through, so the shaded passes sample each consistently. The
    /// empty set while shadows are off.
    pub uniforms: ShadowUniforms,
    /// The cascades re-rendering this frame, one bit each.
    pub cascade_mask: u32,
    /// The re-rendering cascades a world's grass casts into, one bit each.
    pub grass_mask: u32,
    /// Whether the far view redraws this frame.
    pub far_draw: bool,
    /// The spot slices re-rendering this frame, one bit each.
    pub spot_mask: u32,
    // Where casters moved since the last refresh, carried by any rebase.
    moved: MovedCasters,
    // Where spawns placed casters since the last refresh, as given.
    placed: MovedCasters,
    // The cascades this frame draws for itself alone, one bit each: the
    // schedule redraws them on every frame, so none is sampled later.
    every_frame: u32,
    // The views shading with the cascades this frame, whose receivers a cascade
    // drawn for this frame alone holds the casters of.
    readers: ShadowReaders,
    // Texels across each cascade's slice of the map.
    cascade_map_size: u32,
    cascades: ShadowCascadeScheduler,
    far: FarShadowScheduler,
    spots: SpotShadowScheduler,
}

impl ShadowPlan {
    /// A schedule following `cadence`, with nothing rendered yet.
    pub fn new(cadence: ShadowCadence) -> Self {
        Self {
            cadence,
            uniforms: empty_shadow_uniforms(),
            cascade_mask: 0,
            grass_mask: 0,
            far_draw: false,
            spot_mask: 0,
            moved: MovedCasters::default(),
            placed: MovedCasters::default(),
            every_frame: 0,
            readers: ShadowReaders::default(),
            cascade_map_size: 0,
            cascades: ShadowCascadeScheduler::default(),
            far: FarShadowScheduler::default(),
            spots: SpotShadowScheduler::default(),
        }
    }

    /// The cascades this frame's grass casts into, lit from `to_light`, with
    /// blades up to `tallest` meters high, or `None` when no blade casts onto
    /// anything the cascades shade. When every one of them is drawn for this
    /// frame alone, the blades are only those whose shadows can land in view;
    /// otherwise a kept cascade may be sampled after the camera turns, so all
    /// of its blades cast.
    pub fn grass_view(&self, to_light: [f32; 3], tallest: f32) -> Option<GrassShadowView> {
        let mut corners: ReceiverCorners = [[0.0; 3]; _];
        let cut = (self.grass_mask & !self.every_frame == 0).then(|| {
            let depth = grass_receiver_depth(&self.uniforms, to_light, tallest);
            let n = self.readers.corners(0.0, depth, &mut corners);
            ReceiverCut {
                receivers: &corners[..n],
                map_size: self.cascade_map_size,
            }
        });
        GrassShadowView::new(&self.uniforms, self.grass_mask, to_light, cut)
    }

    /// Note the prop reflection probe capturing this frame from `center`, if
    /// any: it shades with the cascades too.
    pub fn set_probe(&mut self, center: Option<[f32; 3]>) {
        self.readers
            .set_probe(center.map(|center| ShadowReader::Cube { center }));
    }

    /// Note the planar mirrors rendering this frame: they shade with the
    /// cascades too. Called before the cascades are culled.
    pub fn set_mirrors(&mut self, mirrors: impl IntoIterator<Item = MirrorView>) {
        self.readers.set_mirrors(
            mirrors.into_iter().map(|m| {
                ShadowReader::through(&m.matrices.view, &m.crop_view_proj, m.matrices.eye)
            }),
        );
    }

    /// Where this frame's cloud shadows lie, or [`CloudShadowLanes::OFF`].
    pub fn set_clouds(&mut self, clouds: CloudShadowLanes) {
        self.uniforms.clouds = clouds;
    }

    /// Carry the cascade and far projections into the frame the world moved
    /// to, so a slice or a far map kept after the move is sampled where it was
    /// drawn.
    pub fn rebase(&mut self, rebase: &Rebase) {
        for vp in &mut self.uniforms.light_vps {
            *vp = rebase.reproject(*vp);
        }
        let far = &mut self.uniforms.far;
        if far.center[3] > 0.0 {
            far.light_vp = rebase.reproject(far.light_vp);
            let [x, y, z, reach] = far.center;
            let [x, y, z] = rebase.apply_point([x, y, z]);
            far.center = [x, y, z, reach];
        }
        self.far.rebase(rebase);
        self.moved.rebase(rebase);
        self.readers.rebase(rebase);
    }

    /// Note that the casters inside the world-space box `bounds` changed, in
    /// the frame the plan is in now, so a far map that reaches them redraws
    /// and the cascades they can shadow something in view in redraw on the
    /// next refresh. `None` is a change anywhere.
    pub fn casters_changed(&mut self, bounds: Option<Aabb>) {
        self.far.casters_changed(bounds);
        self.moved.push(bounds);
    }

    /// Note a caster spawned, shown or hidden at the world-space box `bounds`
    /// this frame (`None` for anywhere). The box comes from the model the
    /// frame's ticks left it at, which a rebase among those ticks may already
    /// have moved into the frame the plan's rebase moves the world to; it is
    /// tested both carried by the rebase and as given.
    pub fn casters_placed(&mut self, bounds: Option<Aabb>) {
        self.casters_changed(bounds);
        self.placed.push(bounds);
    }

    /// Note that a caster moved through the world-space box `bounds` (`None`
    /// for anywhere), in the frame the plan is in now. Each cascade it can cast
    /// a shadow from onto what that cascade's views see redraws on the next
    /// refresh, so the shadow keeps up with it.
    pub fn casters_moved(&mut self, bounds: Option<Aabb>) {
        self.moved.push(bounds);
    }

    /// Render every cascade and the far view on the next frame, for a view
    /// that does not continue the last: a kept slice or far map would cover
    /// where the camera was.
    pub fn reprime(&mut self) {
        self.cascades.reprime();
        self.far.reprime();
    }

    /// Start the spot slices over, so each re-primes before it is sampled.
    pub fn reset_spots(&mut self) {
        self.spots = SpotShadowScheduler::default();
    }

    /// Advance one frame: the cascades and the far view of the shadow maps the
    /// renderer built, and the shadowed spots.
    pub fn refresh(&mut self, frame: ShadowFrame) {
        self.readers.set_camera(frame.camera.reader());
        if frame.maps.cascades > 0 {
            let light = CascadeLight {
                dir_to_source: frame.dir_to_source,
                map_size: frame.maps.cascades,
            };
            let fresh = fit_cascades(&self.cadence, light, &frame.camera);
            // Every cascade redraws each frame anyway under `EveryFrame`.
            let moved = match self.cadence.update {
                ShadowUpdate::EveryFrame => 0,
                ShadowUpdate::Hybrid => self.moved_cascades(&fresh, frame.dir_to_source),
            };
            let redraw = self
                .cascades
                .refresh(&mut self.uniforms, &self.cadence, &fresh, moved);
            self.cascade_mask = redraw.mask;
            self.every_frame = redraw.every_frame;
            self.grass_mask = self.cascade_mask & grass_cascades(&self.uniforms);
            self.cascade_map_size = frame.maps.cascades;
        }
        self.moved.clear();
        self.placed.clear();
        // The mirrors rendering this frame are set again once the backend
        // knows them; last frame's served the motion above.
        self.readers.set_mirrors([]);
        self.refresh_far(&frame);
        let every_frame = matches!(self.cadence.update, ShadowUpdate::EveryFrame);
        self.spot_mask = self.spots.next_mask(every_frame, frame.spots as usize);
    }

    /// The frustum cascade `cascade`'s casters are culled against this frame.
    /// One drawn for this frame alone is only sampled for what this frame's
    /// views see of its depth band, so it holds only their casters; one the
    /// schedule keeps may be sampled after the camera turns, so it holds every
    /// caster its light box reaches.
    pub fn cascade_cull(&self, cascade: usize) -> Frustum {
        let vp = self.uniforms.light_vps[cascade];
        if (self.cascade_mask & self.every_frame) & (1 << cascade) == 0 {
            return Frustum::from_shadow(vp);
        }
        let splits = &self.uniforms.cascade_splits;
        let mut corners: ReceiverCorners = [[0.0; 3]; _];
        let n = self.readers.corners(
            cascade_blend_in(splits, cascade),
            splits[cascade],
            &mut corners,
        );
        let cut = ReceiverCut {
            receivers: &corners[..n],
            map_size: self.cascade_map_size,
        };
        Frustum::from_shadow(cast_box([vp], Some(cut)).unwrap_or(vp))
    }

    // The cascades of this frame's fit `fresh` a caster that moved casts a
    // shadow in that their views can see, one bit each: it lies in the light
    // box the cascade was last drawn through, and the space its shadow can
    // fall in meets the cascade's depth band of the camera's view now, or the
    // probe's or a mirror's last frame.
    fn moved_cascades(&self, fresh: &ShadowUniforms, to_light: [f32; 3]) -> u32 {
        let active = (fresh.active_cascades as usize).min(NUM_SHADOW_CASCADES);
        if self.moved.anywhere() || self.placed.anywhere() {
            return (1 << active) - 1;
        }
        let to_light = normalize_clamped(to_light, 1e-6);
        let splits = &fresh.cascade_splits;
        let held: [Frustum; NUM_SHADOW_CASCADES] =
            core::array::from_fn(|c| Frustum::from_shadow(self.uniforms.light_vps[c]));
        let mut mask = 0;
        for ((lo, hi), height) in self.moved.boxes().chain(self.placed.boxes()) {
            let sweep = ShadowSweep::new((lo, hi), height, to_light);
            for (c, box_c) in held.iter().enumerate().take(active) {
                if mask & (1 << c) == 0
                    && box_c.intersects_aabb(lo, hi)
                    && self
                        .readers
                        .may_receive(cascade_blend_in(splits, c), splits[c], &sweep)
                {
                    mask |= 1 << c;
                }
            }
        }
        mask
    }

    // Advance the far view one frame. It runs only beside the cascades it
    // takes over from, and covers no farther than the camera sees.
    fn refresh_far(&mut self, frame: &ShadowFrame) {
        let camera = &frame.camera;
        let reach = shadow_distance(self.cadence.far_distance, camera.view_distance);
        let grid = if frame.maps.cascades > 0 {
            FarGrid::new(frame.dir_to_source, reach, frame.maps.far)
        } else {
            None
        };
        let far = self.far.next(FarTick {
            grid,
            camera: camera.position,
            elapsed: frame.elapsed,
            refresh_frames: self.cadence.far_refresh_frames,
        });
        self.far_draw = match far {
            FarFrame::Off => {
                self.uniforms.far = FarShadowLanes::OFF;
                false
            }
            FarFrame::Keep => false,
            FarFrame::Draw(lanes) => {
                self.uniforms.far = lanes;
                true
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::IDENTITY;
    use alloc::vec::Vec;

    fn frame(
        maps: ShadowMapSizes,
        dir_to_source: [f32; 3],
        camera: CascadeCamera,
        spots: u32,
    ) -> ShadowFrame {
        ShadowFrame {
            maps,
            dir_to_source,
            camera,
            elapsed: 0.0,
            spots,
        }
    }

    fn camera() -> CascadeCamera {
        CascadeCamera {
            view: IDENTITY,
            position: [0.0; 3],
            fov_y_rad: 1.0,
            aspect: 1.0,
            near: 0.1,
            view_distance: None,
        }
    }

    fn cadence() -> ShadowCadence {
        ShadowCadence {
            update: ShadowUpdate::EveryFrame,
            distance: 100,
            cascades: 4,
            far_distance: 0,
            far_refresh_frames: 2,
        }
    }

    const CASCADES: ShadowMapSizes = ShadowMapSizes {
        cascades: 1024,
        far: 0,
    };

    const WITH_FAR: ShadowMapSizes = ShadowMapSizes {
        cascades: 1024,
        far: 1024,
    };

    fn far_cadence() -> ShadowCadence {
        ShadowCadence {
            far_distance: 3000,
            ..hybrid()
        }
    }

    #[test]
    fn without_a_map_the_cascades_stay_empty_and_unrendered() {
        let mut plan = ShadowPlan::new(cadence());
        plan.refresh(frame(
            ShadowMapSizes::default(),
            [0.0, 1.0, 0.0],
            camera(),
            0,
        ));
        assert_eq!(plan.cascade_mask, 0);
        assert_eq!(
            plan.uniforms.active_cascades,
            empty_shadow_uniforms().active_cascades
        );
    }

    fn hybrid() -> ShadowCadence {
        ShadowCadence {
            update: ShadowUpdate::Hybrid,
            ..cadence()
        }
    }

    // A slice skipped after a rebase projects a carried point where its
    // render saw the point before the move.
    #[test]
    fn a_rebase_carries_the_skipped_cascades() {
        let mut plan = ShadowPlan::new(hybrid());
        plan.refresh(frame(CASCADES, [0.3, 1.0, 0.2], camera(), 0));
        let drawn = plan.uniforms.light_vps;
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([1.0, 0.0, 0.0], 0.02),
            translation: [-900.0, 4.0, 10.0],
        };
        plan.rebase(&rebase);
        plan.refresh(frame(CASCADES, [0.3, 1.0, 0.2], camera(), 0));
        assert_eq!(
            plan.cascade_mask, 0b0101,
            "the first cascade plus one in rotation"
        );
        let clip = |m: [[f32; 4]; 4], p: [f32; 3]| -> [f32; 4] {
            core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
        };
        let p = [3.0, -1.0, -20.0];
        let cascade = 3;
        let before = clip(drawn[cascade], p);
        let after = clip(plan.uniforms.light_vps[cascade], rebase.apply_point(p));
        for i in 0..4 {
            assert!((before[i] - after[i]).abs() < 1e-2, "{before:?} {after:?}");
        }
    }

    fn looking(position: [f32; 3], target: [f32; 3]) -> CascadeCamera {
        CascadeCamera {
            view: crate::gfx::projection::look_at(position, target, [0.0, 1.0, 0.0]),
            position,
            fov_y_rad: 1.0,
            aspect: 16.0 / 9.0,
            ..camera()
        }
    }

    fn inside(f: &Frustum, p: [f32; 3]) -> bool {
        f.planes.iter().all(|pl| {
            pl.normal[0] * p[0] + pl.normal[1] * p[1] + pl.normal[2] * p[2] + pl.d >= -1e-3
        })
    }

    // Points through a box between corners `lo` and `hi`.
    fn grid(lo: [f32; 3], hi: [f32; 3]) -> impl Iterator<Item = [f32; 3]> {
        (0..64).map(move |i| {
            let f = |k: usize, axis: usize| lo[axis] + (hi[axis] - lo[axis]) * (k as f32 / 3.0);
            [f(i % 4, 0), f((i / 4) % 4, 1), f(i / 16, 2)]
        })
    }

    // Every point of cascade `c`'s light box toward the light from one of
    // `receivers` is kept by its cull.
    fn assert_keeps_casters(plan: &ShadowPlan, c: usize, receivers: &[[f32; 3]], toward: [f32; 3]) {
        let whole = Frustum::from_shadow(plan.uniforms.light_vps[c]);
        let cull = plan.cascade_cull(c);
        for &r in receivers {
            for k in 0..16 {
                let s = k as f32 * 8.0;
                let p = core::array::from_fn(|i| r[i] + toward[i] * s);
                if inside(&whole, p) {
                    assert!(inside(&cull, p), "cascade {c} {p:?}");
                }
            }
        }
    }

    // Point `i` of a 3x3x3 lattice through a view's slice.
    fn in_slice(corners: &[[f32; 3]; 8], i: usize) -> [f32; 3] {
        let f = |k: usize| k as f32 * 0.5;
        let (u, v, t) = (f(i % 3), f((i / 3) % 3), f(i / 9));
        let lerp = |a: [f32; 3], b: [f32; 3], w: f32| -> [f32; 3] {
            core::array::from_fn(|k| a[k] + (b[k] - a[k]) * w)
        };
        let quad = |q: &[[f32; 3]]| lerp(lerp(q[0], q[1], u), lerp(q[3], q[2], u), v);
        lerp(quad(&corners[..4]), quad(&corners[4..]), t)
    }

    const TO_LIGHT: [f32; 3] = [0.3, 0.6, 0.2];

    fn toward() -> [f32; 3] {
        normalize_clamped(TO_LIGHT, 1e-6)
    }

    // A cascade drawn every frame culls its casters to the receivers it
    // shades: every point of its light box toward the light from a receiver
    // in view stays, and some of the box goes.
    #[test]
    fn a_cascade_drawn_every_frame_keeps_every_caster_of_its_receivers() {
        let cam = looking([0.0, 2.0, 0.0], [0.0, 1.5, -10.0]);
        let mut plan = ShadowPlan::new(cadence());
        plan.refresh(frame(CASCADES, TO_LIGHT, cam, 0));
        let splits = plan.uniforms.cascade_splits;
        for c in 0..4 {
            let whole = Frustum::from_shadow(plan.uniforms.light_vps[c]);
            assert_ne!(
                plan.cascade_cull(c).packed_planes(),
                whole.packed_planes(),
                "{c}"
            );
            let corners = cam
                .reader()
                .corners(cascade_blend_in(&splits, c), splits[c]);
            let receivers: Vec<_> = (0..27).map(|i| in_slice(&corners, i)).collect();
            assert_keeps_casters(&plan, c, &receivers, toward());
        }
    }

    // The probe and the mirrors shade with the cascades too, so a cut cascade
    // keeps the casters of what they see of its band.
    #[test]
    fn a_cut_cascade_keeps_the_casters_of_every_reader() {
        let cam = looking([0.0, 2.0, 0.0], [0.0, 1.5, -10.0]);
        let mut plan = ShadowPlan::new(cadence());
        plan.refresh(frame(CASCADES, TO_LIGHT, cam, 0));
        let probe = [0.0, 1.0, 6.0];
        plan.set_probe(Some(probe));
        let proj = crate::render::depth::camera_projection(1.0, 16.0 / 9.0, 0.1);
        let mirror_frame = crate::render::planar_reflection::MirrorFrame::new(
            cam.view,
            crate::transform::mat4_mul(proj, cam.view),
            cam.position,
            None,
            (64, 36),
        );
        let crop = crate::render::planar_reflection::PixelRect {
            x: 0,
            y: 0,
            width: 64,
            height: 36,
        };
        let mirror = MirrorView::new(&mirror_frame, [0.0, 1.0, 0.0, 0.0], crop);
        let mirror_reader = ShadowReader::through(
            &mirror.matrices.view,
            &mirror.crop_view_proj,
            mirror.matrices.eye,
        );
        plan.set_mirrors([mirror]);
        let splits = plan.uniforms.cascade_splits;
        for c in 0..4 {
            let far = splits[c];
            let near = cascade_blend_in(&splits, c);
            let cube: Vec<_> = grid(
                core::array::from_fn(|a| probe[a] - far),
                core::array::from_fn(|a| probe[a] + far),
            )
            .collect();
            assert_keeps_casters(&plan, c, &cube, toward());
            let corners = mirror_reader.corners(near, far);
            assert_keeps_casters(&plan, c, &corners, toward());
        }
    }

    // A cascade the schedule may keep is sampled after the camera turns, so it
    // keeps every caster its light box holds.
    #[test]
    fn a_cascade_that_may_be_kept_culls_against_its_whole_box() {
        let mut plan = ShadowPlan::new(hybrid());
        let cam = looking([0.0, 2.0, 0.0], [0.0, 1.5, -10.0]);
        let mut every_frame = u32::MAX;
        let mut culls = Vec::new();
        for _ in 0..4 {
            plan.refresh(frame(CASCADES, [0.3, 1.0, 0.2], cam, 0));
            every_frame &= plan.cascade_mask;
            let frame_culls: [_; 4] = core::array::from_fn(|c| {
                (
                    plan.cascade_cull(c).packed_planes(),
                    Frustum::from_shadow(plan.uniforms.light_vps[c]).packed_planes(),
                )
            });
            culls.push((plan.cascade_mask, frame_culls));
        }
        assert_ne!(every_frame, 0b1111, "some cascades rotate");
        for (mask, frame_culls) in culls {
            for (c, (cull, whole)) in frame_culls.iter().enumerate() {
                if mask & !every_frame & (1 << c) != 0 {
                    assert_eq!(cull, whole, "cascade {c}");
                }
            }
        }
    }

    // A kept cascade's grass casts from every blade its box holds, so the
    // view's blades are cut only when every casting cascade is drawn for the
    // frame alone.
    #[test]
    fn the_grass_is_cut_only_when_every_casting_cascade_is_drawn_alone() {
        let cam = looking([0.0, 2.0, 0.0], [0.0, 1.5, -10.0]);
        for (cadence, cut) in [(cadence(), true), (hybrid(), false)] {
            let mut plan = ShadowPlan::new(cadence);
            plan.refresh(frame(CASCADES, TO_LIGHT, cam, 0));
            plan.refresh(frame(CASCADES, TO_LIGHT, cam, 0));
            let view = plan.grass_view(TO_LIGHT, 1.0).unwrap();
            let whole =
                GrassShadowView::new(&plan.uniforms, plan.grass_mask, TO_LIGHT, None).unwrap();
            assert_eq!(view != whole, cut, "{:?}", plan.cadence.update);
        }
    }

    // Prime a Hybrid plan's cascades and settle it into its rotation.
    fn settled(cam: CascadeCamera) -> ShadowPlan {
        let mut plan = ShadowPlan::new(hybrid());
        for _ in 0..4 {
            plan.refresh(frame(CASCADES, TO_LIGHT, cam, 0));
        }
        plan
    }

    // A caster that moves on the ground in view redraws, on the next frame,
    // the cascade that shades the ground there, so its shadow keeps up with
    // it; once still, the rotation resumes.
    #[test]
    fn a_caster_moving_in_view_redraws_the_cascade_shading_it() {
        let cam = looking([0.0, 2.0, 0.0], [0.0, 1.5, -10.0]);
        let mut plan = settled(cam);
        let splits = plan.uniforms.cascade_splits;
        assert!(splits[0] < 5.0 && splits[1] > 7.0, "{splits:?}");
        for x in 0..4 {
            let at = [x as f32 * 0.1, 0.0, -6.0];
            plan.casters_moved(Some((at, [at[0] + 0.6, 0.6, at[2] + 0.6])));
            plan.refresh(frame(CASCADES, TO_LIGHT, cam, 0));
            assert_eq!(plan.cascade_mask & 0b11, 0b11, "{:#b}", plan.cascade_mask);
        }
        plan.refresh(frame(CASCADES, TO_LIGHT, cam, 0));
        assert_eq!(plan.cascade_mask.count_ones(), 2, "still again");
    }

    // A caster behind the camera, lit from in front, throws its shadow away
    // from everything in view, so no cascade redraws for it; a move anywhere
    // redraws them all.
    #[test]
    fn a_caster_whose_shadow_falls_out_of_view_redraws_nothing_extra() {
        let cam = looking([0.0, 2.0, 0.0], [0.0, 1.5, -10.0]);
        let to_light = [0.0, 0.6, -1.0];
        let mut plan = ShadowPlan::new(hybrid());
        for _ in 0..4 {
            plan.refresh(frame(CASCADES, to_light, cam, 0));
        }
        let behind = [0.0, 0.0, 30.0];
        let whole = Frustum::from_shadow(plan.uniforms.light_vps[3]);
        assert!(whole.intersects_aabb(behind, behind), "inside the last box");
        plan.casters_moved(Some((behind, [0.5, 0.5, 30.5])));
        plan.refresh(frame(CASCADES, to_light, cam, 0));
        assert_eq!(
            plan.cascade_mask.count_ones(),
            2,
            "{:#b}",
            plan.cascade_mask
        );
        plan.casters_moved(None);
        plan.refresh(frame(CASCADES, to_light, cam, 0));
        assert_eq!(plan.cascade_mask, 0b1111);
    }

    // A move noted before a rebase is carried with the world, so it still
    // redraws the cascade shading where the caster is.
    #[test]
    fn a_move_noted_before_a_rebase_is_carried_with_it() {
        let before = looking([900.0, 2.0, 0.0], [900.0, 1.5, -10.0]);
        let after = looking([0.0, 2.0, 0.0], [0.0, 1.5, -10.0]);
        let mut plan = settled(before);
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.0, 1.0, 0.0], 0.0),
            translation: [-900.0, 0.0, 0.0],
        };
        let at = [900.0, 0.0, -6.0];
        plan.casters_moved(Some((at, [900.6, 0.6, -5.4])));
        plan.rebase(&rebase);
        plan.refresh(frame(CASCADES, TO_LIGHT, after, 0));
        assert_eq!(plan.cascade_mask & 0b11, 0b11, "{:#b}", plan.cascade_mask);
    }

    // A caster a spawn places after the frame's rebase is already in the new
    // frame; the plan tests it as given too, so its cascade still redraws.
    #[test]
    fn a_caster_placed_after_a_rebase_redraws_its_cascade() {
        let before = looking([900.0, 2.0, 0.0], [900.0, 1.5, -10.0]);
        let after = looking([0.0, 2.0, 0.0], [0.0, 1.5, -10.0]);
        let mut plan = settled(before);
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.0, 1.0, 0.0], 0.0),
            translation: [-900.0, 0.0, 0.0],
        };
        plan.casters_placed(Some(([0.0, 0.0, -6.0], [0.6, 0.6, -5.4])));
        plan.rebase(&rebase);
        plan.refresh(frame(CASCADES, TO_LIGHT, after, 0));
        assert_eq!(plan.cascade_mask & 0b11, 0b11, "{:#b}", plan.cascade_mask);
    }

    #[test]
    fn a_reprime_renders_every_cascade_once() {
        let mut plan = ShadowPlan::new(hybrid());
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 0));
        assert_eq!(plan.cascade_mask, 0b1111);
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 0));
        assert_eq!(plan.cascade_mask.count_ones(), 2);
        plan.reprime();
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 0));
        assert_eq!(plan.cascade_mask, 0b1111);
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 0));
        assert_eq!(plan.cascade_mask.count_ones(), 2);
    }

    // The grass draws into the cascades near the camera, and only on the
    // frames that re-render them.
    #[test]
    fn the_grass_casts_into_the_near_cascades_that_re_render() {
        let mut plan = ShadowPlan::new(hybrid());
        plan.refresh(frame(CASCADES, [0.3, 1.0, 0.2], camera(), 0));
        let near = grass_cascades(&plan.uniforms);
        assert_eq!(plan.grass_mask, near);
        assert!(near.count_ones() > 1, "{near:#b}");
        for _ in 0..4 {
            plan.refresh(frame(CASCADES, [0.3, 1.0, 0.2], camera(), 0));
            assert_eq!(plan.grass_mask, plan.cascade_mask & near);
        }
        let mut off = ShadowPlan::new(hybrid());
        off.refresh(frame(
            ShadowMapSizes::default(),
            [0.3, 1.0, 0.2],
            camera(),
            0,
        ));
        assert_eq!(off.grass_mask, 0);
    }

    #[test]
    fn every_frame_renders_every_cascade_and_spot() {
        let mut plan = ShadowPlan::new(cadence());
        plan.refresh(frame(CASCADES, [0.0, 1.0, 0.0], camera(), 2));
        assert_eq!(plan.cascade_mask, 0b1111);
        assert_eq!(plan.spot_mask, 0b11);
        assert_eq!(plan.uniforms.active_cascades, 4);
    }

    #[test]
    fn without_a_far_distance_or_map_the_far_view_stays_off() {
        for (cadence, maps) in [
            (hybrid(), WITH_FAR),
            (far_cadence(), CASCADES),
            (
                far_cadence(),
                ShadowMapSizes {
                    cascades: 0,
                    far: 1024,
                },
            ),
        ] {
            let mut plan = ShadowPlan::new(cadence);
            plan.refresh(frame(maps, [0.3, 1.0, 0.2], camera(), 0));
            assert!(!plan.far_draw);
            assert_eq!(plan.uniforms.far, FarShadowLanes::OFF);
        }
    }

    // The far view draws on the frame it first covers anything, so it is
    // never sampled before it holds depth, and turning it off clears what the
    // shaded passes read.
    #[test]
    fn the_far_view_primes_before_it_is_sampled() {
        let mut plan = ShadowPlan::new(far_cadence());
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(plan.far_draw);
        assert_eq!(plan.uniforms.far.center[3], 3000.0);
        let drawn = plan.uniforms.far;
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(!plan.far_draw, "a still camera keeps the map");
        assert_eq!(plan.uniforms.far, drawn);
        plan.cadence.far_distance = 0;
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert_eq!(plan.uniforms.far, FarShadowLanes::OFF);
        plan.cadence.far_distance = 3000;
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(plan.far_draw);
    }

    #[test]
    fn the_far_view_covers_no_farther_than_the_camera_sees() {
        let mut plan = ShadowPlan::new(far_cadence());
        let near_sighted = CascadeCamera {
            view_distance: Some(1200.0),
            ..camera()
        };
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], near_sighted, 0));
        assert_eq!(plan.uniforms.far.center[3], 1200.0);
    }

    #[test]
    fn a_reprime_redraws_the_far_view() {
        let mut plan = ShadowPlan::new(far_cadence());
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(!plan.far_draw);
        plan.reprime();
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        assert!(plan.far_draw);
    }

    // A far map kept after a rebase projects a carried point where its draw
    // saw the point, and measures its coverage from the carried center.
    #[test]
    fn a_rebase_carries_the_far_view() {
        let mut plan = ShadowPlan::new(far_cadence());
        plan.refresh(frame(WITH_FAR, [0.3, 1.0, 0.2], camera(), 0));
        let drawn = plan.uniforms.far;
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.0, 1.0, 0.0], 0.3),
            translation: [-900.0, 4.0, 10.0],
        };
        plan.rebase(&rebase);
        let far = plan.uniforms.far;
        let clip = |m: [[f32; 4]; 4], p: [f32; 3]| -> [f32; 4] {
            core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
        };
        let p = [700.0, -12.0, -1500.0];
        let before = clip(drawn.light_vp, p);
        let after = clip(far.light_vp, rebase.apply_point(p));
        for i in 0..4 {
            assert!((before[i] - after[i]).abs() < 1e-3, "{before:?} {after:?}");
        }
        let center = rebase.apply_point([drawn.center[0], drawn.center[1], drawn.center[2]]);
        for (carried, want) in far.center.iter().zip(center) {
            assert!((carried - want).abs() < 1e-2, "{far:?}");
        }
        assert_eq!(far.center[3], drawn.center[3]);
        assert_eq!(
            (far.fade_start, far.compare_offset, far.texel),
            (drawn.fade_start, drawn.compare_offset, drawn.texel)
        );
        // The camera and light carried by the same move keep the map.
        let carried = CascadeCamera {
            position: rebase.apply_point(camera().position),
            ..camera()
        };
        let light = rebase.apply_vector([0.3, 1.0, 0.2]);
        plan.refresh(frame(WITH_FAR, light, carried, 0));
        assert!(!plan.far_draw);
    }
}
