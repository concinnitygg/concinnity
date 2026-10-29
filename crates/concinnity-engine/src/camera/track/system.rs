use alloc::vec::Vec;

use crate::camera::track::CameraTrackStatus;
use crate::camera::track::carry::{CarryBlend, Filmed};
use crate::camera::track::status::TrackPickup;
use crate::camera::track::timeline::{self, Key};
use crate::components::{Camera3D, CameraBoomKey, CameraTrack, CameraTurnKey, Parent, Transform};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, MenuActive, PipelineContext, SimTiming, StepResult, System, World};
use crate::gfx::camera::view_axes;
use crate::planet::{FrameRebases, RebaseCursor};
use crate::render::temporal::history_reset::{HistoryResetCauses, PendingHistoryReset};
use concinnity_core::math::{cos, quat_mul, quat_normalize, quat_rotate, shortest_turn, sin, vec3};

const FULL_TURN_DEG: f32 = 360.0;

/// Drives the world's [`Camera3D`] along its declared
/// [`CameraTrack`].
///
/// Runs on the fixed simulation clock rather than the frame delta, so the pose
/// at a given track time is the same on every machine and two runs of one
/// track are comparable. Each frame poses the camera between the last two
/// steps by [`SimTiming::alpha`], the way physics blends its bodies, so the
/// camera moves as far as the frame took rather than by whole steps. Frozen
/// while a world-pausing screen is open.
///
/// With a subject, travel carries the subject and the camera is placed from
/// it by the boom in the same step, at the same pose time, so the subject
/// holds its place in the frame exactly.
#[derive(Debug)]
pub(crate) struct CameraTrackSystem {
    travel: Vec<Key<3>>,
    // Held until `init`, which is the first point the heading the first leg
    // turns away from is known.
    turn_legs: Vec<CameraTurnKey>,
    turn: Vec<Key<2>>,
    // As the turn legs: held until `init` knows the framing the camera starts
    // in. Each key's value is the distance, then the offset right and up.
    boom_legs: Vec<CameraBoomKey>,
    boom: Vec<Key<3>>,
    boom_start: [f32; 3],
    // The prop travel carries, while it is there to carry.
    subject: Option<AssetId>,
    // Track time the subject is carried from, whether it is carried yet, how
    // it moved while it was only filmed, and the blend onto travel's path it
    // was picked up with.
    carry_from: f32,
    carrying: bool,
    filmed: Filmed,
    blend: Option<CarryBlend>,
    pickup: Option<TrackPickup>,
    // Fixed once every list is laid out at `init`, so the track runs as long
    // whether or not its subject stays.
    duration: f32,
    start_position: [f32; 3],
    start_angles_deg: [f32; 2],
    // Travel is laid out leg by leg from where the carried point is (the
    // subject, else the camera): `leg` is the leg under way, `anchor` where
    // the point stood when its progress was `anchor_progress` (the leg's
    // start, or where a frame move carried it mid-leg), and `frame_turn` the
    // turn every frame move so far gave the legs, which are authored in the
    // frame the world started in. Every offset the point rides is one leg's
    // at most, so a long track stays as precise as a short one.
    leg: usize,
    anchor: [f32; 3],
    anchor_progress: [f32; 3],
    frame_turn: concinnity_core::math::Quat,
    rebases: RebaseCursor,
    // Ticks are counted and multiplied rather than accumulated: a long run
    // stays free of float drift, and two hosts running the same track at
    // different frame rates land on exactly the same pose.
    ticks: u64,
    tick_dt: f32,
    // Track time the camera was last posed at. Legs are anchored against it
    // rather than the step count, since it is where the camera stands.
    pose_seconds: f32,
    camera: Option<Entity>,
}

impl CameraTrackSystem {
    /// The system for a world that declares both a `CameraTrack` and the
    /// `Camera3D` it drives.
    pub(crate) fn for_world(world: &World) -> Option<Self> {
        let track = world.query::<CameraTrack>().next()?;
        world.query::<Camera3D>().next()?;
        Some(Self::new(track))
    }

    /// The system for a world's baked track. The travel list is ready here;
    /// the turn and boom lists wait for `init` to supply the camera's starting
    /// heading and framing.
    pub(crate) fn new(track: &CameraTrack) -> Self {
        Self {
            travel: track
                .travel
                .iter()
                .map(|key| Key {
                    value: key.delta,
                    end_seconds: key.end_seconds,
                    ease: key.ease,
                    cut: key.cut,
                })
                .collect(),
            turn_legs: track.turn.clone(),
            turn: Vec::new(),
            boom_legs: track.boom.clone(),
            boom: Vec::new(),
            boom_start: [0.0; 3],
            subject: track.subject.map(crate::ecs::Ref::id),
            carry_from: track.carry_from_seconds,
            carrying: false,
            filmed: Filmed::default(),
            blend: None,
            pickup: None,
            duration: 0.0,
            start_position: [0.0; 3],
            start_angles_deg: [0.0; 2],
            leg: 0,
            anchor: [0.0; 3],
            anchor_progress: [0.0; 3],
            frame_turn: [0.0, 0.0, 0.0, 1.0],
            rebases: RebaseCursor::default(),
            ticks: 0,
            tick_dt: SimTiming::TICK_DT,
            pose_seconds: 0.0,
            camera: None,
        }
    }

    /// How long the whole track runs: whichever of its lists lasts longest,
    /// as laid out at `init`.
    pub(crate) fn duration(&self) -> f32 {
        self.duration
    }

    /// Seconds of simulated time the track has run for.
    pub(crate) fn elapsed(&self) -> f32 {
        self.ticks as f32 * self.tick_dt
    }

    // Lay the turn legs out on the clock, starting from the heading the camera
    // was authored at. Yaw is carried forward as a continuous angle so a leg
    // takes the shorter way round and the interpolation never wraps.
    fn resolve_turn(&mut self) {
        let [mut yaw, mut pitch] = self.start_angles_deg;
        let mut end_seconds = 0.0;
        self.turn.clear();
        self.turn.reserve(self.turn_legs.len());
        for leg in &self.turn_legs {
            let target_yaw = leg
                .yaw_deg
                .map_or(yaw, |to| yaw + shortest_turn(yaw, to, FULL_TURN_DEG));
            let target_pitch = leg.pitch_deg.unwrap_or(pitch);
            let sweep = (target_yaw - yaw).abs().max((target_pitch - pitch).abs());
            let seconds = turn_seconds(leg, sweep);
            end_seconds += seconds;
            self.turn.push(Key {
                value: [target_yaw, target_pitch],
                end_seconds,
                ease: leg.ease,
                cut: leg.cut || (seconds == 0.0 && sweep > 0.0),
            });
            yaw = target_yaw;
            pitch = target_pitch;
        }
    }

    // Lay the boom legs out from the framing the camera starts in: each value
    // a leg leaves unset holds, and until the first leg that sets a value the
    // boom holds that value.
    fn resolve_boom(&mut self, authored_distance: f32) {
        let distance = self
            .boom_legs
            .iter()
            .find_map(|leg| leg.distance)
            .unwrap_or(authored_distance);
        let offset = self
            .boom_legs
            .iter()
            .find_map(|leg| leg.offset_deg)
            .unwrap_or([0.0; 2]);
        self.boom_start = [distance, offset[0], offset[1]];
        let mut at = self.boom_start;
        let mut from_seconds = 0.0;
        self.boom.clear();
        self.boom.reserve(self.boom_legs.len());
        for leg in &self.boom_legs {
            let distance = leg.distance.unwrap_or(at[0]);
            let [right, up] = leg.offset_deg.unwrap_or([at[1], at[2]]);
            let value = [distance, right, up];
            let instant = leg.end_seconds <= from_seconds;
            self.boom.push(Key {
                value,
                end_seconds: leg.end_seconds,
                ease: leg.ease,
                cut: leg.cut || (instant && value != at),
            });
            at = value;
            from_seconds = leg.end_seconds;
        }
    }

    // Re-anchor on every leg the pose has finished: each starts where the
    // last one ended.
    fn advance(&mut self) {
        while let Some(key) = self.travel.get(self.leg)
            && self.pose_seconds >= key.end_seconds
        {
            let rest = vec3::sub(key.value, self.anchor_progress);
            self.anchor = vec3::add(self.anchor, quat_rotate(self.frame_turn, rest));
            self.anchor_progress = [0.0; 3];
            self.leg += 1;
        }
    }

    // Where the track puts the carried point at the pose time.
    fn position(&self) -> [f32; 3] {
        let progress = timeline::progress(&self.travel, self.leg, self.pose_seconds);
        let traveled = quat_rotate(self.frame_turn, vec3::sub(progress, self.anchor_progress));
        vec3::add(self.anchor, traveled)
    }

    // Pick up the subject the track has been filming, where it is and as
    // fast as it is going, to ease it onto travel's path from there.
    fn pick_up(&mut self, ctx: &PipelineContext<'_>) {
        let Some(at) = self
            .subject
            .and_then(|id| ctx.get_by_id::<Transform>(id))
            .map(|t| t.position)
        else {
            return;
        };
        self.advance();
        let planned = self.position();
        let pace = quat_rotate(
            self.frame_turn,
            timeline::rate(&self.travel, self.leg, self.pose_seconds),
        );
        let gap = vec3::sub(at, planned);
        self.blend = Some(CarryBlend::new(
            gap,
            vec3::sub(self.filmed.velocity(), pace),
            self.pose_seconds,
        ));
        self.pickup = Some(TrackPickup {
            seconds: self.pose_seconds,
            gap: vec3::length(gap),
            speed: vec3::length(self.filmed.velocity()),
        });
        self.carrying = true;
    }

    // Carry travel on from `at` mid-leg: where a frame move carried the point,
    // or where the camera stands once the subject it filmed is gone.
    fn reanchor(&mut self, at: [f32; 3]) {
        self.anchor = at;
        self.anchor_progress = timeline::progress(&self.travel, self.leg, self.pose_seconds);
    }

    // Let go of a subject that has gone since the camera was last posed: the
    // track carries on driving the camera from where it stands.
    fn check_subject(&mut self, ctx: &PipelineContext<'_>) {
        let Some(id) = self.subject else {
            return;
        };
        if ctx.get_by_id::<Transform>(id).is_some() {
            return;
        }
        tracing::warn!("camera track: its subject ({id}) is gone, so the track drives the camera");
        self.subject = None;
        if let Some(camera) = self.camera.and_then(|c| ctx.get::<Camera3D>(c)) {
            let at = camera.position;
            self.reanchor(at);
        }
    }

    // Write the pose the track holds at the pose time onto the subject and
    // the camera, and publish where the track has reached.
    fn apply(&mut self, ctx: &mut PipelineContext<'_>) {
        self.advance();
        let elapsed = self.elapsed();
        let angles = timeline::sample(self.start_angles_deg, &self.turn, self.pose_seconds);
        let [yaw, pitch] = angles.map(f32::to_radians);
        let position = self.position();
        if let Some(subject) = self.subject {
            let filmed_at = if self.carrying {
                let gap = self.blend.map_or([0.0; 3], |b| b.at(self.pose_seconds));
                let at = vec3::add(position, gap);
                if let Some(transform) = ctx.get_mut_by_id::<Transform>(subject) {
                    transform.position = at;
                }
                at
            } else {
                let at = ctx
                    .get_by_id::<Transform>(subject)
                    .map_or(position, |t| t.position);
                self.filmed.see(at, self.pose_seconds);
                at
            };
            let boom = timeline::sample(self.boom_start, &self.boom, self.pose_seconds);
            if let Some(camera) = self.camera
                && let Some(cam) = ctx.get_mut::<Camera3D>(camera)
            {
                let eye = boom_eye(filmed_at, yaw, pitch, cam.up, boom);
                cam.set_pose(eye, yaw, pitch);
            }
        } else if let Some(camera) = self.camera
            && let Some(cam) = ctx.get_mut::<Camera3D>(camera)
        {
            cam.set_pose(position, yaw, pitch);
        }
        let duration = self.duration();
        ctx.insert_resource(CameraTrackStatus {
            elapsed_seconds: elapsed,
            pose_seconds: self.pose_seconds,
            duration_seconds: duration,
            finished: elapsed >= duration,
            subject: self.subject.filter(|_| self.carrying),
            pickup: self.pickup,
        });
    }
}

impl System for CameraTrackSystem {
    fn init(&mut self, ctx: &mut PipelineContext<'_>) {
        if let Some((entity, camera)) = ctx.query_with_entity::<Camera3D>().next() {
            self.camera = Some(entity);
            self.start_position = camera.position;
            self.start_angles_deg = [camera.yaw.to_degrees(), camera.pitch.to_degrees()];
        }
        self.anchor = self.start_position;
        if self.subject.is_none() && !self.boom_legs.is_empty() {
            tracing::warn!("camera track: it has boom legs but no subject, so they are not used");
        }
        if let Some(id) = self.subject {
            match subject_start(ctx, id) {
                Ok(position) => {
                    let authored = vec3::length(vec3::sub(self.start_position, position));
                    self.anchor = position;
                    self.resolve_boom(authored);
                    self.carrying = self.carry_from <= 0.0;
                }
                Err(why) => {
                    tracing::warn!(
                        "camera track: its subject ({id}) {why}, so the track drives the camera"
                    );
                    self.subject = None;
                }
            }
        }
        self.rebases = RebaseCursor::at(ctx.resource::<FrameRebases>());
        self.resolve_turn();
        self.duration = timeline::duration(&self.travel)
            .max(timeline::duration(&self.turn))
            .max(timeline::duration(&self.boom));
        self.apply(ctx);
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        if ctx.resource::<MenuActive>().is_some_and(|m| m.0) {
            return StepResult::Continue;
        }
        // The frame moved the carried point: travel carries on from where it
        // was carried to, its offsets turned with the frame.
        if let Some(rebase) = self.rebases.take(ctx.resource::<FrameRebases>()) {
            // A subject travel carries exactly is where travel put it; one
            // only filmed, or still easing onto the path, is not, so the path
            // itself is carried.
            let exact = self.carrying && self.blend.is_none();
            let carried = match self.subject {
                Some(id) if exact => ctx.get_by_id::<Transform>(id).map(|t| t.position),
                Some(_) => {
                    self.advance();
                    Some(rebase.apply_point(self.position()))
                }
                None => self
                    .camera
                    .and_then(|c| ctx.get::<Camera3D>(c))
                    .map(|c| c.position),
            };
            if let Some(at) = carried {
                self.reanchor(at);
            }
            self.frame_turn = quat_normalize(quat_mul(rebase.rotation, self.frame_turn));
            self.filmed.rebase(&rebase);
            if let Some(blend) = self.blend.as_mut() {
                blend.rebase(&rebase);
            }
        }
        // Before the clock moves, so travel is handed over at the pose the
        // camera stands in.
        self.check_subject(ctx);
        let timing = ctx.resource::<SimTiming>().copied().unwrap_or_default();
        let before = self.elapsed();
        self.ticks += u64::from(timing.ticks);
        self.tick_dt = timing.tick_dt;
        let now = self.elapsed();
        let cut = timeline::cut_crossed(&self.travel, before, now)
            || timeline::cut_crossed(&self.turn, before, now)
            || (self.subject.is_some() && timeline::cut_crossed(&self.boom, before, now));
        self.pose_seconds = pose_seconds(now, &timing, self.pose_seconds, cut);
        if !self.carrying && self.subject.is_some() && self.pose_seconds >= self.carry_from {
            self.pick_up(ctx);
        }
        self.apply(ctx);
        if cut && self.camera.is_some() {
            PendingHistoryReset::raise(ctx, HistoryResetCauses::CAMERA_CUT);
        }
        StepResult::Continue
    }
}

// Where the subject `id` starts, or why it cannot be carried: its Transform
// is the one travel writes, so it has to be a world-space root. A subject the
// physics simulates follows that Transform kinematically.
fn subject_start(ctx: &PipelineContext<'_>, id: AssetId) -> Result<[f32; 3], &'static str> {
    let entity = ctx.entity_of(id).ok_or("is missing")?;
    let transform = ctx.get::<Transform>(entity).ok_or("has no transform")?;
    if ctx.get::<Parent>(entity).is_some() {
        return Err("has a parent, so its transform is not in world space");
    }
    Ok(transform.position)
}

// The eye of a camera at `yaw` and `pitch` about `up` that sees `subject`
// `boom` away: its distance, then degrees right of and above the center of
// view.
fn boom_eye(subject: [f32; 3], yaw: f32, pitch: f32, up: [f32; 3], boom: [f32; 3]) -> [f32; 3] {
    let [distance, right_deg, up_deg] = boom;
    let (right, rise) = (right_deg.to_radians(), up_deg.to_radians());
    let [x_axis, y_axis, back] = view_axes(yaw, pitch, up);
    let along = [cos(rise) * sin(right), sin(rise), -cos(rise) * cos(right)];
    let toward = vec3::add(
        vec3::add(vec3::scale(x_axis, along[0]), vec3::scale(y_axis, along[1])),
        vec3::scale(back, along[2]),
    );
    vec3::sub(subject, vec3::scale(toward, distance))
}

// The track time a frame poses the camera at: `alpha` of the way from the
// step before `elapsed` to it. A frame whose steps crossed a cut shows the
// step itself, so the jump lands on the frame that resets history, and the
// pose never runs back across it after.
fn pose_seconds(elapsed: f32, timing: &SimTiming, previous: f32, cut: bool) -> f32 {
    if cut {
        return elapsed;
    }
    let blended = elapsed - (1.0 - timing.alpha.clamp(0.0, 1.0)) * timing.tick_dt;
    blended.max(previous)
}

// How long a turn leg runs: its authored duration, or the one its rate implies
// over the wider of the yaw and pitch sweeps.
fn turn_seconds(leg: &CameraTurnKey, sweep_deg: f32) -> f32 {
    if leg.cut {
        0.0
    } else if leg.seconds > 0.0 {
        leg.seconds
    } else if leg.degrees_per_second > 0.0 {
        sweep_deg / leg.degrees_per_second
    } else {
        0.0
    }
}

#[cfg(test)]
mod rig_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::cook::Camera3D as Camera3DArgs;
    use crate::components::{CameraTravel, CameraTurn, Ease};
    use crate::ecs::World;
    use alloc::vec;
    use concinnity_core::math::SaturatingFrom;

    const TICKS_PER_SECOND: usize = 60;

    fn track(travel: Vec<CameraTravel>, turn: Vec<CameraTurn>) -> CameraTrack {
        CameraTrack::bake(crate::components::CameraTrackArgs {
            travel,
            turn,
            ..Default::default()
        })
    }

    // A world holding one camera at `pose` (position, yaw degrees, pitch
    // degrees) and the track that drives it.
    fn world_with(
        track: CameraTrack,
        position: [f32; 3],
        yaw_deg: f32,
        pitch_deg: f32,
    ) -> (World, CameraTrackSystem) {
        let system = CameraTrackSystem::new(&track);
        let mut world = World::new();
        world.add_component(Camera3D::bake(Camera3DArgs {
            position,
            yaw: yaw_deg.to_radians(),
            pitch: pitch_deg.to_radians(),
            controller: None,
            ..Default::default()
        }));
        world.add_component(track);
        (world, system)
    }

    fn run(world: &mut World, system: &mut CameraTrackSystem, seconds: f32) {
        let ticks = usize::saturating_from((seconds * TICKS_PER_SECOND as f32).round());
        for _ in 0..ticks {
            system.step(&mut world.context());
        }
    }

    fn camera_position(world: &World) -> [f32; 3] {
        world.query::<Camera3D>().next().expect("camera").position
    }

    fn camera_yaw_deg(world: &World) -> f32 {
        world
            .query::<Camera3D>()
            .next()
            .expect("camera")
            .yaw
            .to_degrees()
    }

    fn status(world: &World) -> CameraTrackStatus {
        world
            .resource::<CameraTrackStatus>()
            .copied()
            .expect("published")
    }

    fn travel(direction: [f32; 3], distance: f32, speed: f32) -> CameraTravel {
        CameraTravel {
            direction,
            distance,
            speed,
            ..Default::default()
        }
    }

    #[test]
    fn the_camera_starts_where_it_was_authored() {
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 8.0, 2.0)], vec![]),
            [3.0, 1.0, -2.0],
            45.0,
            0.0,
        );
        system.init(&mut world.context());
        assert_eq!(camera_position(&world), [3.0, 1.0, -2.0]);
        assert!((camera_yaw_deg(&world) - 45.0).abs() < 1e-3);
        assert_eq!(status(&world).elapsed_seconds, 0.0);
        assert!(!status(&world).finished);
    }

    // A frame move mid-track carries the camera on from where it was carried
    // to, its later travel turned with the frame, as if the track ran on in
    // the old frame and every pose were carried after.
    #[test]
    fn a_frame_move_carries_the_track_on() {
        use crate::planet::Rebase;
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 20.0, 4.0)], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        world.insert_resource(FrameRebases::default());
        system.init(&mut world.context());
        run(&mut world, &mut system, 1.0);
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.0, 0.0, 1.0], 0.02),
            translation: [-4.0, 0.1, 0.0],
        };
        let mut rebases = FrameRebases::default();
        rebases.push(rebase);
        world.insert_resource(rebases);
        for camera in world.context().query_mut::<Camera3D>() {
            camera.position = rebase.apply_point(camera.position);
        }
        run(&mut world, &mut system, 1.0);
        let expected = rebase.apply_point([8.0, 0.0, 0.0]);
        let got = camera_position(&world);
        assert!(
            (0..3).all(|i| (got[i] - expected[i]).abs() < 1e-3),
            "{got:?} {expected:?}"
        );
    }

    // A 300 km track crossing hundreds of frame moves ends where its legs
    // add up to, to millimeters: each leg is laid out from where the last
    // one ended, so no offset grows with the track.
    #[test]
    fn a_long_track_through_many_moves_stays_precise() {
        use crate::planet::Rebase;
        let legs: Vec<CameraTravel> = (0..300)
            .map(|i| {
                let side = if i % 2 == 0 { 0.6 } else { -0.6 };
                travel([1.0, 0.0, side], 1_000.0, 2_000.0)
            })
            .collect();
        let (mut world, mut system) = world_with(track(legs, vec![]), [0.25, 2.0, 0.0], 0.0, 0.0);
        world.insert_resource(FrameRebases::default());
        system.init(&mut world.context());
        let mut rebases = FrameRebases::default();
        // Where the frame's origin sits, in the frame the track started in.
        let mut origin = [0.0_f64; 3];
        for _ in 0..(150 * TICKS_PER_SECOND + 30) {
            system.step(&mut world.context());
            let at = camera_position(&world);
            if !crate::planet::needs_rebase(at, 1_000.0) {
                continue;
            }
            let rebase = Rebase {
                rotation: [0.0, 0.0, 0.0, 1.0],
                translation: [-at[0], -at[1], -at[2]],
            };
            for i in 0..3 {
                origin[i] += f64::from(at[i]);
            }
            rebases.push(rebase);
            world.insert_resource(rebases.clone());
            for camera in world.context().query_mut::<Camera3D>() {
                camera.position = rebase.apply_point(camera.position);
            }
        }
        assert!(rebases.count() > 250, "{}", rebases.count());
        let leg = 1_000.0 / (1.0_f64 + 0.36).sqrt();
        let expected = [0.25 + 300.0 * leg, 2.0, 0.0];
        let end = camera_position(&world);
        let reached: [f64; 3] = core::array::from_fn(|i| origin[i] + f64::from(end[i]));
        for i in 0..3 {
            assert!(
                (reached[i] - expected[i]).abs() < 5e-3,
                "{reached:?} {expected:?}"
            );
        }
    }

    #[test]
    fn travel_offsets_ride_on_the_authored_position() {
        // The track's offsets are relative, so a camera authored away from the
        // origin follows the same shape from where it stands.
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 8.0, 2.0)], vec![]),
            [10.0, 0.0, 0.0],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 2.0);
        assert!((camera_position(&world)[0] - 14.0).abs() < 1e-3);
        run(&mut world, &mut system, 2.0);
        assert!((camera_position(&world)[0] - 18.0).abs() < 1e-3);
    }

    #[test]
    fn the_clock_is_the_fixed_step_so_two_tick_rates_reach_the_same_pose() {
        // What the camera is looking at is a function of track time, never of
        // how fast the host renders.
        let legs = || vec![travel([0.0, 0.0, -1.0], 12.0, 3.0)];
        let (mut a, mut sa) = world_with(track(legs(), vec![]), [0.0; 3], 0.0, 0.0);
        sa.init(&mut a.context());
        for _ in 0..120 {
            sa.step(&mut a.context());
        }

        let (mut b, mut sb) = world_with(track(legs(), vec![]), [0.0; 3], 0.0, 0.0);
        sb.init(&mut b.context());
        // Half the frames, each carrying twice the simulation steps: the same
        // two seconds of track time.
        for _ in 0..60 {
            b.context().insert_resource(SimTiming {
                ticks: 2,
                ..Default::default()
            });
            sb.step(&mut b.context());
        }
        assert_eq!(camera_position(&a), camera_position(&b));
    }

    #[test]
    fn a_finished_track_holds_its_last_pose() {
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 4.0, 2.0)], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 10.0);
        assert!((camera_position(&world)[0] - 4.0).abs() < 1e-3);
        let s = status(&world);
        assert!(s.finished);
        assert!((s.duration_seconds - 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_turn_takes_its_sweep_divided_by_its_rate_from_the_authored_heading() {
        // 90 degrees at 45 per second is two seconds, and the sweep is measured
        // from where the camera already looks rather than from zero.
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![CameraTurn {
                    yaw_deg: Some(90.0),
                    degrees_per_second: 45.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            30.0,
            0.0,
        );
        system.init(&mut world.context());
        assert!((system.duration() - 60.0 / 45.0).abs() < 1e-5);
        run(&mut world, &mut system, 60.0 / 45.0);
        assert!((camera_yaw_deg(&world) - 90.0).abs() < 1e-2);
    }

    #[test]
    fn a_turn_takes_the_shorter_way_round() {
        // From 350 to 10 degrees is 20 degrees forward, not 340 back.
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![CameraTurn {
                    yaw_deg: Some(10.0),
                    degrees_per_second: 20.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            350.0,
            0.0,
        );
        system.init(&mut world.context());
        assert!((system.duration() - 1.0).abs() < 1e-5);
        run(&mut world, &mut system, 0.5);
        // Half way is 360, which is the same heading as 0 and never 180.
        let half = camera_yaw_deg(&world);
        assert!((half - 360.0).abs() < 1e-2, "{half}");
    }

    #[test]
    fn a_turn_leg_with_no_target_holds_the_heading_it_started_from() {
        // An omitted angle has to mean "leave it alone": zero is a heading
        // like any other, so a hold leg would otherwise swing the camera to it.
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![
                    CameraTurn {
                        seconds: 2.0,
                        ..Default::default()
                    },
                    CameraTurn {
                        yaw_deg: Some(60.0),
                        degrees_per_second: 30.0,
                        ..Default::default()
                    },
                ],
            ),
            [0.0; 3],
            25.0,
            -8.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 2.0);
        assert!((camera_yaw_deg(&world) - 25.0).abs() < 1e-3);
        // The turn that follows starts from the held heading, so its sweep is
        // 35 degrees rather than 60.
        assert!((system.duration() - (2.0 + 35.0 / 30.0)).abs() < 1e-4);
    }

    #[test]
    fn a_turn_leg_moves_only_the_angle_it_names() {
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![CameraTurn {
                    yaw_deg: Some(90.0),
                    degrees_per_second: 45.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            0.0,
            -30.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 2.0);
        assert!((camera_yaw_deg(&world) - 90.0).abs() < 1e-2);
        let pitch = world.query::<Camera3D>().next().expect("camera").pitch;
        assert!((pitch.to_degrees() + 30.0).abs() < 1e-2);
    }

    #[test]
    fn yaw_and_pitch_arrive_together_on_the_wider_sweep() {
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![CameraTurn {
                    yaw_deg: Some(20.0),
                    pitch_deg: Some(-60.0),
                    degrees_per_second: 30.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        // Pitch sweeps 60 degrees and yaw only 20, so the leg is two seconds.
        assert!((system.duration() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn the_two_tracks_run_against_one_clock_without_bending_each_other() {
        // The path is world space, so a camera turning through 90 degrees while
        // it travels still ends up where the travel leg alone would put it.
        let (mut world, mut system) = world_with(
            track(
                vec![travel([0.0, 0.0, -1.0], 10.0, 5.0)],
                vec![CameraTurn {
                    yaw_deg: Some(90.0),
                    degrees_per_second: 45.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 2.0);
        let p = camera_position(&world);
        assert!((p[2] + 10.0).abs() < 1e-3, "{p:?}");
        assert!(p[0].abs() < 1e-4, "{p:?}");
        assert!((camera_yaw_deg(&world) - 90.0).abs() < 1e-2);
    }

    #[test]
    fn the_longer_list_fixes_the_duration_and_the_shorter_one_holds() {
        let (mut world, mut system) = world_with(
            track(
                vec![travel([1.0, 0.0, 0.0], 2.0, 2.0)],
                vec![CameraTurn {
                    yaw_deg: Some(90.0),
                    degrees_per_second: 15.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        // Travel runs 1s, the turn 6s.
        assert!((system.duration() - 6.0).abs() < 1e-5);
        run(&mut world, &mut system, 3.0);
        assert!(!status(&world).finished);
        // Travel finished at 1s and has been holding its end ever since.
        assert!((camera_position(&world)[0] - 2.0).abs() < 1e-3);
        run(&mut world, &mut system, 3.0);
        assert!(status(&world).finished);
    }

    #[test]
    fn an_open_menu_freezes_the_track() {
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 8.0, 2.0)], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 1.0);
        let held = camera_position(&world);
        world.context().insert_resource(MenuActive(true));
        run(&mut world, &mut system, 2.0);
        assert_eq!(camera_position(&world), held);
        world.context().insert_resource(MenuActive(false));
        run(&mut world, &mut system, 1.0);
        assert!(camera_position(&world)[0] > held[0]);
    }

    #[test]
    fn easing_reaches_the_same_end_as_a_linear_leg() {
        let eased = CameraTravel {
            ease: Ease::InOut,
            ..travel([1.0, 0.0, 0.0], 10.0, 5.0)
        };
        let (mut world, mut system) = world_with(track(vec![eased], vec![]), [0.0; 3], 0.0, 0.0);
        system.init(&mut world.context());
        // A quarter of the way in the camera is still behind where a constant
        // rate would have put it, and three quarters in it is ahead.
        run(&mut world, &mut system, 0.5);
        assert!(camera_position(&world)[0] < 2.5);
        run(&mut world, &mut system, 1.0);
        assert!(camera_position(&world)[0] > 7.5);
        // The endpoint is the leg's whole distance either way.
        run(&mut world, &mut system, 0.5);
        assert!((camera_position(&world)[0] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn a_world_with_no_camera_still_publishes_where_the_track_reached() {
        let t = track(vec![travel([1.0, 0.0, 0.0], 4.0, 2.0)], vec![]);
        let mut system = CameraTrackSystem::new(&t);
        let mut world = World::new();
        world.add_component(t);
        system.init(&mut world.context());
        run(&mut world, &mut system, 3.0);
        assert!(status(&world).finished);
    }

    // The ticks, counted from the first, on which the track raised a cut.
    fn cut_ticks(world: &mut World, system: &mut CameraTrackSystem, seconds: f32) -> Vec<usize> {
        let ticks = usize::saturating_from((seconds * TICKS_PER_SECOND as f32).round());
        let mut cuts = Vec::new();
        for tick in 0..ticks {
            let mut ctx = world.context();
            system.step(&mut ctx);
            if PendingHistoryReset::take(&mut ctx).contains(HistoryResetCauses::CAMERA_CUT) {
                cuts.push(tick);
            }
        }
        cuts
    }

    #[test]
    fn a_cut_leg_jumps_and_raises_one_camera_cut_on_the_tick_it_lands() {
        let jump = CameraTravel {
            cut: true,
            ..travel([0.0, 0.0, 1.0], 12.0, 0.0)
        };
        let hold = CameraTravel {
            seconds: 0.5,
            ..Default::default()
        };
        let (mut world, mut system) = world_with(
            track(vec![hold.clone(), jump, hold], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        // The hold ends half a second in, on the 30th tick (index 29).
        assert_eq!(cut_ticks(&mut world, &mut system, 2.0), [29]);
        assert_eq!(camera_position(&world), [0.0, 0.0, 12.0]);
    }

    #[test]
    fn a_cut_turn_snaps_and_raises_a_camera_cut() {
        let hold = CameraTurn {
            seconds: 0.5,
            ..Default::default()
        };
        let snap = CameraTurn {
            yaw_deg: Some(120.0),
            degrees_per_second: 10.0,
            cut: true,
            ..Default::default()
        };
        let (mut world, mut system) =
            world_with(track(vec![], vec![hold, snap]), [0.0; 3], 0.0, 0.0);
        system.init(&mut world.context());
        assert_eq!(cut_ticks(&mut world, &mut system, 1.0), [29]);
        assert!((camera_yaw_deg(&world) - 120.0).abs() < 1e-3);
    }

    // One frame carrying `ticks` steps, with `alpha` of a step left over.
    fn frame(world: &mut World, system: &mut CameraTrackSystem, ticks: u32, alpha: f32) {
        world.context().insert_resource(SimTiming {
            ticks,
            tick_dt: SimTiming::TICK_DT,
            alpha,
            ..Default::default()
        });
        system.step(&mut world.context());
    }

    fn assert_near(got: [f32; 3], expected: [f32; 3]) {
        assert!(
            (0..3).all(|i| (got[i] - expected[i]).abs() < 1e-4),
            "{got:?} {expected:?}"
        );
    }

    #[test]
    fn a_frame_poses_the_camera_between_its_last_two_steps() {
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 8.0, 2.0)], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        for _ in 0..60 {
            frame(&mut world, &mut system, 1, 1.0);
        }
        assert_near(camera_position(&world), [2.0, 0.0, 0.0]);
        frame(&mut world, &mut system, 1, 0.25);
        let pose = 60.25 / 60.0;
        assert_near(camera_position(&world), [2.0 * pose, 0.0, 0.0]);
        // The track's own clock still counts whole steps.
        let s = status(&world);
        assert_eq!(s.elapsed_seconds, 61.0 * SimTiming::TICK_DT);
        assert!((s.pose_seconds - pose).abs() < 1e-6, "{s:?}");
    }

    // Frames that do not divide the step evenly each carry one or two steps,
    // yet every one moves the camera as far as the frame took.
    #[test]
    fn steady_frames_move_the_camera_evenly() {
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 100.0, 2.0)], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        let frame_dt = 1.0 / 45.0;
        let mut accumulator = 0.0_f32;
        let mut last = None;
        let mut step_counts = Vec::new();
        for _ in 0..200 {
            accumulator += frame_dt;
            let ticks = u32::saturating_from(accumulator / SimTiming::TICK_DT);
            accumulator -= ticks as f32 * SimTiming::TICK_DT;
            frame(
                &mut world,
                &mut system,
                ticks,
                accumulator / SimTiming::TICK_DT,
            );
            step_counts.push(ticks);
            let x = camera_position(&world)[0];
            if let Some(previous) = last.replace(x) {
                let moved = x - previous;
                assert!((moved - 2.0 * frame_dt).abs() < 1e-3, "{moved}");
            }
        }
        assert!(step_counts.contains(&1) && step_counts.contains(&2));
    }

    #[test]
    fn the_pose_follows_the_steps_through_a_leg_boundary() {
        let (mut world, mut system) = world_with(
            track(
                vec![
                    travel([1.0, 0.0, 0.0], 2.0, 2.0),
                    travel([0.0, 0.0, 1.0], 2.0, 2.0),
                ],
                vec![],
            ),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        for _ in 0..59 {
            frame(&mut world, &mut system, 1, 1.0);
        }
        // The steps have finished the first leg, the pose not quite.
        frame(&mut world, &mut system, 1, 0.0);
        assert_near(camera_position(&world), [2.0 * 59.0 / 60.0, 0.0, 0.0]);
        frame(&mut world, &mut system, 1, 0.5);
        assert_near(camera_position(&world), [2.0, 0.0, 2.0 * 0.5 / 60.0]);
    }

    #[test]
    fn a_cut_shows_on_the_frame_that_resets_history_and_holds() {
        let jump = CameraTravel {
            cut: true,
            ..travel([0.0, 0.0, 1.0], 12.0, 0.0)
        };
        let hold = CameraTravel {
            seconds: 1.0,
            ..Default::default()
        };
        let (mut world, mut system) = world_with(
            track(vec![hold.clone(), jump, hold], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        let mut cut_at = None;
        for n in 0..90 {
            frame(&mut world, &mut system, 1, 0.3);
            let cut = PendingHistoryReset::take(&mut world.context())
                .contains(HistoryResetCauses::CAMERA_CUT);
            let z = camera_position(&world)[2];
            if cut {
                assert!(cut_at.replace(n).is_none());
            }
            assert_eq!(z == 12.0, cut_at.is_some(), "frame {n}: z {z}");
        }
        assert_eq!(cut_at, Some(59));
    }

    #[test]
    fn a_frame_move_between_steps_carries_the_track_on() {
        use crate::planet::Rebase;
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 20.0, 4.0)], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        world.insert_resource(FrameRebases::default());
        system.init(&mut world.context());
        for _ in 0..60 {
            frame(&mut world, &mut system, 1, 0.5);
        }
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.0, 0.0, 1.0], 0.02),
            translation: [-4.0, 0.1, 0.0],
        };
        let mut rebases = FrameRebases::default();
        rebases.push(rebase);
        world.insert_resource(rebases);
        for camera in world.context().query_mut::<Camera3D>() {
            camera.position = rebase.apply_point(camera.position);
        }
        for _ in 0..60 {
            frame(&mut world, &mut system, 1, 0.5);
        }
        let expected = rebase.apply_point([4.0 * 119.5 / 60.0, 0.0, 0.0]);
        let got = camera_position(&world);
        assert!(
            (0..3).all(|i| (got[i] - expected[i]).abs() < 1e-3),
            "{got:?} {expected:?}"
        );
    }

    #[test]
    fn a_track_without_cuts_raises_none() {
        let (mut world, mut system) = world_with(
            track(
                vec![travel([1.0, 0.0, 0.0], 300.0, 300.0)],
                vec![CameraTurn {
                    yaw_deg: Some(170.0),
                    degrees_per_second: 90.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        assert!(cut_ticks(&mut world, &mut system, 3.0).is_empty());
    }
}
