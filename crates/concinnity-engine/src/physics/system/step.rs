//! One frame of `PhysicsSystem::step`, phase by phase: take in what moved
//! outside the simulation, run the frame's fixed ticks, publish what they
//! produced, and write the blended poses back to the world.

use concinnity_core::math::vec3::{add, scale, sub};
use concinnity_core::math::{cos, sin, sqrt};

use crate::components::{
    Camera3D, Collider, ContactEvent, Held, RigidBody, Transform, TriggerFilter, VolumeEvent,
    WakeRequest,
};
use crate::ecs::{Entity, PipelineContext, ScheduleMode, SimTiming};
use crate::physics::{CharacterMoveInput, Simulation, probes, rig};
use crate::planet::{LocalFrame, PlanetFrame};

use super::drive;
use super::{
    HOLD_DISTANCE, HOLD_DROP, LAYER_CHARACTER, LAYER_PROP, LAYER_WORLD, PICKUP_MIN_DOT,
    PICKUP_REACH, PhysicsSystem, PlayerPhysics, THROW_SPEED, prop_snap, publish_stirs,
    tracked_subject, up_at,
};

/// What the camera asked of physics this frame, read once before any tick.
#[derive(Clone, Copy)]
struct CameraInput {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    desired_move: [f32; 3],
    jump: bool,
    interact: bool,
}

impl CameraInput {
    // The first camera's request, or a still camera over the floor without one.
    fn read(ctx: &PipelineContext<'_>, floor_y: f32) -> Self {
        ctx.query::<Camera3D>().next().map_or(
            Self {
                position: [0.0, floor_y, 0.0],
                yaw: 0.0,
                pitch: 0.0,
                desired_move: [0.0; 3],
                jump: false,
                interact: false,
            },
            |c| Self {
                position: c.position,
                yaw: c.yaw,
                pitch: c.pitch,
                desired_move: c.desired_move,
                jump: c.jump_requested,
                interact: c.interact_requested,
            },
        )
    }

    // Facing direction flattened onto the ground plane.
    fn forward_flat(&self) -> [f32; 3] {
        [-sin(self.yaw), 0.0, -cos(self.yaw)]
    }

    // Full facing direction, pitch included.
    fn forward(&self) -> [f32; 3] {
        [
            -(sin(self.yaw) * cos(self.pitch)),
            -sin(self.pitch),
            -(cos(self.yaw) * cos(self.pitch)),
        ]
    }

    // The carried prop's hover point in front of the camera.
    fn hold_point(&self) -> [f32; 3] {
        sub(
            add(self.position, scale(self.forward(), HOLD_DISTANCE)),
            [0.0, HOLD_DROP, 0.0],
        )
    }
}

/// What every tick of one frame shares.
struct FrameTicks<'a> {
    timing: SimTiming,
    camera: CameraInput,
    hold_point: [f32; 3],
    frame: LocalFrame,
    surroundings: drive::Surroundings<'a>,
    mode: ScheduleMode,
}

impl PhysicsSystem {
    pub(super) fn step_world(&mut self, world: &mut Simulation, ctx: &mut PipelineContext<'_>) {
        // The frame's fixed-tick budget and render blend factor. Absent (a
        // directly-stepped world with no App), every step runs exactly one tick
        // and writes the freshly simulated state.
        let timing = ctx.resource::<SimTiming>().copied().unwrap_or_default();
        let camera = CameraInput::read(ctx, self.floor_y);
        // Whichever pool the schedule names, read once for the frame. Both
        // land in the same place; only how long the step takes differs.
        let mode = crate::ecs::schedule_mode(ctx.resources);

        let planet = ctx.resource::<PlanetFrame>().cloned();
        let frame = planet.as_ref().map_or(LocalFrame::AUTHORED, |p| p.frame);
        // A planet reshaped while the world runs moves its sea's shore.
        if let (Some(water), Some(planet)) = (self.water.as_mut(), planet.as_ref()) {
            water.reshape(&planet.shape);
        }
        let sea_level = ctx
            .query::<crate::components::Planet>()
            .next()
            .and_then(|p| p.sea.as_ref())
            .map(|sea| sea.level);
        let surroundings = drive::Surroundings {
            north: frame.dir_to_local([0.0, 0.0, -1.0]),
            altimeter: drive::Altimeter {
                planet: planet
                    .as_ref()
                    .map(|planet| drive::PlanetSurface { planet, sea_level }),
                mask: self.layers.mask(LAYER_PROP),
            },
        };

        self.track_bodies(world, ctx);
        let held_changed = if camera.interact {
            self.toggle_carry(world, ctx, &camera)
        } else {
            None
        };
        self.adopt_external_camera(world, camera.position);
        self.adopt_external_poses(world, ctx, timing.tick_dt);

        let ticks = FrameTicks {
            timing,
            camera,
            hold_point: camera.hold_point(),
            frame,
            surroundings,
            mode,
        };
        for tick in 0..timing.ticks {
            self.run_tick(world, ctx, &ticks, tick);
        }

        self.publish_frame(world, ctx);
        self.write_back(world, ctx, timing.alpha, held_changed);
    }

    // Keep the tracked bodies in step with the world's entities: reap the
    // despawned, adopt the newly spawned, and wake the ones behaviors asked
    // for.
    fn track_bodies(&mut self, world: &mut Simulation, ctx: &PipelineContext<'_>) {
        // Reap bodies whose entity was despawned before the step, keeping
        // self.held a valid index into the compacted list.
        self.held = self
            .props
            .reap(world, self.held, |entity| ctx.is_alive(entity));

        // Adopt collider-bearing entities that appeared since init (runtime
        // spawns): each gets a body at its spawn transform, with its pose
        // snapshots seeded there so the render blend starts clean. The scan
        // materializes into scratch because adopting one mutates the tracked
        // set the scan itself reads.
        self.new_props.clear();
        self.new_props.extend(
            ctx.join2::<Collider, Transform>()
                .filter(|(entity, _, _)| {
                    !self.props.is_tracked(*entity) && !self.props.is_refused(*entity)
                })
                .map(|(entity, collider, transform)| {
                    (entity, prop_snap(ctx, entity, collider, transform))
                }),
        );
        for (entity, snap) in self.new_props.drain(..) {
            self.props
                .adopt(&self.layers, world, entity, &snap, self.body_cap);
        }

        // Wake the bodies behaviors asked for, after the adopt so a copy
        // spawned this frame can be woken in it too.
        if let Some(requests) = ctx.events::<WakeRequest>() {
            for request in requests.read(&mut self.wake_cursor) {
                if let Some(handle) = self.props.handle_of(request.target) {
                    world.wake_body(handle);
                }
            }
        }
    }

    // Pick up or drop on the interact edge. Returns the entity whose Held tag
    // the write-back toggles, and whether it is now held.
    fn toggle_carry(
        &mut self,
        world: &mut Simulation,
        ctx: &PipelineContext<'_>,
        camera: &CameraInput,
    ) -> Option<(Entity, bool)> {
        if let Some(held_idx) = self.held.take() {
            // drop: hand the prop back to dynamic simulation with a throw.
            let pp = self.props.get(held_idx)?;
            let throw = add(scale(camera.forward(), THROW_SPEED), [0.0, 1.0, 0.0]);
            world.make_dynamic(pp.handle, throw);
            return Some((pp.entity, false));
        }

        // pickup: nearest carriable prop within reach the player faces.
        let forward = camera.forward_flat();
        let mut best: Option<(f32, usize)> = None;
        for (idx, pp) in self.props.iter().enumerate() {
            if !pp.pickup {
                continue;
            }
            let pos = ctx
                .get::<Transform>(pp.entity)
                .map_or(camera.position, |t| t.position);
            let dx = pos[0] - camera.position[0];
            let dz = pos[2] - camera.position[2];
            let dist = sqrt(dx * dx + dz * dz);
            if dist >= PICKUP_REACH || dist <= 0.0 {
                continue;
            }
            let dot = (forward[0] * dx + forward[2] * dz) / dist;
            if dot > PICKUP_MIN_DOT && best.is_none_or(|(d, _)| dist < d) {
                best = Some((dist, idx));
            }
        }
        let (_, idx) = best?;
        let pp = self.props.get(idx)?;
        world.make_kinematic(pp.handle);
        self.held = Some(idx);
        Some((pp.entity, true))
    }

    // Adopt an externally moved camera (free-fly, a teleport): a position that
    // differs from the eye written back last frame was not ours, so the
    // capsule snaps to it with no blend across the jump.
    fn adopt_external_camera(&mut self, world: &Simulation, camera_position: [f32; 3]) {
        if let Some(player) = self.player.as_mut()
            && player.written_eye != Some(camera_position)
        {
            let up = up_at(world, camera_position);
            player
                .center
                .snap(sub(camera_position, scale(up, player.eye_offset)));
        }
    }

    // Take in what moved outside physics before any tick runs: prop poses
    // written from elsewhere, a camera track's subject, root motion, and the
    // steering of driven bodies.
    fn adopt_external_poses(
        &mut self,
        world: &mut Simulation,
        ctx: &mut PipelineContext<'_>,
        tick_dt: f32,
    ) {
        // Adopt prop transforms written from outside physics (a behavior's
        // `set_transform`, an editor drag): a Transform that differs from the
        // one written back last frame teleports the body there before any tick
        // runs, so the pose the world asked for is the one it simulates from.
        self.props.sync_external_poses(world, |entity| {
            ctx.get::<Transform>(entity)
                .map(|t| (t.position, t.rotation_deg))
        });

        // A camera track's subject is posed by the track: its body follows the
        // Transform the track writes, kinematically.
        let owned = tracked_subject(ctx);
        self.props.pose_only(world, owned, tick_dt);

        // Root-motion displacements published since last frame, applied on the
        // frame's first tick. Rig capsules whose entity moved externally snap
        // before any tick runs.
        rig::drain_motions_into(ctx, &mut self.root_cursor, &mut self.motion_scratch);
        rig::sync_rigs(ctx, &mut self.rigs);
        self.drives.gather(ctx, &self.props, tick_dt);
    }

    // One fixed tick: pose the kinematic bodies, move the capsules, push what
    // floats, advance the simulation, and record what it produced.
    fn run_tick(
        &mut self,
        world: &mut Simulation,
        ctx: &mut PipelineContext<'_>,
        ticks: &FrameTicks<'_>,
        tick: u32,
    ) {
        let timing = &ticks.timing;
        let dt = timing.tick_dt;

        // carried prop hovers in front of the camera
        if let Some(prop) = self.held.and_then(|idx| self.props.get(idx)) {
            world.set_kinematic_translation(prop.handle, ticks.hold_point);
        }
        self.props.drive_posed(world, tick, timing.ticks);
        self.drives.tick(
            world,
            ctx,
            &self.props,
            (timing.tick_start(tick), dt),
            &ticks.surroundings,
        );

        if let Some(player) = self.player.as_mut() {
            let jump = tick == 0 && ticks.camera.jump;
            player.advance(
                world,
                ticks.camera.desired_move,
                jump,
                self.gravity,
                dt,
                self.layers.mask(LAYER_CHARACTER),
            );
        }

        // move the root-motion character rig capsules
        rig::tick_rigs(
            world,
            ctx,
            &mut self.rigs,
            if tick == 0 { &self.motion_scratch } else { &[] },
            dt,
            self.gravity,
            self.layers.mask(LAYER_CHARACTER),
        );

        // the water's push on whatever floats in it; with nothing afloat its
        // waves are not built
        if let Some(water) = self.water.as_mut() {
            let time = timing.tick_start(tick);
            let mut floaters = self.props.floaters_mut().peekable();
            if floaters.peek().is_some() {
                water.prepare(time);
            }
            for (handle, floater) in floaters {
                floater.push(handle, world, water, &ticks.frame, time, dt);
            }
        }

        self.fanout.step(world, dt, ticks.mode);

        // batch the tick's contact hits (strongest per pair this frame)
        self.contact_gate.advance_tick();
        world.drain_contact_hits_into(&mut self.contact_scratch);
        for hit in self.contact_scratch.drain(..) {
            self.contact_batch.add(hit);
        }

        // record the tick's dynamic prop poses for the render blend
        self.props.record_tick_poses(world);
    }

    // Publish what the frame's ticks produced: driven-body and water state,
    // probe answers, contact events and sensor crossings.
    fn publish_frame(&mut self, world: &mut Simulation, ctx: &mut PipelineContext<'_>) {
        self.drives.publish(world, ctx, &self.props);
        if self.water.is_some() {
            publish_stirs(ctx, &self.props);
        }

        // answer the IK ground probes and the follow camera's occlusion probe
        probes::step_probes(
            world,
            ctx,
            &self.rigs,
            &self.props,
            self.layers
                .query_mask(LAYER_CHARACTER, &[LAYER_WORLD, LAYER_PROP]),
        );

        self.publish_contacts(ctx);
        self.publish_crossings(world, ctx);
    }

    // The frame's contact events: one per body pair that passed the impulse
    // threshold, gated by the per-pair refractory. `a` is always a prop
    // entity; a hit whose sides both lack one (terrain, capsules) has no
    // consumer-visible subject and is dropped.
    fn publish_contacts(&mut self, ctx: &mut PipelineContext<'_>) {
        for hit in self.contact_batch.drain() {
            let event = match (self.props.entity_of(hit.a), self.props.entity_of(hit.b)) {
                (Some(a), b) => ContactEvent {
                    a,
                    b,
                    point: hit.point,
                    normal: hit.normal,
                    impulse: hit.impulse,
                },
                (None, Some(b)) => ContactEvent {
                    a: b,
                    b: None,
                    point: hit.point,
                    normal: [-hit.normal[0], -hit.normal[1], -hit.normal[2]],
                    impulse: hit.impulse,
                },
                (None, None) => continue,
            };
            if self.contact_gate.admit(&hit) {
                ctx.events_mut::<ContactEvent>().send(event);
            }
        }
    }

    // The sensor boundary crossings that pass their volume's filter. A
    // crossing whose body was removed this same step has no `other` to
    // classify, so only an `any` volume reports it.
    fn publish_crossings(&mut self, world: &mut Simulation, ctx: &mut PipelineContext<'_>) {
        world.drain_sensor_crossings_into(&mut self.sensor_scratch);
        for crossing in self.sensor_scratch.drain(..) {
            let Some(&(volume, filter)) = self.sensor_filters.get(&crossing.tag) else {
                continue;
            };
            let passes = match filter {
                TriggerFilter::Player => crossing.other.is_some_and(|h| {
                    self.player.as_ref().is_some_and(|p| p.handle == h)
                        || self.rigs.iter().any(|r| r.handle == h)
                }),
                TriggerFilter::Props => crossing
                    .other
                    .is_some_and(|h| self.props.entity_of(h).is_some()),
                TriggerFilter::Any => true,
            };
            if passes {
                ctx.events_mut::<VolumeEvent>().send(VolumeEvent {
                    volume,
                    entered: crossing.entered,
                });
            }
        }
    }

    // Write the frame's blended state back to the world: prop poses, the Held
    // tag, the camera eye, the grounded flag, and the rig positions.
    fn write_back(
        &mut self,
        world: &Simulation,
        ctx: &mut PipelineContext<'_>,
        alpha: f32,
        held_changed: Option<(Entity, bool)>,
    ) {
        // Positions lerped, rotations slerped as quaternions, with the Euler
        // decomposition happening only here at the write boundary.
        for &(entity, pos, rot) in self.props.sample_poses(alpha) {
            if let Some(t) = ctx.get_mut::<Transform>(entity) {
                t.position = pos;
                t.rotation_deg = rot;
            }
        }
        if let Some((entity, is_held)) = held_changed {
            if is_held {
                if ctx.get::<Held>(entity).is_none() {
                    ctx.insert(entity, Held);
                }
            } else {
                ctx.remove::<Held>(entity);
            }
        }

        // write the blended camera position + view matrix
        let mut grounded = true;
        if let Some(player) = self.player.as_mut() {
            let center = player.center.sample(alpha);
            let eye = add(center, scale(up_at(world, center), player.eye_offset));
            player.written_eye = Some(eye);
            grounded = player.grounded;
            for camera in ctx.query_mut::<Camera3D>() {
                camera.position = eye;
                camera.recompose_view();
            }
        }

        // publish grounded state for jump gating
        for body in ctx.query_mut::<RigidBody>() {
            body.is_grounded = grounded;
        }

        // write the blended rig positions for the render follow
        rig::publish_rigs(ctx, &mut self.rigs, alpha);
    }
}

impl PlayerPhysics {
    // Move the capsule through one tick: gravity and a requested jump along
    // the pull's up, then the requested walk resolved against the world.
    fn advance(
        &mut self,
        world: &mut Simulation,
        desired_move: [f32; 3],
        jump: bool,
        gravity: f32,
        dt: f32,
        mask: crate::physics::LayerMask,
    ) {
        if self.has_gravity {
            if jump && self.grounded && self.jump_height > 0.0 {
                self.vy = sqrt(2.0 * gravity * self.gravity_scale * self.jump_height);
            }
            self.vy -= gravity * self.gravity_scale * dt;
        }

        // Up is away from the pull, so the vertical speed rides it.
        let center = self.center.current();
        let up = up_at(world, center);
        let desired = add(scale(desired_move, dt), scale(scale(up, self.vy), dt));
        let moved = world.move_character(
            &self.shape,
            &CharacterMoveInput {
                center,
                desired,
                dt,
                exclude: self.handle,
                mask,
            },
        );
        let new_center = add(center, moved.translation);
        world.set_kinematic_translation(self.handle, new_center);

        self.grounded = moved.grounded;
        if moved.grounded && self.vy < 0.0 {
            self.vy = 0.0;
        }
        self.center.push(new_center);
    }
}
