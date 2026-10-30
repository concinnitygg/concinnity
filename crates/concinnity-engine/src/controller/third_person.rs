//! Third-person follow controller. An internal system (not a declarable
//! asset): `World::start` constructs one instead of `Camera3DSystem` when the
//! controlling `Camera3D`'s controller carries a `follow` block. The mouse
//! orbits the camera around what it follows, about the ground's up, and WASD
//! steers it camera-relative along the ground.
//!
//! A followed `CharacterRig` turns its facing yaw toward the input heading and
//! feeds the travel speed to its `AnimationGraph` parameter so a locomotion
//! blendspace picks the gait. Displacement comes from the clips' root motion,
//! or from the controller itself in `direct` drive; either way
//! `PhysicsSystem` resolves it against the scene on the next step.
//!
//! A followed prop is steered through the `RollingControl` its
//! `RollingDrive` gives it: the heading along the ground, the sprint and the
//! jump, which `PhysicsSystem` turns into the push on its body.

use alloc::string::String;

use crate::camera::probe::Followed;
use crate::components::{
    AnimationGraph, AnimationParams, Camera3D, CameraController, CameraProbe, CharacterRig,
    ControlsCommand, FollowDrive, FrameInput, RollingControl, Transform,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{
    Access, EventCursor, FrameTime, PipelineContext, SkinnedMeshHandle, StepResult, System,
};
use crate::gfx::camera::view_axes;
use crate::planet::{FrameRebases, PlanetFrame, RebaseCursor};
use concinnity_core::math;
use concinnity_core::math::vec3::{add, length, scale, sub};

// Seconds for the smoothed travel speed to close half the gap to its target.
const SPEED_HALF_LIFE: f32 = 0.12;
// Speeds below this snap to zero so the idle state is exactly at rest.
const SPEED_EPSILON: f32 = 0.01;

// Rotate `current` toward `desired` by at most `max_step` radians along the
// shorter arc. Angles wrap; the result is not normalized.
fn turn_toward(current: f32, desired: f32, max_step: f32) -> f32 {
    let diff = math::shortest_turn(current, desired, core::f32::consts::TAU);
    current + diff.clamp(-max_step, max_step)
}

// Away from the pull at `p`: the planet's up, or `+Y`.
fn up_at(ctx: &PipelineContext<'_>, p: [f32; 3]) -> [f32; 3] {
    ctx.resource::<PlanetFrame>()
        .map_or([0.0, 1.0, 0.0], |frame| frame.up_at(p))
}

// The camera-relative move the input asks for at `yaw` about `up`: along the
// ground, its length the throttle, which the keys hold at 1 and the stick
// sets by its deflection.
fn ground_move(input: &FrameInput, yaw: f32, up: [f32; 3]) -> [f32; 3] {
    let [right, _, back] = view_axes(yaw, 0.0, up);
    let fwd = scale(back, -1.0);
    let mut forward = input.move_axis[1];
    let mut strafe = input.move_axis[0];
    forward += f32::from(u8::from(input.forward)) - f32::from(u8::from(input.backward));
    strafe += f32::from(u8::from(input.right)) - f32::from(u8::from(input.left));
    let dir = add(scale(fwd, forward), scale(right, strafe));
    let norm = length(dir);
    if norm > 1.0 {
        scale(dir, 1.0 / norm)
    } else {
        dir
    }
}

/// Third-person controller behavior. Built by the host's system table from
/// the controlling `Camera3D`'s `CameraController`; never a world-declared
/// asset.
#[derive(Debug)]
pub struct ThirdPersonSystem {
    // From the camera controller (shared with the first-person modes).
    move_speed: f32,
    sprint_multiplier: f32,
    mouse_sensitivity: f32,
    // Gamepad look speed in radians per second at full stick deflection.
    gamepad_look_sensitivity: f32,
    // From the follow block: the authored SkinnedMesh handle, which keys the
    // rig / graph correlation web directly, or the followed prop.
    target: Option<SkinnedMeshHandle>,
    prop: Option<AssetId>,
    distance: f32,
    height: f32,
    lag_seconds: f32,
    drive: FollowDrive,
    turn_speed: f32,
    speed_parameter: String,
    jump_height: f32,
    // What is followed, resolved at init.
    followed: Option<Followed>,
    // Index of the speed parameter in the target graph's declaration order,
    // resolved once at init. `None` disables parameter writes.
    speed_param_index: Option<usize>,
    // Smoothed commanded travel speed (world units/second).
    speed: f32,
    // Orbit pivot, refreshed from what is followed each step; keeps the
    // camera stable if it ever disappears. Carried through frame moves.
    pivot: [f32; 3],
    rebases: RebaseCursor,
    controls_cursor: EventCursor,
}

impl ThirdPersonSystem {
    /// The controller for a world whose first controlled `Camera3D` has a
    /// `follow` block.
    pub fn for_world(world: &crate::ecs::World) -> Option<Self> {
        let controller = super::controlled_camera(world)?;
        controller.follow.is_some().then(|| Self::new(&controller))
    }

    /// Build from a `Camera3D`'s controller settings; the caller has already
    /// established that `controller.follow` is set.
    pub fn new(controller: &CameraController) -> Self {
        let follow = controller.follow.clone().unwrap_or_default();
        Self {
            move_speed: controller.move_speed,
            sprint_multiplier: controller.sprint_multiplier,
            mouse_sensitivity: controller.mouse_sensitivity,
            gamepad_look_sensitivity: crate::settings::DEFAULT_GAMEPAD_LOOK_SENSITIVITY,
            target: follow.target,
            prop: follow.prop.map(crate::ecs::Ref::id),
            distance: follow.distance.max(0.1),
            height: follow.height,
            lag_seconds: follow.lag_seconds.max(0.0),
            drive: follow.drive,
            turn_speed: follow.turn_speed.max(0.0),
            speed_parameter: follow.speed_parameter,
            jump_height: follow.jump_height.max(0.0),
            followed: None,
            speed_param_index: None,
            speed: 0.0,
            pivot: [0.0; 3],
            rebases: RebaseCursor::default(),
            controls_cursor: EventCursor::default(),
        }
    }

    // Where the followed rig or prop stands, before the pivot's height.
    fn followed_position(&self, ctx: &PipelineContext<'_>) -> Option<[f32; 3]> {
        match self.followed? {
            Followed::Rig(target) => ctx
                .query::<CharacterRig>()
                .find(|r| r.target == target)
                .map(|r| r.position),
            Followed::Prop(entity) => ctx.get::<Transform>(entity).map(|t| t.position),
        }
    }

    // Resolve the speed parameter to its declaration index, while the
    // target's AnimationGraph component still exists (AnimationSystem drains
    // it during its own init, which runs after this one).
    fn resolve_speed_parameter(&mut self, ctx: &PipelineContext<'_>, target: SkinnedMeshHandle) {
        if self.speed_parameter.is_empty() {
            return;
        }
        self.speed_param_index = ctx
            .query::<AnimationGraph>()
            .find(|g| g.target == Some(target))
            .and_then(|g| {
                g.parameters
                    .iter()
                    .position(|p| p.name == self.speed_parameter)
            });
        if self.speed_param_index.is_none() {
            tracing::warn!(
                "ThirdPersonSystem: no AnimationGraph parameter '{}' on follow target {target:?}, \
                 speed writes disabled",
                self.speed_parameter
            );
        }
    }

    // Steer the followed rig along `dir` (unit, or zero when idle) at the
    // smoothed speed, and hand physics any jump.
    #[expect(
        clippy::float_cmp,
        reason = "zero direction is no input; an unchanged yaw writes nothing"
    )]
    fn steer_rig(
        &self,
        ctx: &mut PipelineContext<'_>,
        target: SkinnedMeshHandle,
        dir: [f32; 3],
        input: &FrameInput,
        dt: f32,
    ) {
        if let Some(rig) = ctx.query_mut::<CharacterRig>().find(|r| r.target == target) {
            if dir != [0.0; 3] {
                let desired = math::heading_of(dir);
                let turned = turn_toward(rig.yaw, desired, self.turn_speed * dt);
                if turned != rig.yaw {
                    rig.yaw = turned;
                    rig.moved = true;
                }
            }
            rig.desired_move = match self.drive {
                // The clips carry the displacement; the capsule only
                // needs the facing and the speed parameter.
                FollowDrive::RootMotion => [0.0; 3],
                // In-place clips: move the capsule along the facing so
                // the character travels where it visually walks.
                FollowDrive::Direct => {
                    let [x, _, z] = math::flat_forward(rig.yaw);
                    [x * self.speed, 0.0, z * self.speed]
                }
            };
            if input.jump && self.jump_height > 0.0 {
                rig.jump_velocity = math::sqrt(2.0 * crate::physics::GRAVITY * self.jump_height);
            }
        }
        if let Some(index) = self.speed_param_index
            && let Some(params) = ctx
                .query_mut::<AnimationParams>()
                .find(|p| p.target == target)
        {
            params.set(index, self.speed);
        }
    }
}

impl System for ThirdPersonSystem {
    fn access(&self) -> Access {
        Access::new()
            .reads_components(crate::component_mask![FrameInput, Transform])
            .writes_components(crate::component_mask![
                Camera3D,
                CharacterRig,
                AnimationParams,
                CameraProbe,
                RollingControl,
            ])
            .reads_resources(crate::resource_mask![
                ControlsCommand,
                FrameTime,
                PlanetFrame,
                FrameRebases
            ])
    }

    fn init(&mut self, ctx: &mut PipelineContext<'_>) {
        super::look_controls::apply_persisted(
            ctx,
            super::look_controls::Look {
                mouse_sensitivity: &mut self.mouse_sensitivity,
                gamepad_look_sensitivity: &mut self.gamepad_look_sensitivity,
            },
        );
        self.rebases = RebaseCursor::at(ctx.resource::<FrameRebases>());

        self.followed = if let Some(target) = self.target {
            self.resolve_speed_parameter(ctx, target);
            Some(Followed::Rig(target))
        } else if let Some(id) = self.prop {
            let entity = ctx.entity_of(id);
            if entity.is_none() {
                tracing::warn!(
                    "ThirdPersonSystem: the followed prop ({id}) is missing, controller idle"
                );
            }
            entity.map(Followed::Prop)
        } else {
            tracing::warn!("ThirdPersonSystem: follow has no target or prop, controller idle");
            None
        };
        let Some(followed) = self.followed else {
            return;
        };

        // Seed the orbit pivot from where the followed thing was placed.
        if let Some(at) = self.followed_position(ctx) {
            self.pivot = add(at, scale(up_at(ctx, at), self.height));
        }

        // Occlusion probe: PhysicsSystem raycasts pivot-to-camera each frame
        // and reports the largest unobstructed distance, so walls never cut
        // between the camera and what it follows.
        ctx.push(CameraProbe {
            followed,
            pivot: self.pivot,
            desired: self.pivot,
            clearance: None,
        });
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        // Live settings-menu changes sent this tick by the renderer, which runs
        // first. FOV is written in the camera loop below, which holds the
        // mutable Camera3D borrow.
        let pending_fov = super::look_controls::drain_commands(
            ctx,
            &mut self.controls_cursor,
            super::look_controls::Look {
                mouse_sensitivity: &mut self.mouse_sensitivity,
                gamepad_look_sensitivity: &mut self.gamepad_look_sensitivity,
            },
        );
        if let Some(rebase) = self.rebases.take(ctx.resource::<FrameRebases>()) {
            self.pivot = rebase.apply_point(self.pivot);
        }

        // Read (not drain) the input snapshot deposited by the renderer, so
        // UiInputSystem can read the same snapshot. Movement fields are frozen
        // while a menu is open, which parks what is followed and the orbit.
        let Some(input) = ctx.query::<FrameInput>().next().cloned() else {
            return StepResult::Continue;
        };

        let dt = ctx
            .resource::<FrameTime>()
            .copied()
            .unwrap_or_default()
            .dt
            .clamp(0.0, 0.1);

        // Orbit angles from the mouse. The camera component keeps the
        // authoritative yaw/pitch; read it, advance, write back at the end.
        let Some((mut yaw, mut pitch)) = ctx.query::<Camera3D>().next().map(|c| (c.yaw, c.pitch))
        else {
            return StepResult::Continue;
        };
        // Pixel-based mouse deltas plus the rate-based right stick (deflection
        // x radians/second x dt, so orbiting is frame-rate correct).
        yaw -= input.mouse_dx * self.mouse_sensitivity
            + input.look_axis[0] * self.gamepad_look_sensitivity * dt;
        pitch = (pitch
            - input.mouse_dy * self.mouse_sensitivity
            - input.look_axis[1] * self.gamepad_look_sensitivity * dt)
            .clamp(
                -core::f32::consts::FRAC_PI_2 + 0.01,
                core::f32::consts::FRAC_PI_2 - 0.01,
            );

        let at = self.followed_position(ctx);
        let up = up_at(ctx, at.unwrap_or(self.pivot));
        let dir = ground_move(&input, yaw, up);
        let throttle = length(dir);

        // Smooth the commanded speed toward the input target, time-correct.
        let target_speed = if throttle > 1.0e-4 {
            self.move_speed
                * throttle
                * if input.sprint {
                    self.sprint_multiplier
                } else {
                    1.0
                }
        } else {
            0.0
        };
        let decay = 1.0 - math::half_life_decay(dt, SPEED_HALF_LIFE);
        self.speed += (target_speed - self.speed) * decay;
        if self.speed < SPEED_EPSILON && target_speed == 0.0 {
            self.speed = 0.0;
        }

        match self.followed {
            Some(Followed::Rig(target)) => {
                let heading = if throttle > 1.0e-4 {
                    scale(dir, 1.0 / throttle)
                } else {
                    [0.0; 3]
                };
                self.steer_rig(ctx, target, heading, &input, dt);
            }
            Some(Followed::Prop(entity)) => {
                if let Some(control) = ctx.get_mut::<RollingControl>(entity) {
                    control.heading = dir;
                    control.sprint = input.sprint;
                    control.jump |= input.jump;
                }
            }
            None => {}
        }

        // The pivot rides `height` above what is followed, closing on it by
        // the lag's half life.
        if let Some(at) = at {
            let goal = add(at, scale(up, self.height));
            self.pivot = if self.lag_seconds > 0.0 {
                let close = 1.0 - math::half_life_decay(dt, self.lag_seconds);
                add(self.pivot, scale(sub(goal, self.pivot), close))
            } else {
                goal
            };
        }

        // Place the camera on the orbit sphere behind the pivot, pulled in
        // to the occlusion probe's clearance (answered by PhysicsSystem this
        // frame from last frame's ray) so a wall never cuts the view, and
        // commit the pose. The player-capsule intents stay cleared: in third
        // person what is followed moves, not a camera capsule.
        let look = scale(view_axes(yaw, pitch, up)[2], -1.0);
        let desired = sub(self.pivot, scale(look, self.distance));
        let mut distance = self.distance;
        if let Some(probe) = ctx
            .query_mut::<CameraProbe>()
            .find(|p| Some(p.followed) == self.followed)
        {
            if let Some(clearance) = probe.clearance {
                distance = distance.min(clearance);
            }
            probe.pivot = self.pivot;
            probe.desired = desired;
        }
        let position = sub(self.pivot, scale(look, distance));
        for camera in ctx.query_mut::<Camera3D>() {
            if let Some(fov) = pending_fov {
                camera.fov_y_degrees = fov;
            }
            camera.up = up;
            camera.desired_move = [0.0; 3];
            camera.jump_requested = false;
            camera.interact_requested = false;
            camera.set_pose(position, yaw, pitch);
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{FollowController, PhysicsConfig};
    use crate::ecs::World;
    use crate::gfx::render_types::SkinnedIndex;
    use crate::test_support::from_json;
    use crate::transform;
    use alloc::string::ToString;
    use alloc::vec;
    use concinnity_core::math::SaturatingFrom;

    #[test]
    fn turn_toward_takes_the_short_arc_and_clamps() {
        // A quarter turn limited to 0.1 rad advances exactly 0.1.
        let stepped = turn_toward(0.0, core::f32::consts::FRAC_PI_2, 0.1);
        assert!((stepped - 0.1).abs() < 1e-6);
        // Reachable within the step: lands exactly on the target.
        let landed = turn_toward(0.0, 0.05, 0.1);
        assert!((landed - 0.05).abs() < 1e-6);
        // Wrap-around: from just below +pi to just above -pi is a short hop
        // forward across the seam, not a full turn back.
        let across = turn_toward(3.0, -3.0, 0.5);
        assert!(across > 3.0, "{across} should cross the seam forward");
    }

    // The handle the authored "hero" reference resolves to under the test
    // resolvers, which map a name to its byte length.
    fn hero() -> SkinnedMeshHandle {
        SkinnedMeshHandle::new(u32::saturating_from("hero".len()))
    }

    fn follow_controller(drive: FollowDrive, jump_height: f32) -> CameraController {
        CameraController {
            move_speed: 2.0,
            follow: Some(FollowController {
                target: Some(hero()),
                distance: 4.0,
                height: 1.5,
                drive,
                // Effectively instant turns, so heading asserts are exact.
                turn_speed: 100.0,
                speed_parameter: "speed".to_string(),
                jump_height,
                ..FollowController::default()
            }),
            ..CameraController::default()
        }
    }

    // A world with a followed rig: a seeded CharacterRig (the renderer would
    // publish it in a rendering world) and an AnimationGraph declaring the
    // speed parameter, which the test seeds the AnimationParams block for.
    fn follow_world(drive: FollowDrive, jump_height: f32, physics: bool) -> World {
        let controller = follow_controller(drive, jump_height);
        let mut world = World::new();
        world.add_component(Camera3D {
            fov_y_degrees: 75.0,
            near: 0.05,
            view_distance: None,
            view_matrix: [[0.0; 4]; 4],
            position: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            up: [0.0, 1.0, 0.0],
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller: Some(controller),
        });
        world.add_component(CharacterRig::new(
            hero(),
            SkinnedIndex(0),
            transform::IDENTITY,
            0.5,
            0.3,
        ));
        let graph: AnimationGraph =
            from_json(r#"{"target": "hero", "parameters": [{"name": "speed", "default": 0.0}]}"#);
        world.add_component(graph);
        world.add_component(AnimationParams::new(hero(), vec![0.0]));
        if physics {
            world.add_component(PhysicsConfig::default());
        }
        world
            .start(crate::ecs::HEADLESS_SYSTEMS)
            .expect("the test world starts");
        world
    }

    fn step_held(world: &mut World, input: FrameInput, steps: usize) {
        world.add_component(input);
        world.insert_resource(FrameTime {
            dt: 0.005,
            elapsed: 0.0,
        });
        for _ in 0..steps {
            world.step();
        }
    }

    // Holding forward ramps the speed parameter, drives the capsule along
    // the camera forward (-Z at yaw 0) in direct drive, and keeps the camera
    // on the orbit sphere behind the pivot.
    #[test]
    fn forward_input_steers_rig_params_and_camera() {
        let mut world = follow_world(FollowDrive::Direct, 0.0, false);
        step_held(
            &mut world,
            FrameInput {
                forward: true,
                ..Default::default()
            },
            4,
        );

        let rig = world.query::<CharacterRig>().next().expect("rig survives");
        assert!(rig.yaw.abs() < 1e-4, "forward at yaw 0 keeps heading 0");
        assert!(
            rig.desired_move[2] < -1e-3,
            "direct drive pushes along -Z: {:?}",
            rig.desired_move
        );
        let params = world
            .query::<AnimationParams>()
            .find(|p| p.target == hero())
            .expect("params survive");
        assert!(
            params.values[0] > 1e-3,
            "speed parameter ramped: {:?}",
            params.values
        );
        let camera = world.query::<Camera3D>().next().unwrap();
        // Pivot is the rig position raised by `height`; the camera sits
        // `distance` behind it along +Z (looking down -Z).
        assert!((camera.position[1] - (rig.position[1] + 1.5)).abs() < 1e-3);
        assert!((camera.position[2] - (rig.position[2] + 4.0)).abs() < 1e-3);
    }

    // Strafing right turns the character toward the input heading; root
    // motion drive leaves the capsule displacement to the clips.
    #[test]
    fn strafe_turns_heading_and_root_motion_drive_stays_passive() {
        let mut world = follow_world(FollowDrive::RootMotion, 0.0, false);
        // The turn budget is turn_speed x accumulated dt, so give the quarter
        // turn several times the steps it needs (the arrival assert is exact).
        step_held(
            &mut world,
            FrameInput {
                right: true,
                ..Default::default()
            },
            12,
        );

        let rig = world.query::<CharacterRig>().next().unwrap();
        assert!(
            (rig.yaw + core::f32::consts::FRAC_PI_2).abs() < 1e-3,
            "heading turned to -pi/2 (world +X): {}",
            rig.yaw
        );
        assert!(rig.moved, "a turn marks the render follow");
        assert_eq!(rig.desired_move, [0.0; 3], "root motion drives the capsule");
    }

    // A jump press hands the rig its takeoff velocity for physics to consume.
    #[test]
    fn jump_press_sets_takeoff_velocity() {
        let mut world = follow_world(FollowDrive::RootMotion, 1.0, false);
        step_held(
            &mut world,
            FrameInput {
                jump: true,
                ..Default::default()
            },
            1,
        );
        let rig = world.query::<CharacterRig>().next().unwrap();
        // v = sqrt(2 g h) with g = 20, h = 1.
        assert!(
            (rig.jump_velocity - math::sqrt(2.0 * crate::physics::GRAVITY)).abs() < 1e-4,
            "{}",
            rig.jump_velocity
        );
    }

    // End to end with physics: direct drive moves the capsule through
    // PhysicsSystem (controller writes desired_move, physics resolves it
    // against the flat floor on the next step).
    #[test]
    fn direct_drive_moves_the_capsule_through_physics() {
        let mut world = follow_world(FollowDrive::Direct, 0.0, true);
        step_held(
            &mut world,
            FrameInput {
                forward: true,
                ..Default::default()
            },
            8,
        );

        let rig = world.query::<CharacterRig>().next().unwrap();
        assert!(
            rig.position[2] < -1e-4,
            "capsule advanced along -Z: {:?}",
            rig.position
        );
        assert!(
            rig.position[1] > -0.2,
            "flat floor holds the capsule up: {:?}",
            rig.position
        );
    }

    // The floor must hold the capsule indefinitely, not just for a few
    // steps. (Regression: without the first-person player capsule the
    // shared character controller was never configured, and the rig sank
    // through the flat slab at a fraction of a unit per second.)
    #[test]
    fn idle_rig_rests_on_the_floor() {
        let mut world = follow_world(FollowDrive::RootMotion, 0.0, true);
        step_held(&mut world, FrameInput::default(), 120);

        let rig = world.query::<CharacterRig>().next().unwrap();
        assert!(
            (-0.02..0.08).contains(&rig.position[1]),
            "capsule settles onto the slab top: {:?}",
            rig.position
        );
    }

    // A follow camera on a prop on a planet's side: its world, the prop's
    // entity, and up there.
    mod on_a_sphere {
        use super::*;
        use crate::components::cook::Camera3D as Camera3DArgs;
        use crate::components::{RollingDrive, Transform};
        use crate::ecs::asset_id::AssetId;
        use crate::ecs::{Entity, Ref};
        use crate::planet::{Continents, LocalFrame, PlanetShape};
        use concinnity_core::approx::close;
        use concinnity_core::math::vec3::{dot, normalize_or};

        const SHAPE: PlanetShape = PlanetShape {
            center: [0.0, -5_000.0, 0.0],
            radius: 5_000.0,
            amplitude: 0.0,
            feature_size: 100.0,
            octaves: 1,
            seed: 0,
            continents: Continents::NONE,
            edits: None,
        };
        const BALL: AssetId = AssetId(3);

        fn rig(lag_seconds: f32) -> (World, ThirdPersonSystem, Entity, [f32; 3]) {
            let mut world = World::new();
            world.insert_resource(PlanetFrame {
                shape: SHAPE,
                frame: LocalFrame::AUTHORED,
            });
            world.insert_resource(FrameRebases::default());
            world.insert_resource(FrameTime {
                dt: 1.0 / 60.0,
                elapsed: 0.0,
            });
            let at = [600.0, -36.0, 0.0];
            let ball = world.push(Transform {
                position: at,
                ..Default::default()
            });
            world.identify(ball, BALL);
            world.insert(ball, RollingControl::new(&RollingDrive::default()));
            let controller = CameraController {
                follow: Some(FollowController {
                    prop: Some(Ref::new(BALL)),
                    distance: 6.0,
                    height: 1.0,
                    lag_seconds,
                    ..FollowController::default()
                }),
                ..CameraController::default()
            };
            world.push(Camera3D::bake(Camera3DArgs {
                pitch: -0.3,
                controller: Some(controller.clone()),
                ..Default::default()
            }));
            let mut system = ThirdPersonSystem::new(&controller);
            system.init(&mut world.context());
            let up = PlanetFrame {
                shape: SHAPE,
                frame: LocalFrame::AUTHORED,
            }
            .up_at(at);
            (world, system, ball, up)
        }

        fn frame(world: &mut World, system: &mut ThirdPersonSystem, input: FrameInput) {
            for stale in world.context().query_mut::<FrameInput>() {
                *stale = FrameInput::default();
            }
            if world.query::<FrameInput>().next().is_none() {
                world.add_component(FrameInput::default());
            }
            *world.context().query_mut::<FrameInput>().next().unwrap() = input;
            system.step(&mut world.context());
        }

        // The camera stands up its own way on the planet's side, orbits the
        // pivot `height` above the prop along that up, and looks at it.
        #[test]
        fn the_camera_orbits_the_prop_about_the_planets_up() {
            let (mut world, mut system, ball, up) = rig(0.0);
            frame(&mut world, &mut system, FrameInput::default());
            let camera = world.query::<Camera3D>().next().unwrap();
            assert!(close(camera.up, up, 1e-6), "{:?} vs {up:?}", camera.up);
            let at = world.get::<Transform>(ball).unwrap().position;
            let pivot = add(at, scale(up, 1.0));
            let back = sub(camera.position, pivot);
            assert!((length(back) - 6.0).abs() < 1e-3, "{back:?}");
            let look = scale(view_axes(camera.yaw, camera.pitch, up)[2], -1.0);
            assert!(close(
                normalize_or(back, 1e-6, [0.0; 3]),
                scale(look, -1.0),
                1e-4
            ));
            // Pitched down, the camera is above the pivot along up.
            assert!(dot(back, up) > 1.0, "{back:?}");
        }

        // Forward steers the prop along the ground the way the camera faces,
        // and a jump press is handed to its control.
        #[test]
        fn forward_steers_the_prop_along_the_ground_the_camera_faces() {
            let (mut world, mut system, ball, up) = rig(0.0);
            frame(
                &mut world,
                &mut system,
                FrameInput {
                    forward: true,
                    sprint: true,
                    jump: true,
                    ..Default::default()
                },
            );
            let control = *world.get::<RollingControl>(ball).unwrap();
            assert!((length(control.heading) - 1.0).abs() < 1e-5);
            assert!(dot(control.heading, up).abs() < 1e-5, "along the ground");
            let camera = world.query::<Camera3D>().next().unwrap();
            let look = scale(view_axes(camera.yaw, camera.pitch, up)[2], -1.0);
            let flat = sub(look, scale(up, dot(look, up)));
            let facing = normalize_or(flat, 1e-6, [0.0; 3]);
            assert!(
                close(control.heading, facing, 1e-4),
                "{:?}",
                control.heading
            );
            assert!(control.sprint && control.jump);
            // The jump stays asked for until the physics takes it.
            frame(&mut world, &mut system, FrameInput::default());
            assert!(world.get::<RollingControl>(ball).unwrap().jump);
        }

        // A frame move carries a lagging pivot with everything else, so the
        // camera holds its place behind the prop rather than swinging after
        // where the prop used to be.
        #[test]
        fn a_frame_move_carries_the_lagging_pivot() {
            let (mut world, mut system, ball, _) = rig(0.2);
            for _ in 0..240 {
                frame(&mut world, &mut system, FrameInput::default());
            }
            let offset = |world: &World| {
                let at = world.get::<Transform>(ball).unwrap().position;
                sub(world.query::<Camera3D>().next().unwrap().position, at)
            };
            let before = offset(&world);
            let at = world.get::<Transform>(ball).unwrap().position;
            let authored = LocalFrame::AUTHORED;
            let (moved, rebase) = authored.recentered(at, SHAPE.up_at(authored.to_world(at)));
            world.insert_resource(PlanetFrame {
                shape: SHAPE,
                frame: moved,
            });
            let mut ctx = world.context();
            for t in ctx.query_mut::<Transform>() {
                t.position = rebase.apply_point(t.position);
            }
            for camera in ctx.query_mut::<Camera3D>() {
                camera.position = rebase.apply_point(camera.position);
            }
            let mut rebases = FrameRebases::default();
            rebases.push(rebase);
            world.insert_resource(rebases);
            frame(&mut world, &mut system, FrameInput::default());
            let after = offset(&world);
            let expected = rebase.apply_vector(before);
            assert!(close(after, expected, 0.05), "{after:?} vs {expected:?}");
        }
    }
}
