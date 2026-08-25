// src/gfx/third_person.rs
//
// Third-person character controller. An internal system (not a declarable
// asset): `World::start` constructs one instead of `Camera3DSystem` when the
// controlling `Camera3D`'s controller carries a `follow` block. The mouse
// orbits the camera around the followed character's `CharacterRig`; WASD
// steers the character camera-relative, turning its facing yaw and feeding
// the travel speed to its `AnimationGraph` parameter so a locomotion blendspace
// picks the gait. Displacement comes from the clips' root motion, or from
// the controller itself in `direct` drive; either way `PhysicsSystem`
// resolves it against the scene on the next step.

use crate::components::{
    AnimationGraph, AnimationParams, Camera3D, CameraController, CharacterRig, FollowDrive,
    FrameInput,
};
use crate::ecs::{PipelineContext, SkinnedMeshHandle, StepResult, System};
use std::time::Instant;

// Seconds for the smoothed travel speed to close half the gap to its target.
const SPEED_HALF_LIFE: f32 = 0.12;
// Speeds below this snap to zero so the idle state is exactly at rest.
const SPEED_EPSILON: f32 = 0.01;

// The heading yaw whose forward vector is `dir` (camera convention: yaw 0
// looks down -Z, so forward(yaw) = [-sin yaw, 0, -cos yaw]).
pub(crate) fn heading_of(dir: [f32; 3]) -> f32 {
    (-dir[0]).atan2(-dir[2])
}

// Rotate `current` toward `desired` by at most `max_step` radians along the
// shorter arc. Angles wrap; the result is not normalized.
pub(crate) fn turn_toward(current: f32, desired: f32, max_step: f32) -> f32 {
    let mut diff = (desired - current).rem_euclid(std::f32::consts::TAU);
    if diff > std::f32::consts::PI {
        diff -= std::f32::consts::TAU;
    }
    current + diff.clamp(-max_step, max_step)
}

#[derive(Debug)]
pub struct ThirdPersonSystem {
    // From the camera controller (shared with the first-person modes).
    move_speed: f32,
    sprint_multiplier: f32,
    mouse_sensitivity: f32,
    // Gamepad look speed in radians per second at full stick deflection.
    gamepad_look_sensitivity: f32,
    // From the follow block: the authored SkinnedMesh handle, which keys the
    // rig / graph correlation web directly.
    target: Option<SkinnedMeshHandle>,
    distance: f32,
    height: f32,
    drive: FollowDrive,
    turn_speed: f32,
    speed_parameter: String,
    jump_height: f32,
    // Index of the speed parameter in the target graph's declaration order,
    // resolved once at init. `None` disables parameter writes.
    speed_param_index: Option<usize>,
    // Smoothed commanded travel speed (world units/second).
    speed: f32,
    // Orbit pivot, refreshed from the rig each step; keeps the camera stable
    // if the rig ever disappears.
    pivot: [f32; 3],
    last_step: Option<Instant>,
    controls_cursor: crate::ecs::EventCursor,
}

impl ThirdPersonSystem {
    // Build from a `Camera3D`'s controller settings; the caller has already
    // established that `controller.follow` is set.
    pub fn new(controller: &CameraController) -> Self {
        let follow = controller.follow.clone().unwrap_or_default();
        Self {
            move_speed: controller.move_speed,
            sprint_multiplier: controller.sprint_multiplier,
            mouse_sensitivity: controller.mouse_sensitivity,
            gamepad_look_sensitivity: crate::gfx::settings::DEFAULT_GAMEPAD_LOOK_SENSITIVITY,
            target: follow.target,
            distance: follow.distance.max(0.1),
            height: follow.height,
            drive: follow.drive,
            turn_speed: follow.turn_speed.max(0.0),
            speed_parameter: follow.speed_parameter,
            jump_height: follow.jump_height.max(0.0),
            speed_param_index: None,
            speed: 0.0,
            pivot: [0.0; 3],
            last_step: None,
            controls_cursor: crate::ecs::EventCursor::default(),
        }
    }
}

impl System for ThirdPersonSystem {
    fn access(&self) -> crate::ecs::Access {
        crate::ecs::Access::new()
            .reads_components(crate::component_mask![crate::components::FrameInput])
            .writes_components(crate::component_mask![
                crate::components::Camera3D,
                crate::components::CharacterRig,
                crate::components::AnimationParams,
                crate::components::CameraProbe,
            ])
            .reads_resources(crate::resource_mask![crate::components::ControlsCommand])
    }

    fn init(&mut self, ctx: &mut PipelineContext) {
        self.last_step = Some(Instant::now());

        crate::gfx::look_controls::apply_persisted(
            ctx,
            crate::gfx::look_controls::Look {
                mouse_sensitivity: &mut self.mouse_sensitivity,
                gamepad_look_sensitivity: &mut self.gamepad_look_sensitivity,
            },
        );

        let Some(target) = self.target else {
            tracing::warn!("ThirdPersonSystem: follow has no target, controller idle");
            return;
        };

        // Resolve the speed parameter to its declaration index now, while the
        // target's AnimationGraph component still exists (AnimationSystem drains
        // it during its own init, which runs after this one).
        if !self.speed_parameter.is_empty() {
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

        // Seed the orbit pivot from the rig's authored placement.
        if let Some(rig) = ctx.query::<CharacterRig>().find(|r| r.target == target) {
            self.pivot = [
                rig.position[0],
                rig.position[1] + self.height,
                rig.position[2],
            ];
        }

        // Occlusion probe: PhysicsSystem raycasts pivot-to-camera each frame
        // and reports the largest unobstructed distance, so walls never cut
        // between the camera and the character.
        ctx.push(crate::components::CameraProbe {
            target,
            pivot: self.pivot,
            desired: self.pivot,
            clearance: None,
        });
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Live settings-menu changes sent this tick by GraphicsSystem, which runs
        // first. FOV is written in the camera loop below, which holds the
        // mutable Camera3D borrow.
        let pending_fov = crate::gfx::look_controls::drain_commands(
            ctx,
            &mut self.controls_cursor,
            crate::gfx::look_controls::Look {
                mouse_sensitivity: &mut self.mouse_sensitivity,
                gamepad_look_sensitivity: &mut self.gamepad_look_sensitivity,
            },
        );

        // Read (not drain) the input snapshot deposited by GraphicsSystem, so
        // UiInputSystem can read the same snapshot. Movement fields are frozen
        // while a menu is open, which parks the character and the orbit.
        let input = match ctx.query::<FrameInput>().next().cloned() {
            Some(i) => i,
            None => return StepResult::Continue,
        };

        let now = Instant::now();
        let dt = self
            .last_step
            .map(|t| now.duration_since(t).as_secs_f32().min(0.1))
            .unwrap_or(0.0);
        self.last_step = Some(now);

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
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );

        // Camera-relative movement intent on the ground plane.
        let fwd = [-yaw.sin(), 0.0_f32, -yaw.cos()];
        let right = [yaw.cos(), 0.0_f32, -yaw.sin()];
        let mut dir = [0.0_f32; 3];
        if input.forward {
            dir[0] += fwd[0];
            dir[2] += fwd[2];
        }
        if input.backward {
            dir[0] -= fwd[0];
            dir[2] -= fwd[2];
        }
        if input.right {
            dir[0] += right[0];
            dir[2] += right[2];
        }
        if input.left {
            dir[0] -= right[0];
            dir[2] -= right[2];
        }
        // The left stick rides the same bases; its magnitude survives into the
        // commanded speed below, so partial deflection sets a slower gait.
        dir[0] += fwd[0] * input.move_axis[1] + right[0] * input.move_axis[0];
        dir[2] += fwd[2] * input.move_axis[1] + right[2] * input.move_axis[0];
        let norm = (dir[0] * dir[0] + dir[2] * dir[2]).sqrt();
        let moving = norm > 1.0e-4;
        if moving {
            dir[0] /= norm;
            dir[2] /= norm;
        }

        // Smooth the commanded speed toward the input target, time-correct.
        // The key vectors have magnitude >= 1, so the clamp leaves keyboard
        // movement at full speed; a partially deflected stick scales it down.
        let target_speed = if moving {
            self.move_speed
                * norm.clamp(0.0, 1.0)
                * if input.sprint {
                    self.sprint_multiplier
                } else {
                    1.0
                }
        } else {
            0.0
        };
        let decay = 1.0 - 2.0_f32.powf(-dt / SPEED_HALF_LIFE);
        self.speed += (target_speed - self.speed) * decay;
        if self.speed < SPEED_EPSILON && target_speed == 0.0 {
            self.speed = 0.0;
        }

        // Drive the rig: turn toward the input heading, hand physics the
        // direct-drive velocity and any jump, and refresh the orbit pivot.
        if let Some(target) = self.target {
            if let Some(rig) = ctx.query_mut::<CharacterRig>().find(|r| r.target == target) {
                if moving {
                    let desired = heading_of(dir);
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
                    FollowDrive::Direct => [
                        -rig.yaw.sin() * self.speed,
                        0.0,
                        -rig.yaw.cos() * self.speed,
                    ],
                };
                if input.jump && self.jump_height > 0.0 {
                    rig.jump_velocity =
                        (2.0 * concinnity_physics::GRAVITY * self.jump_height).sqrt();
                }
                self.pivot = [
                    rig.position[0],
                    rig.position[1] + self.height,
                    rig.position[2],
                ];
            }

            if let Some(index) = self.speed_param_index
                && let Some(params) = ctx
                    .query_mut::<AnimationParams>()
                    .find(|p| p.target == target)
            {
                params.set(index, self.speed);
            }
        }

        // Place the camera on the orbit sphere behind the pivot, pulled in
        // to the occlusion probe's clearance (answered by PhysicsSystem this
        // frame from last frame's ray) so a wall never cuts the view, and
        // commit the pose. The player-capsule intents stay cleared: in third
        // person the character capsule moves, not a camera capsule.
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_pitch, cos_pitch) = pitch.sin_cos();
        let look = [-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch];
        let desired = [
            self.pivot[0] - look[0] * self.distance,
            self.pivot[1] - look[1] * self.distance,
            self.pivot[2] - look[2] * self.distance,
        ];
        let mut distance = self.distance;
        if let Some(probe) = ctx
            .query_mut::<crate::components::CameraProbe>()
            .find(|p| Some(p.target) == self.target)
        {
            if let Some(clearance) = probe.clearance {
                distance = distance.min(clearance);
            }
            probe.pivot = self.pivot;
            probe.desired = desired;
        }
        let position = [
            self.pivot[0] - look[0] * distance,
            self.pivot[1] - look[1] * distance,
            self.pivot[2] - look[2] * distance,
        ];
        for camera in ctx.query_mut::<Camera3D>() {
            if let Some(fov) = pending_fov {
                camera.fov_y_degrees = fov;
            }
            camera.yaw = yaw;
            camera.pitch = pitch;
            camera.position = position;
            camera.desired_move = [0.0; 3];
            camera.jump_requested = false;
            camera.interact_requested = false;
            camera.view_matrix = crate::gfx::camera::view_matrix(position, yaw, pitch);
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{FollowController, FrameInput};
    use crate::ecs::World;
    use crate::ecs::asset_id::intern;
    use std::time::Duration;

    #[test]
    fn heading_matches_the_camera_forward_convention() {
        // yaw 0 looks down -Z; strafing right is world +X at yaw 0.
        assert!(heading_of([0.0, 0.0, -1.0]).abs() < 1e-6);
        assert!((heading_of([1.0, 0.0, 0.0]) + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((heading_of([-1.0, 0.0, 0.0]) - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn turn_toward_takes_the_short_arc_and_clamps() {
        // A quarter turn limited to 0.1 rad advances exactly 0.1.
        let stepped = turn_toward(0.0, std::f32::consts::FRAC_PI_2, 0.1);
        assert!((stepped - 0.1).abs() < 1e-6);
        // Reachable within the step: lands exactly on the target.
        let landed = turn_toward(0.0, 0.05, 0.1);
        assert!((landed - 0.05).abs() < 1e-6);
        // Wrap-around: from just below +pi to just above -pi is a short hop
        // forward across the seam, not a full turn back.
        let across = turn_toward(3.0, -3.0, 0.5);
        assert!(across > 3.0, "{across} should cross the seam forward");
    }

    fn follow_camera(
        target: &str,
        drive: FollowDrive,
        jump_height: f32,
    ) -> crate::components::Camera3D {
        use crate::components::{Camera3D, CameraController};
        let controller = CameraController {
            move_speed: 2.0,
            follow: Some(FollowController {
                target: Some(SkinnedMeshHandle(intern(target).0)),
                distance: 4.0,
                height: 1.5,
                drive,
                // Effectively instant turns, so heading asserts are exact.
                turn_speed: 100.0,
                speed_parameter: "speed".to_string(),
                jump_height,
            }),
            ..CameraController::default()
        };
        Camera3D {
            fov_y_degrees: 75.0,
            near: 0.05,
            far: 200.0,
            view_matrix: [[0.0; 4]; 4],
            position: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller: Some(controller),
        }
    }

    // A world with a followed rig: the third-person controller, a seeded
    // CharacterRig (GraphicsSystem would publish it in a rendering world),
    // and an AnimationGraph declaring the speed parameter. AnimationSystem's init
    // fails the graph install (no clips) and that is fine: the controller
    // resolved its parameter index before the drain, and this test seeds the
    // AnimationParams block itself.
    fn follow_world(drive: FollowDrive, jump_height: f32) -> (World, SkinnedMeshHandle) {
        // The authored "hero" references below deserialize through the
        // resolver's interner fallback, so the handle carries the interned id;
        // the components this seeds must use the same value.
        let target = SkinnedMeshHandle(intern("hero").0);
        let mut world = World::new();
        world.add_component(follow_camera("hero", drive, jump_height));
        world.add_component(crate::components::CharacterRig::new(
            target,
            0,
            crate::gfx::skinning::IDENTITY,
            0.5,
            0.3,
        ));
        let graph: crate::components::AnimationGraph = serde_json::from_value(serde_json::json!({
            "target": "hero",
            "parameters": [{"name": "speed", "default": 0.0}],
        }))
        .unwrap();
        world.add_component(graph);
        world.add_component(crate::components::AnimationParams::new(target, vec![0.0]));
        (world, target)
    }

    fn step_held(world: &mut World, input: FrameInput, steps: usize) {
        world.add_component(input);
        for _ in 0..steps {
            world.step();
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    // The spawn gate: a follow block selects the third-person controller.
    #[test]
    fn follow_controller_spawns_third_person_system() {
        let (mut world, _) = follow_world(FollowDrive::RootMotion, 0.0);
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert!(names.contains(&"ThirdPersonSystem"), "{names:?}");
        assert!(!names.contains(&"Camera3DSystem"), "{names:?}");
    }

    // Holding forward ramps the speed parameter, drives the capsule along
    // the camera forward (-Z at yaw 0) in direct drive, and keeps the camera
    // on the orbit sphere behind the pivot.
    #[test]
    fn forward_input_steers_rig_params_and_camera() {
        let (mut world, target) = follow_world(FollowDrive::Direct, 0.0);
        world.start().unwrap();
        step_held(
            &mut world,
            FrameInput {
                forward: true,
                ..Default::default()
            },
            4,
        );

        let rig = world
            .query::<crate::components::CharacterRig>()
            .next()
            .expect("rig survives");
        assert!(rig.yaw.abs() < 1e-4, "forward at yaw 0 keeps heading 0");
        assert!(
            rig.desired_move[2] < -1e-3,
            "direct drive pushes along -Z: {:?}",
            rig.desired_move
        );
        let params = world
            .query::<crate::components::AnimationParams>()
            .find(|p| p.target == target)
            .expect("params survive");
        assert!(
            params.values[0] > 1e-3,
            "speed parameter ramped: {:?}",
            params.values
        );
        let camera = world.query::<crate::components::Camera3D>().next().unwrap();
        // Pivot is the rig position raised by `height`; the camera sits
        // `distance` behind it along +Z (looking down -Z).
        assert!((camera.position[1] - (rig.position[1] + 1.5)).abs() < 1e-3);
        assert!((camera.position[2] - (rig.position[2] + 4.0)).abs() < 1e-3);
    }

    // Strafing right turns the character toward the input heading; root
    // motion drive leaves the capsule displacement to the clips.
    #[test]
    fn strafe_turns_heading_and_root_motion_drive_stays_passive() {
        let (mut world, _) = follow_world(FollowDrive::RootMotion, 0.0);
        world.start().unwrap();
        // The turn budget is turn_speed x accumulated wall-clock dt, so give
        // the quarter turn several times the steps it needs: with tight 5 ms
        // sleeps 4 steps sat right at the pi/2 boundary and failed on fast
        // runners (the arrival assert below is exact).
        step_held(
            &mut world,
            FrameInput {
                right: true,
                ..Default::default()
            },
            12,
        );

        let rig = world
            .query::<crate::components::CharacterRig>()
            .next()
            .unwrap();
        assert!(
            (rig.yaw + std::f32::consts::FRAC_PI_2).abs() < 1e-3,
            "heading turned to -pi/2 (world +X): {}",
            rig.yaw
        );
        assert!(rig.moved, "a turn marks the render follow");
        assert_eq!(rig.desired_move, [0.0; 3], "root motion drives the capsule");
    }

    // A jump press hands the rig its takeoff velocity for physics to consume.
    #[test]
    fn jump_press_sets_takeoff_velocity() {
        let (mut world, _) = follow_world(FollowDrive::RootMotion, 1.0);
        world.start().unwrap();
        step_held(
            &mut world,
            FrameInput {
                jump: true,
                ..Default::default()
            },
            1,
        );
        let rig = world
            .query::<crate::components::CharacterRig>()
            .next()
            .unwrap();
        // v = sqrt(2 g h) with g = 20, h = 1.
        assert!(
            (rig.jump_velocity - (2.0 * concinnity_physics::GRAVITY).sqrt()).abs() < 1e-4,
            "{}",
            rig.jump_velocity
        );
    }

    // End to end with physics: direct drive moves the capsule through
    // PhysicsSystem (controller writes desired_move, physics resolves it
    // against the flat floor on the next step).
    #[test]
    fn direct_drive_moves_the_capsule_through_physics() {
        let (mut world, _) = follow_world(FollowDrive::Direct, 0.0);
        world.add_component(crate::components::PhysicsConfig::default());
        world.start().unwrap();
        step_held(
            &mut world,
            FrameInput {
                forward: true,
                ..Default::default()
            },
            8,
        );

        let rig = world
            .query::<crate::components::CharacterRig>()
            .next()
            .unwrap();
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

    // A wall between the character and the orbit camera pulls the camera in
    // front of it: the occlusion probe (answered by PhysicsSystem) clamps
    // the follow distance so the view is never cut.
    #[test]
    fn wall_behind_the_camera_pulls_it_in() {
        let (mut world, _) = follow_world(FollowDrive::RootMotion, 0.0);
        world.add_component(crate::components::PhysicsConfig::default());
        // A wall crossing the camera's line at z = +2 (the camera orbits to
        // z = +4 at yaw 0, the pivot sits at z = 0).
        world.add_component(crate::components::Prop {
            asset_id: intern("wall"),
            position: [0.0, 1.5, 2.0],
            collider: Some(crate::components::PropCollider {
                shape: "cuboid".to_string(),
                half_extents: [3.0, 1.5, 0.2],
                radius: 0.0,
                half_height: 0.0,
                layer: String::new(),
            }),
            ..Default::default()
        });
        world.start().unwrap();
        step_held(&mut world, FrameInput::default(), 6);

        let camera = world.query::<crate::components::Camera3D>().next().unwrap();
        assert!(
            camera.position[2] < 1.9,
            "camera pulled in front of the wall at z = 2: {:?}",
            camera.position
        );
        assert!(
            camera.position[2] > 0.3,
            "camera stays behind the pivot: {:?}",
            camera.position
        );
    }

    // The floor must hold the capsule indefinitely, not just for a few
    // steps. (Regression: without the first-person player capsule the
    // shared character controller was never configured, and the rig sank
    // through the flat slab at a fraction of a unit per second.)
    #[test]
    fn idle_rig_rests_on_the_floor() {
        let (mut world, _) = follow_world(FollowDrive::RootMotion, 0.0);
        world.add_component(crate::components::PhysicsConfig::default());
        world.start().unwrap();
        step_held(&mut world, FrameInput::default(), 120);

        let rig = world
            .query::<crate::components::CharacterRig>()
            .next()
            .unwrap();
        assert!(
            (-0.02..0.08).contains(&rig.position[1]),
            "capsule settles onto the slab top: {:?}",
            rig.position
        );
    }
}
