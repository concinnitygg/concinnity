//! First-person / fly-through camera controller. An internal system (not a
//! declarable asset): `World::start` constructs one when the world has a
//! `Camera3D` whose `controller` is set, reading that controller's config. It
//! turns mouse/keyboard input into a `Camera3D` orientation and a movement
//! intent for the player's `RigidBody`.

use alloc::vec::Vec;

use crate::components::{
    Camera3D, CameraController, ControlsCommand, FrameInput, InteractEvent, Interactable, Transform,
};
use crate::ecs::{Access, Entity, EventCursor, FrameTime, PipelineContext, StepResult, System};
use crate::gfx::camera;
use crate::planet::{FrameRebases, RebaseCursor};
use concinnity_core::math;

// Reach distance for interacting with a Prop, in world units.
const INTERACT_REACH: f32 = 3.0;
// Minimum facing dot product (~60-degree cone) for an interaction.
const INTERACT_MIN_DOT: f32 = 0.5;

/// First-person / fly-through controller behavior. Constructed internally by
/// `World::start` from the controlling `Camera3D`'s `CameraController`;
/// never a world-declared asset.
#[derive(Debug)]
pub struct Camera3DSystem {
    free_fly: bool,
    move_speed: f32,
    sprint_multiplier: f32,
    mouse_sensitivity: f32,
    // Gamepad look speed in radians per second at full stick deflection
    // (rate-based, unlike the per-pixel mouse sensitivity).
    gamepad_look_sensitivity: f32,
    player_radius: f32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    // smoothed horizontal velocity; lerped toward the target each tick so
    // WASD movement accelerates and decelerates instead of snapping
    velocity: [f32; 3],
    // Interactable entities (those carrying the Interactable tag), collected at
    // init so step() rotates only their Transforms on interact.
    interactable_entities: Vec<Entity>,
    // Cursor into the Events<ControlsCommand> queue (live settings changes).
    controls_cursor: EventCursor,
    // How far the smoothed velocity has caught up with the moves of a planet
    // world's simulated frame.
    rebases: RebaseCursor,
}

impl Camera3DSystem {
    /// The controller for a world whose first controlled `Camera3D` has no
    /// `follow` block.
    pub fn for_world(world: &crate::ecs::World) -> Option<Self> {
        let controller = super::controlled_camera(world)?;
        controller.follow.is_none().then(|| Self::new(&controller))
    }

    /// Build a controller from a `Camera3D`'s controller settings.
    pub fn new(c: &CameraController) -> Self {
        Self {
            free_fly: c.free_fly,
            move_speed: c.move_speed,
            sprint_multiplier: c.sprint_multiplier,
            mouse_sensitivity: c.mouse_sensitivity,
            gamepad_look_sensitivity: crate::settings::DEFAULT_GAMEPAD_LOOK_SENSITIVITY,
            player_radius: c.player_radius,
            bounds_min: c.bounds_min,
            bounds_max: c.bounds_max,
            velocity: [0.0; 3],
            interactable_entities: Vec::new(),
            controls_cursor: EventCursor::default(),
            rebases: RebaseCursor::default(),
        }
    }

    /// Zero the smoothed movement velocity. Called when an external source (the
    /// cn run -f `camera-set` command) teleports the camera, so free-fly velocity
    /// integration does not drift the new pose on the next step. Only reached
    /// from the dev tooling crate's debug drive, hence dead in a `--lib` build.
    pub fn reset_velocity(&mut self) {
        self.velocity = [0.0; 3];
    }
}

impl System for Camera3DSystem {
    fn access(&self) -> Access {
        Access::new()
            .reads_components(crate::component_mask![
                FrameInput,
                crate::components::Identity
            ])
            .writes_components(crate::component_mask![Camera3D, Transform])
            .reads_resources(crate::resource_mask![
                ControlsCommand,
                FrameTime,
                FrameRebases
            ])
            .writes_resources(crate::resource_mask![InteractEvent])
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

        // Collect interact targets: every entity carrying the Interactable tag.
        self.interactable_entities = ctx
            .query_with_entity::<Interactable>()
            .map(|(entity, _)| entity)
            .collect();

        let registered = self.interactable_entities.len();
        if registered > 0 {
            tracing::debug!(
                "Camera3DSystem: registered {} interactable prop(s)",
                registered
            );
        }
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

        // Read (not drain) the input snapshot deposited by the renderer this
        // frame, so UiInputSystem can read the same snapshot (e.g. for a pause
        // menu over this camera). The renderer clears it before the next push.
        let Some(input) = ctx.query::<FrameInput>().next().cloned() else {
            return StepResult::Continue;
        };

        // The velocity is a direction in the simulated frame, so it turns with
        // the frame when a planet world moves it.
        if let Some(rebase) = self.rebases.take(ctx.resource::<FrameRebases>()) {
            self.velocity = rebase.apply_vector(self.velocity);
        }

        let dt = ctx
            .resource::<FrameTime>()
            .copied()
            .unwrap_or_default()
            .dt
            .clamp(0.0, 0.1);

        // update every Camera3D in the world (normally exactly one)
        for camera in ctx.query_mut::<Camera3D>() {
            // A live FOV change (settings-menu slider) applies to the camera's
            // projection on the next rendered frame.
            if let Some(fov) = pending_fov {
                camera.fov_y_degrees = fov;
            }
            // Look: pixel-based mouse deltas plus the rate-based right stick
            // (deflection x radians/second x dt, so it is frame-rate correct).
            let look_dx = input.mouse_dx * self.mouse_sensitivity
                + input.look_axis[0] * self.gamepad_look_sensitivity * dt;
            let look_dy = input.mouse_dy * self.mouse_sensitivity
                + input.look_axis[1] * self.gamepad_look_sensitivity * dt;
            camera.yaw -= look_dx;
            camera.pitch = (camera.pitch - look_dy).clamp(
                -core::f32::consts::FRAC_PI_2 + 0.01,
                core::f32::consts::FRAC_PI_2 - 0.01,
            );

            let speed = if input.sprint {
                self.move_speed * self.sprint_multiplier
            } else {
                self.move_speed
            };

            // Two movement modes share the same input/decay/screen-matrix
            // outer loop; only the basis vectors and how velocity is
            // committed differ. Free-fly drives the camera position
            // directly and adds a vertical component; the FPS walker keeps
            // motion horizontal and delegates to PhysicsSystem.
            // The bases are built level about +Y, then tilted onto the camera's
            // own up, which on a planet points away from its center.
            let fwd = if self.free_fly {
                math::forward(camera.yaw, camera.pitch)
            } else {
                math::flat_forward(camera.yaw)
            };
            let right = math::flat_right(camera.yaw);
            let fwd = camera::tilt_to_up(camera.up, fwd);
            let right = camera::tilt_to_up(camera.up, right);

            // build the target velocity from current key state
            let mut target = [0.0_f32; 3];
            if input.forward {
                target[0] += fwd[0] * speed;
                target[1] += fwd[1] * speed;
                target[2] += fwd[2] * speed;
            }
            if input.backward {
                target[0] -= fwd[0] * speed;
                target[1] -= fwd[1] * speed;
                target[2] -= fwd[2] * speed;
            }
            if input.right {
                target[0] += right[0] * speed;
                target[1] += right[1] * speed;
                target[2] += right[2] * speed;
            }
            if input.left {
                target[0] -= right[0] * speed;
                target[1] -= right[1] * speed;
                target[2] -= right[2] * speed;
            }
            // The left stick rides the same bases: partial deflection walks
            // proportionally slower (the axis magnitude is at most 1).
            target[0] += (fwd[0] * input.move_axis[1] + right[0] * input.move_axis[0]) * speed;
            target[1] += (fwd[1] * input.move_axis[1] + right[1] * input.move_axis[0]) * speed;
            target[2] += (fwd[2] * input.move_axis[1] + right[2] * input.move_axis[0]) * speed;
            // Free-fly: jump is "rise"; no down key, descend by pitching down + W.
            if self.free_fly && input.jump {
                let up = camera::tilt_to_up(camera.up, [0.0, 1.0, 0.0]);
                target[0] += up[0] * speed;
                target[1] += up[1] * speed;
                target[2] += up[2] * speed;
            }

            // Exponential decay toward target, time-correct so frame rate does not
            // affect the feel. half_life controls how quickly speed builds/drops.
            let half_life = 0.08_f32; // seconds to reach ~50% of target speed
            let decay = 1.0 - math::half_life_decay(dt, half_life);
            self.velocity[0] += (target[0] - self.velocity[0]) * decay;
            self.velocity[1] += (target[1] - self.velocity[1]) * decay;
            self.velocity[2] += (target[2] - self.velocity[2]) * decay;

            if self.free_fly {
                // Apply directly; no PhysicsSystem, no bounds, no gravity.
                camera.position[0] += self.velocity[0] * dt;
                camera.position[1] += self.velocity[1] * dt;
                camera.position[2] += self.velocity[2] * dt;
                camera.desired_move = [0.0; 3];
                camera.jump_requested = false;
            } else {
                // soft containment: pull the camera back inside the bounds box.
                // PhysicsSystem owns the position, so this is a one-frame-lagged
                // correction applied before it runs.
                let r = self.player_radius;
                camera.position[0] =
                    camera.position[0].clamp(self.bounds_min[0] + r, self.bounds_max[0] - r);
                camera.position[2] =
                    camera.position[2].clamp(self.bounds_min[2] + r, self.bounds_max[2] - r);

                // hand the movement intent to PhysicsSystem, which resolves it
                // against the world and writes the final camera position back
                camera.desired_move = self.velocity;
                camera.jump_requested = input.jump;
            }
            camera.interact_requested = input.interact;

            // write the view matrix as a fallback for worlds with no
            // PhysicsSystem; PhysicsSystem overwrites it once it has moved.
            camera.recompose_view();
        }

        // interactable props: press the interact key while facing one to rotate
        // it 45 degrees. Pickup/drop is handled by PhysicsSystem. The target
        // rotation lives on the entity's Transform.
        if input.interact && !self.interactable_entities.is_empty() {
            let (cam_pos, cam_yaw) = ctx
                .query::<Camera3D>()
                .next()
                .map_or(([0.0; 3], 0.0), |c| (c.position, c.yaw));
            let fwd = math::flat_forward(cam_yaw);

            // nearest interactable entity within reach the player faces
            let mut best: Option<(f32, Entity)> = None;
            for &entity in &self.interactable_entities {
                if let Some(t) = ctx.get::<Transform>(entity) {
                    let dx = t.position[0] - cam_pos[0];
                    let dz = t.position[2] - cam_pos[2];
                    let dist = math::sqrt(dx * dx + dz * dz);
                    if dist < INTERACT_REACH && dist > 0.0 {
                        let dot = (fwd[0] * dx + fwd[2] * dz) / dist;
                        if dot > INTERACT_MIN_DOT && best.is_none_or(|(d, _)| dist < d) {
                            best = Some((dist, entity));
                        }
                    }
                }
            }
            if let Some((_, entity)) = best {
                if let Some(t) = ctx.get_mut::<Transform>(entity) {
                    t.rotation_deg[1] = (t.rotation_deg[1] + 45.0) % 360.0;
                    tracing::info!(
                        "interacted with prop, yaw now {:.0}\u{00b0}",
                        t.rotation_deg[1]
                    );
                }
                // Announce the press for declarative logic (Behavior interact
                // sources); an unnamed entity has no addressable identity to
                // announce.
                let target = ctx
                    .get::<crate::components::Identity>(entity)
                    .map(|identity| identity.id());
                if let Some(target) = target {
                    ctx.events_mut::<InteractEvent>()
                        .send(InteractEvent { target });
                }
            }
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::Camera3DSystem;
    use crate::components::{Camera3D, CameraController, ControlsCommand, FrameInput};
    use crate::ecs::World;

    fn camera(controller: Option<CameraController>) -> Camera3D {
        Camera3D {
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
            controller,
        }
    }

    // Start `world` on the headless table, which builds the controller from
    // the world's camera.
    fn start(world: &mut World) {
        world
            .start(crate::ecs::HEADLESS_SYSTEMS)
            .expect("the test world starts");
    }

    // A ControlsCommand pushed mid-tick updates the live mouse sensitivity, so
    // the same frame's mouse-look uses the new value (not the init-time one).
    // This is the settings-menu sensitivity slider applying without a restart.
    #[test]
    fn controls_command_updates_sensitivity_live() {
        let mut world = World::new();
        // Free-fly avoids the PhysicsSystem path; start from a known sensitivity.
        let ctrl = CameraController {
            free_fly: true,
            mouse_sensitivity: 0.001,
            ..CameraController::default()
        };
        world.add_component(camera(Some(ctrl)));
        start(&mut world);

        // A mouse delta in the same frame must rotate by the NEW sensitivity
        // (0.005), not the controller's 0.001.
        world.events_mut::<ControlsCommand>().send(ControlsCommand {
            mouse_sensitivity: Some(0.005),
            ..Default::default()
        });
        world.add_component(FrameInput {
            mouse_dx: 10.0,
            ..Default::default()
        });
        world.step();

        let yaw = world.query::<Camera3D>().next().map(|c| c.yaw).unwrap();
        assert!(
            (yaw - (-10.0 * 0.005)).abs() < 1.0e-6,
            "yaw {yaw} should reflect the live sensitivity 0.005"
        );
    }

    // A ControlsCommand carrying a new FOV updates the Camera3D's fov_y_degrees
    // live, and an event with fov_y_degrees: None leaves the FOV untouched.
    #[test]
    fn controls_command_updates_fov_live() {
        let mut world = World::new();
        let ctrl = CameraController {
            free_fly: true,
            ..CameraController::default()
        };
        world.add_component(camera(Some(ctrl)));
        start(&mut world);
        let fov = |world: &World| world.query::<Camera3D>().next().map(|c| c.fov_y_degrees);
        assert_eq!(fov(&world), Some(75.0));

        world.events_mut::<ControlsCommand>().send(ControlsCommand {
            fov_y_degrees: Some(90.0),
            ..Default::default()
        });
        world.add_component(FrameInput::default());
        world.step();
        assert_eq!(fov(&world), Some(90.0), "the live FOV applies");

        world.events_mut::<ControlsCommand>().send(ControlsCommand {
            mouse_sensitivity: Some(0.004),
            ..Default::default()
        });
        world.step();
        assert_eq!(
            fov(&world),
            Some(90.0),
            "a sensitivity-only command leaves the FOV"
        );
    }

    // A grounded (non-free-fly) camera clamps its position back inside the
    // bounds box and hands the movement intent + jump to PhysicsSystem instead
    // of moving itself, driving the FPS-walker basis and commit branch.
    #[test]
    fn fps_walker_clamps_bounds_and_hands_off_movement() {
        use crate::gfx::camera::view_matrix;

        let mut world = World::new();
        let ctrl = CameraController {
            free_fly: false,
            move_speed: 5.0,
            player_radius: 0.5,
            bounds_min: [-10.0, -10.0, -10.0],
            bounds_max: [10.0, 10.0, 10.0],
            ..CameraController::default()
        };
        let mut cam = camera(Some(ctrl));
        // Well outside the +X wall so the containment clamp is observable.
        cam.position = [100.0, 2.0, 0.0];
        world.add_component(cam);
        start(&mut world);

        world.add_component(FrameInput {
            forward: true,
            jump: true,
            ..Default::default()
        });
        world.step();

        let c = world.query::<Camera3D>().next().unwrap();
        // Pulled back to the wall minus the player radius (10 - 0.5).
        assert!(
            (c.position[0] - 9.5).abs() < 1e-4,
            "x clamped: {}",
            c.position[0]
        );
        assert!(c.jump_requested, "jump intent handed off");
        assert_eq!(
            c.position[1], 2.0,
            "grounded walker leaves height to physics"
        );
        assert_eq!(c.view_matrix, view_matrix(c.position, c.yaw, c.pitch));
    }

    // reset_velocity zeroes the smoothed movement velocity so a teleport does
    // not drift the next step.
    #[test]
    fn reset_velocity_zeroes_smoothed_velocity() {
        let mut sys = Camera3DSystem::new(&CameraController::default());
        sys.velocity = [1.0, -2.0, 3.0];
        sys.reset_velocity();
        assert_eq!(sys.velocity, [0.0; 3]);
    }
}
