//! The verbs that pose the active camera: a one-shot teleport, which the next
//! frame draws as a camera cut, and a sustained per-frame motion the debug
//! drive re-applies until it runs out or is stopped. Both write the pose
//! straight onto the `Camera3D` and zero the controller velocity, so a
//! free-fly camera does not drift the pose away.

use concinnity_core::behavior::camera_driver;
use concinnity_core::components::Camera3D;
use concinnity_core::ecs::World;
use concinnity_core::render::history_reset::{HistoryResetCauses, PendingHistoryReset};
use concinnity_engine::controller::camera::Camera3DSystem;
use serde_json::json;

use crate::debug::call::Call;
use crate::debug::verb::{Access, Args, Kind, Reply, Verb, optional};

pub(in crate::debug) const VERBS: &[Verb] = &[
    Verb {
        name: "camera-set",
        description: "Teleport the active camera to a pose, optionally changing its field of view.",
        access: Access::Mutating,
        params: &[
            optional(
                "position",
                Kind::Vec3,
                "World-space position. Defaults to the origin.",
            ),
            optional("yaw", Kind::Number, "Yaw in radians. Defaults to zero."),
            optional("pitch", Kind::Number, "Pitch in radians. Defaults to zero."),
            optional(
                "fov_y_degrees",
                Kind::NumberOrNull,
                "Vertical field of view in degrees; omit to leave it untouched.",
            ),
        ],
        run: camera_set,
    },
    Verb {
        name: "camera-move",
        description: "Apply a per-frame camera pose delta over a span of frames so the renderer sees sustained motion.",
        access: Access::Mutating,
        params: &[
            optional(
                "forward",
                Kind::Number,
                "Per-frame offset along the look direction, in world units.",
            ),
            optional(
                "right",
                Kind::Number,
                "Per-frame offset along the right vector, in world units.",
            ),
            optional(
                "up",
                Kind::Number,
                "Per-frame offset along the up vector, in world units.",
            ),
            optional("yaw", Kind::Number, "Per-frame yaw delta in radians."),
            optional("pitch", Kind::Number, "Per-frame pitch delta in radians."),
            optional(
                "frames",
                Kind::Count,
                "How many frames to apply the delta for. Zero holds until camera-stop.",
            ),
        ],
        run: camera_move,
    },
    Verb {
        name: "camera-stop",
        description: "Clear any camera motion left running by camera-move.",
        access: Access::Mutating,
        params: &[],
        run: camera_stop,
    },
];

// A new pose for the active camera. `fov_y_degrees` is `None` to leave the
// field of view as it is.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default)]
struct CameraPose {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    fov_y_degrees: Option<f32>,
}

// A per-frame pose delta: `forward` / `right` / `up` offsets along the
// free-fly look basis, `yaw` / `pitch` radian deltas.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default)]
struct CameraStep {
    forward: f32,
    right: f32,
    up: f32,
    yaw: f32,
    pitch: f32,
    frames: u32,
}

// The debug tick runs before the world step, so the controller sees the new
// pose the same frame.
fn camera_set(call: &Call, args: Args) -> Reply {
    let pose: CameraPose = args.parse()?;
    call.on_world(move |world, _| apply_camera_set(&pose, world))?;
    Ok(json!({ "set": true }))
}

// The reply fires when the motion is installed, not when it finishes, so a long
// move never outlasts the call's wait.
fn camera_move(call: &Call, args: Args) -> Reply {
    let step: CameraStep = args.parse()?;
    let frames = step.frames;
    call.on_world(move |world, motion| {
        if world.query::<Camera3D>().next().is_none() {
            return Err("camera-move: no Camera3D in world".to_string());
        }
        *motion = Some(CameraMotion::from_step(&step));
        Ok(())
    })?;
    Ok(json!({ "frames": frames, "holding": frames == 0 }))
}

fn camera_stop(call: &Call, _: Args) -> Reply {
    call.on_world(|_, motion| {
        *motion = None;
        Ok(())
    })?;
    Ok(json!({ "stopped": true }))
}

/// An in-progress camera-move, held by the debug server and advanced once per
/// frame until exhausted or cleared by a camera-stop.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::debug) struct CameraMotion {
    forward: f32,
    right: f32,
    up: f32,
    yaw: f32,
    pitch: f32,
    // Frames still to apply. `None` is an indefinite hold (cleared only by
    // `camera-stop`); `Some(n)` counts down to the auto-stop.
    frames_left: Option<u32>,
}

impl CameraMotion {
    fn from_step(step: &CameraStep) -> Self {
        Self {
            forward: step.forward,
            right: step.right,
            up: step.up,
            yaw: step.yaw,
            pitch: step.pitch,
            frames_left: (step.frames > 0).then_some(step.frames),
        }
    }

    // The motion to apply next frame, after one step has just been applied: a
    // finite countdown decremented by one (returning `None` once exhausted),
    // an indefinite hold returning itself unchanged.
    fn advanced(self) -> Option<Self> {
        match self.frames_left {
            None => Some(self),
            Some(n) if n > 1 => Some(Self {
                frames_left: Some(n - 1),
                ..self
            }),
            Some(_) => None,
        }
    }
}

/// Apply one step of `motion` to the active camera and return the motion left
/// for the next frame. A world without a `Camera3D` drops the motion instead
/// of holding it forever.
pub(in crate::debug) fn advance_camera_motion(
    motion: CameraMotion,
    world: &mut World,
) -> Option<CameraMotion> {
    let camera = world.query_mut::<Camera3D>().next()?;
    let (position, yaw, pitch) = advance_pose(camera.position, camera.yaw, camera.pitch, &motion);
    camera.set_pose(position, yaw, pitch);
    reset_controller_velocity(world);
    motion.advanced()
}

// The pose after one camera-move step from the given pose. `forward` / `right`
// follow the free-fly look basis and `up` is world up. Pitch is clamped to the
// controller's near-vertical limit so a sustained pitch delta cannot flip the
// camera over.
fn advance_pose(
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    motion: &CameraMotion,
) -> ([f32; 3], f32, f32) {
    let cp = pitch.cos();
    let fwd = [-yaw.sin() * cp, pitch.sin(), -yaw.cos() * cp];
    let right = [yaw.cos(), 0.0, -yaw.sin()];
    let new_pos = [
        position[0] + fwd[0] * motion.forward + right[0] * motion.right,
        position[1] + fwd[1] * motion.forward + motion.up,
        position[2] + fwd[2] * motion.forward + right[2] * motion.right,
    ];
    let new_pitch = (pitch + motion.pitch).clamp(
        -std::f32::consts::FRAC_PI_2 + 0.01,
        std::f32::consts::FRAC_PI_2 - 0.01,
    );
    (new_pos, yaw + motion.yaw, new_pitch)
}

fn apply_camera_set(pose: &CameraPose, world: &mut World) -> Result<(), String> {
    let driver = {
        let ctx = world.context();
        let camera = ctx.query_with_entity::<Camera3D>().next().map(|(e, _)| e);
        camera.and_then(|e| camera_driver(&ctx, e))
    };
    if let Some(driver) = driver {
        tracing::warn!(
            "camera-set: {driver} rewrites the camera's pose every tick, so the pose lasts one frame"
        );
    }
    let Some(camera) = world.query_mut::<Camera3D>().next() else {
        return Err("camera-set: no Camera3D in world".to_string());
    };
    camera.set_pose(pose.position, pose.yaw, pose.pitch);
    if let Some(fov) = pose.fov_y_degrees {
        camera.fov_y_degrees = fov;
    }
    reset_controller_velocity(world);
    PendingHistoryReset::raise(&mut world.context(), HistoryResetCauses::CAMERA_CUT);
    Ok(())
}

// The free-fly controller integrates a smoothed velocity onto the camera every
// step, so a leftover velocity would drift an externally written pose away.
fn reset_controller_velocity(world: &mut World) {
    for system in world.systems_mut() {
        if let Some(c) = system.downcast_mut::<Camera3DSystem>() {
            c.reset_velocity();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::verbs::testing::Engine;
    use concinnity_core::components::CameraController;

    fn camera_world() -> World {
        let mut world = World::new();
        world.add_component(Camera3D {
            fov_y_degrees: 75.0,
            near: 0.05,
            view_distance: None,
            view_matrix: [[0.0; 4]; 4],
            position: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller: Some(CameraController::default()),
        });
        world.start(concinnity_engine::ecs::SYSTEMS).unwrap();
        world
    }

    fn camera(engine: &Engine) -> &Camera3D {
        engine
            .world
            .query::<Camera3D>()
            .next()
            .expect("camera present")
    }

    #[test]
    fn camera_set_writes_the_pose_and_refreshes_the_view() {
        let mut engine = Engine::new(camera_world());
        let reply = engine.call(
            "camera-set",
            json!({ "position": [10.0, 20.0, 30.0], "yaw": 1.0, "pitch": -0.5, "fov_y_degrees": 50.0 }),
        );
        assert_eq!(reply, Ok(json!({ "set": true })));
        let cam = camera(&engine);
        assert_eq!(cam.position, [10.0, 20.0, 30.0]);
        assert_eq!((cam.yaw, cam.pitch, cam.fov_y_degrees), (1.0, -0.5, 50.0));
        assert_ne!(cam.view_matrix, [[0.0; 4]; 4]);
    }

    // A teleport is a cut; a camera-move steps the pose continuously and is
    // not.
    #[test]
    fn camera_set_raises_a_camera_cut_and_camera_move_does_not() {
        let mut engine = Engine::new(camera_world());
        let taken = |engine: &mut Engine| PendingHistoryReset::take(&mut engine.world.context());
        let reply = engine.call("camera-set", json!({ "position": [400.0, 0.0, 0.0] }));
        assert!(reply.is_ok(), "{reply:?}");
        assert_eq!(taken(&mut engine), HistoryResetCauses::CAMERA_CUT);

        let motion = CameraMotion::from_step(&CameraStep {
            forward: 1.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames: 2,
        });
        advance_camera_motion(motion, &mut engine.world);
        assert!(!taken(&mut engine).any());
    }

    #[test]
    fn camera_set_keeps_the_field_of_view_when_omitted() {
        let mut engine = Engine::new(camera_world());
        let reply = engine.call("camera-set", json!({ "position": [1.0, 1.0, 1.0] }));
        assert!(reply.is_ok(), "{reply:?}");
        assert_eq!(camera(&engine).fov_y_degrees, 75.0);
    }

    #[test]
    fn the_pose_verbs_need_a_camera() {
        let mut engine = Engine::new(World::new());
        for (verb, args) in [
            ("camera-set", json!({})),
            ("camera-move", json!({ "forward": 1.0 })),
        ] {
            let error = engine.call(verb, args);
            assert_eq!(error, Err(format!("{verb}: no Camera3D in world")));
        }
        assert!(engine.motion.is_none(), "a refused move installs nothing");
    }

    #[test]
    fn camera_move_installs_a_finite_motion() {
        let mut engine = Engine::new(camera_world());
        let reply = engine.call("camera-move", json!({ "forward": 1.5, "frames": 4 }));
        assert_eq!(reply, Ok(json!({ "frames": 4, "holding": false })));
        let motion = engine.motion.clone().expect("motion installed");
        assert_eq!((motion.forward, motion.frames_left), (1.5, Some(4)));
    }

    #[test]
    fn camera_move_defaults_to_an_indefinite_hold_until_camera_stop() {
        let mut engine = Engine::new(camera_world());
        let reply = engine.call("camera-move", json!({}));
        assert_eq!(reply, Ok(json!({ "frames": 0, "holding": true })));
        assert_eq!(engine.motion.as_ref().map(|m| m.frames_left), Some(None));

        assert_eq!(
            engine.call("camera-stop", json!({})),
            Ok(json!({ "stopped": true }))
        );
        assert!(engine.motion.is_none());
    }

    #[test]
    fn a_finite_motion_counts_down_then_stops() {
        let m = CameraMotion::from_step(&CameraStep {
            forward: 1.0,
            frames: 3,
            ..CameraStep::default()
        });
        let m = m.advanced().expect("2 frames remain");
        assert_eq!(m.frames_left, Some(2));
        let m = m.advanced().expect("1 frame remains");
        assert_eq!(m.frames_left, Some(1));
        assert!(m.advanced().is_none(), "last frame exhausts the motion");
    }

    #[test]
    fn a_hold_never_runs_out() {
        let m = CameraMotion::from_step(&CameraStep {
            yaw: 0.1,
            ..CameraStep::default()
        });
        assert_eq!(m.clone().advanced(), Some(m));
    }

    #[test]
    fn advance_pose_moves_along_look_basis() {
        // yaw = 0, pitch = 0 looks down -Z; forward moves -Z, right +X, up +Y.
        let motion = CameraMotion::from_step(&CameraStep {
            forward: 2.0,
            right: 3.0,
            up: 4.0,
            frames: 1,
            ..CameraStep::default()
        });
        let (pos, yaw, pitch) = advance_pose([0.0, 0.0, 0.0], 0.0, 0.0, &motion);
        assert!((pos[0] - 3.0).abs() < 1e-5, "right -> +X");
        assert!((pos[1] - 4.0).abs() < 1e-5, "up -> +Y");
        assert!((pos[2] + 2.0).abs() < 1e-5, "forward -> -Z");
        assert_eq!((yaw, pitch), (0.0, 0.0));
    }

    #[test]
    fn advance_pose_accumulates_yaw_and_clamps_pitch() {
        let motion = CameraMotion::from_step(&CameraStep {
            yaw: 0.5,
            // A huge pitch delta must clamp, not flip the camera over.
            pitch: 100.0,
            frames: 1,
            ..CameraStep::default()
        });
        let (_, yaw, pitch) = advance_pose([0.0, 0.0, 0.0], 1.0, 0.0, &motion);
        assert!((yaw - 1.5).abs() < 1e-6);
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        assert!(
            (pitch - limit).abs() < 1e-5,
            "pitch clamps to near-vertical"
        );
    }

    // Steps accumulate displacement across frames: sustained motion.
    #[test]
    fn advancing_a_motion_moves_the_active_camera_each_frame() {
        let mut world = camera_world();
        let motion = CameraMotion::from_step(&CameraStep {
            forward: 1.0,
            frames: 2,
            ..CameraStep::default()
        });
        let motion = advance_camera_motion(motion, &mut world).expect("one frame left");
        assert!(advance_camera_motion(motion, &mut world).is_none());

        let cam = world.query::<Camera3D>().next().expect("camera present");
        assert!((cam.position[2] + 2.0).abs() < 1e-5);
        assert_ne!(cam.view_matrix, [[0.0; 4]; 4]);
    }

    #[test]
    fn a_motion_without_a_camera_is_dropped() {
        let motion = CameraMotion::from_step(&CameraStep::default());
        assert!(advance_camera_motion(motion, &mut World::new()).is_none());
    }
}
