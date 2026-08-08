// src/debug/runtime_spawn.rs
//
// Runtime decal / emitter / screenshot spawn queue + dispatch (`cn debug`
// only). Both halves live here so the library never compiles them:
//
//   queue     a process-wide command queue the debug WS handlers push onto
//             (`enqueue`) and the per-frame debug drive drains (`drain`).
//   dispatch  `dispatch_runtime_spawn`, run by `DebugServer::drive_hot_reload`
//             against the live backend + the init-captured texture-name table.
//
// The WS server pushes commands off the engine thread; the drive applies them
// at frame start on the main thread. Each command carries a reply channel so
// the WS handler can hand the new stable slot index back to its client
// synchronously: the wait is bounded by one frame (~16 ms at 60 Hz). `cn run`
// has no debug hook and never reaches any of this.

use std::sync::Mutex;

use crate::gfx::graphics_system::WorldReloadState;

// A runtime decal-spawn request. `texture` is the world.jsonl name of the
// Texture asset to project; `None` (or an unresolvable name) falls back to
// the renderer's white slot 0 so the tint still stamps. Geometry is the
// same TRS triple the [`crate::assets::Decal`] component carries.
#[derive(Debug, Clone)]
pub(crate) struct DecalSpawnArgs {
    pub texture: Option<String>,
    pub position: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub size: [f32; 3],
    pub tint: [f32; 4],
}

impl Default for DecalSpawnArgs {
    fn default() -> Self {
        Self {
            texture: None,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            size: [1.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

// A runtime emitter-spawn request. Same field shape as the
// [`crate::assets::ParticleEmitter`] asset; the engine clamps + normalises
// via [`crate::gfx::particles::build_particle_records`].
#[derive(Debug, Clone)]
pub(crate) struct EmitterSpawnArgs {
    pub texture: Option<String>,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub spread_deg: f32,
    pub speed_min: f32,
    pub speed_max: f32,
    pub lifetime_min: f32,
    pub lifetime_max: f32,
    pub gravity: [f32; 3],
    pub spawn_rate: f32,
    pub max_particles: u32,
    pub size_start: f32,
    pub size_end: f32,
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
}

impl Default for EmitterSpawnArgs {
    fn default() -> Self {
        Self {
            texture: None,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 1.0, 0.0],
            spread_deg: 15.0,
            speed_min: 1.0,
            speed_max: 2.0,
            lifetime_min: 1.0,
            lifetime_max: 2.0,
            gravity: [0.0, -9.8, 0.0],
            spawn_rate: 32.0,
            max_particles: 256,
            size_start: 0.2,
            size_end: 0.05,
            color_start: [1.0, 1.0, 1.0, 1.0],
            color_end: [1.0, 1.0, 1.0, 0.0],
        }
    }
}

// A runtime camera-set request: a new pose for the active `Camera3D`. `yaw` /
// `pitch` are radians (the controller's own convention); `fov_y_degrees` is
// `None` to leave the field untouched. Applied against the live ECS, not the
// backend, so it carries no texture/slot fields.
#[derive(Debug, Clone)]
pub(crate) struct CameraSetArgs {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_degrees: Option<f32>,
}

impl Default for CameraSetArgs {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            fov_y_degrees: None,
        }
    }
}

// A runtime camera-move request: a per-frame pose delta applied to the active
// `Camera3D` for a span of frames, so the renderer sees sustained motion (TAA
// ghosting, SSGI temporal noise, motion blur) that a one-shot `camera-set`
// teleport never produces. `forward` / `right` / `up` are per-frame position
// offsets (world units) along the free-fly look basis; `yaw` / `pitch` are
// per-frame radian deltas. `frames == 0` holds the motion indefinitely until a
// `camera-stop` command clears it; `frames > 0` applies it for exactly that
// many frames then auto-stops.
#[derive(Debug, Clone)]
pub(crate) struct CameraMoveArgs {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub frames: u32,
}

impl Default for CameraMoveArgs {
    fn default() -> Self {
        Self {
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames: 0,
        }
    }
}

// An in-progress camera-move the per-frame debug drive applies to the active
// `Camera3D` each tick. Built from [`CameraMoveArgs`] when a `camera-move`
// command is drained; held on the `DebugServer` (main-thread only) and
// advanced once per frame until exhausted or a `camera-stop` clears it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CameraMotion {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub yaw: f32,
    pub pitch: f32,
    // Frames still to apply. `None` is an indefinite hold (cleared only by
    // `camera-stop`); `Some(n)` counts down to the auto-stop.
    pub frames_left: Option<u32>,
}

impl CameraMotion {
    // Build a motion from a drained `camera-move` request. `frames == 0` maps
    // to an indefinite hold.
    pub fn from_args(args: &CameraMoveArgs) -> Self {
        Self {
            forward: args.forward,
            right: args.right,
            up: args.up,
            yaw: args.yaw,
            pitch: args.pitch,
            frames_left: if args.frames == 0 {
                None
            } else {
                Some(args.frames)
            },
        }
    }

    // The motion to apply next frame, after one step has just been applied: a
    // finite countdown decremented by one (returning `None` once exhausted),
    // an indefinite hold returning itself unchanged.
    pub fn advanced(self) -> Option<Self> {
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

// Compute the pose after applying one camera-move step to a free-fly camera at
// the given pose. `forward` / `right` follow the look-direction basis (matching
// `Camera3DSystem`'s free-fly mode); `up` is world up. Pitch is clamped to the
// same near-vertical limit the controller uses so a sustained pitch delta can
// not flip the camera over.
pub(crate) fn advance_pose(
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
    let new_yaw = yaw + motion.yaw;
    let new_pitch = (pitch + motion.pitch).clamp(
        -std::f32::consts::FRAC_PI_2 + 0.01,
        std::f32::consts::FRAC_PI_2 - 0.01,
    );
    (new_pos, new_yaw, new_pitch)
}

// One runtime spawn / despawn command pushed onto [`enqueue`] by the debug
// WS server and drained by the per-frame debug drive. Each variant carries a
// `std::sync::mpsc::SyncSender` reply channel so the WS handler can block
// (with timeout) on the result and hand a JSON reply back to its client.
pub(crate) enum RuntimeCommand {
    DecalAdd {
        args: DecalSpawnArgs,
        reply: std::sync::mpsc::SyncSender<Result<usize, String>>,
    },
    DecalRemove {
        id: usize,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    EmitterAdd {
        args: EmitterSpawnArgs,
        reply: std::sync::mpsc::SyncSender<Result<usize, String>>,
    },
    EmitterRemove {
        id: usize,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Capture the last presented frame to a PNG at `path`; the reply carries the
    // saved path. Routed to `RenderBackend::screenshot` on the render thread.
    Screenshot {
        path: String,
        reply: std::sync::mpsc::SyncSender<Result<String, String>>,
    },
    // Teleport the active `Camera3D` to a new pose. Applied against the ECS by
    // `apply_camera_set`, not the backend, so the per-frame drive routes it to
    // `dispatch_camera_set` (which holds the `World`) rather than
    // `dispatch_runtime_spawn`.
    CameraSet {
        args: CameraSetArgs,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Install a sustained camera-move motion on the active `Camera3D`. Like
    // `CameraSet` it mutates the ECS, so the per-frame drive partitions it out
    // and installs it on the `DebugServer` rather than touching the backend.
    // The reply fires as soon as the motion is accepted (a Camera3D exists),
    // not when it finishes, so a long move never outlasts the WS timeout.
    CameraMove {
        args: CameraMoveArgs,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Clear any in-progress camera-move motion. Also ECS-side (it clears the
    // `DebugServer`'s motion slot), so the per-frame drive routes it like
    // `CameraSet` / `CameraMove`.
    CameraStop {
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Toggle a Quality-group graphics setting (taa / ssao / ssr / ssgi /
    // auto_exposure) live by pushing the same `SettingCommand` the settings
    // menu emits. Like `CameraSet` it mutates the ECS (not the backend
    // directly), so the per-frame drive partitions it out and applies it once
    // the `systems_mut` borrow ends, via `dispatch_quality_set`.
    QualitySet {
        setting: String,
        op: crate::assets::SettingOp,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Bind a movement action (`key_forward` / ... ) to a key, live, by pushing
    // the same `Rebind` `SettingCommand` the settings menu emits. Like
    // `QualitySet` it mutates the ECS, so the per-frame drive routes it to
    // `dispatch_rebind` once the `systems_mut` borrow ends.
    Rebind {
        setting: String,
        key: crate::assets::Key,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Despawn an authored placement (and its descendants) by name. ECS-side: it
    // sends a `DespawnRequest` event the GraphicsSystem drains on its next step
    // (resolving the name to its entity, hiding the draw slots, removing the
    // entity), so the per-frame drive routes it to `dispatch_despawn` once the
    // `systems_mut` borrow ends, like `CameraSet` / `QualitySet`.
    Despawn {
        name: String,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Re-parent an authored placement by name (parent `None` detaches it to a
    // root). ECS-side like `Despawn`: it sends a `ReparentRequest` event the
    // GraphicsSystem drains on its next step, so the per-frame drive routes it to
    // `dispatch_reparent` once the `systems_mut` borrow ends.
    Reparent {
        child: String,
        parent: Option<String>,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Spawn a runtime copy of an authored placement `template` at a new
    // transform, registered under `name`, optionally with a `lifetime` after
    // which it auto-despawns. ECS-side like `Despawn`: it sends a `SpawnRequest`
    // event the GraphicsSystem drains on its next step (cloning the template's
    // draw slots into recycled slots and building the new entity), so the
    // per-frame drive routes it to `dispatch_spawn` once the `systems_mut`
    // borrow ends.
    Spawn {
        template: String,
        name: String,
        position: [f32; 3],
        rotation_deg: [f32; 3],
        scale: [f32; 3],
        lifetime: Option<f32>,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Drive the story system: the same `StoryCommand` event a stage click or
    // key press fires, so a headless harness can start, advance, and choose
    // through a story and screenshot each page. ECS-side like `Despawn`.
    Story {
        command: crate::assets::StoryCommand,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
}

static QUEUE: Mutex<Vec<RuntimeCommand>> = Mutex::new(Vec::new());

// Push a command onto the runtime-spawn queue. Returns immediately; the
// caller blocks on its own reply receiver to get the result. A poisoned
// mutex is recovered and used regardless (an unrelated panic in another
// thread must not silently drop spawn commands).
pub(crate) fn enqueue(cmd: RuntimeCommand) {
    let mut q = match QUEUE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    q.push(cmd);
}

// Take every queued command. Called by the `cn debug` drive
// (`DebugHook::tick`) at frame start. The returned `Vec` is the live list:
// the queue is reset to empty.
pub(crate) fn drain() -> Vec<RuntimeCommand> {
    let mut q = match QUEUE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    std::mem::take(&mut *q)
}

// Process one runtime-spawn command (drained from the debug WS queue)
// against the live backend. Resolves texture-name strings via the init-time
// interner snapshot + `world_reload.texture_name_to_slot` before building
// the backend record, and sends the result back via the command's reply
// channel. Reply-channel send failures are silently dropped: the WS thread
// may have already given up waiting (e.g. its client disconnected), and
// that is not a renderer error.
pub(crate) fn dispatch_runtime_spawn(
    cmd: RuntimeCommand,
    world_reload: Option<&WorldReloadState>,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) {
    match cmd {
        RuntimeCommand::DecalAdd { args, reply } => {
            let result = resolve_texture_slot(args.texture.as_deref(), world_reload)
                .and_then(|slot| {
                    let model = crate::gfx::decal::decal_model_matrix(
                        args.position,
                        args.rotation_deg,
                        args.size,
                    );
                    let inv_model = crate::gfx::decal::invert_decal_model(model)
                        .ok_or_else(|| "decal-add: degenerate size".to_string())?;
                    Ok(crate::gfx::decal::DecalRecord {
                        model,
                        inv_model,
                        texture_slot: slot,
                        tint: args.tint,
                    })
                })
                .and_then(|rec| backend.add_decal(rec));
            let _ = reply.send(result);
        }
        RuntimeCommand::DecalRemove { id, reply } => {
            let _ = reply.send(backend.remove_decal(id));
        }
        RuntimeCommand::EmitterAdd { args, reply } => {
            let result = resolve_texture_slot(args.texture.as_deref(), world_reload)
                .map(|slot| {
                    // Mirror the clamp / normalise rules used by
                    // `build_particle_records` so a WS-spawned emitter and
                    // an authored one behave identically. We do not call
                    // that helper directly because it takes a
                    // `&[&ParticleEmitter]` and a texture-id-keyed map;
                    // here we already have a resolved slot.
                    let dir = {
                        let d = args.direction;
                        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        if !len.is_finite() || len < 1e-6 {
                            [0.0, 1.0, 0.0]
                        } else {
                            [d[0] / len, d[1] / len, d[2] / len]
                        }
                    };
                    let spread_cos = args.spread_deg.clamp(0.0, 180.0).to_radians().cos();
                    let lifetime_min = args.lifetime_min.max(0.001);
                    let lifetime_max = args.lifetime_max.max(lifetime_min);
                    let speed_min = args.speed_min.max(0.0);
                    let speed_max = args.speed_max.max(speed_min);
                    let max_particles = args
                        .max_particles
                        .clamp(1, crate::gfx::particles::MAX_PARTICLES_PER_EMITTER);
                    crate::gfx::particles::ParticleEmitterRecord {
                        texture_slot: slot,
                        position: args.position,
                        direction: dir,
                        spread_cos,
                        speed_min,
                        speed_max,
                        lifetime_min,
                        lifetime_max,
                        gravity: args.gravity,
                        spawn_rate: args.spawn_rate.max(0.0),
                        max_particles,
                        size_start: args.size_start.max(0.0),
                        size_end: args.size_end.max(0.0),
                        color_start: args.color_start,
                        color_end: args.color_end,
                    }
                })
                .and_then(|rec| backend.add_emitter(rec));
            let _ = reply.send(result);
        }
        RuntimeCommand::EmitterRemove { id, reply } => {
            let _ = reply.send(backend.remove_emitter(id));
        }
        RuntimeCommand::Screenshot { path, reply } => {
            let _ = reply.send(backend.screenshot(&path));
        }
        RuntimeCommand::CameraSet { reply, .. } => {
            // CameraSet mutates the ECS, not the backend; the per-frame drive
            // partitions it out and routes it to `dispatch_camera_set`. Reaching
            // here means a future caller misrouted it.
            let _ = reply.send(Err("camera-set: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::CameraMove { reply, .. } => {
            // ECS-side like CameraSet; the per-frame drive installs it on the
            // DebugServer. Reaching the backend dispatch means a misroute.
            let _ = reply.send(Err("camera-move: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::CameraStop { reply } => {
            let _ = reply.send(Err("camera-stop: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::QualitySet { reply, .. } => {
            // ECS-side like CameraSet; the per-frame drive routes it to
            // `dispatch_quality_set`. Reaching the backend dispatch is a misroute.
            let _ = reply.send(Err("quality-set: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::Rebind { reply, .. } => {
            // ECS-side like QualitySet; routed to `dispatch_rebind`.
            let _ = reply.send(Err("rebind: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::Despawn { reply, .. } => {
            // ECS-side like CameraSet; routed to `dispatch_despawn`.
            let _ = reply.send(Err("despawn: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::Reparent { reply, .. } => {
            // ECS-side like CameraSet; routed to `dispatch_reparent`.
            let _ = reply.send(Err("reparent: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::Spawn { reply, .. } => {
            // ECS-side like CameraSet; routed to `dispatch_spawn`.
            let _ = reply.send(Err("spawn: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::Story { reply, .. } => {
            // ECS-side like CameraSet; routed to `dispatch_story`.
            let _ = reply.send(Err("story: misrouted to backend dispatch".to_string()));
        }
    }
}

// Apply a drained `CameraSet` command against the live ECS and reply. Routed
// here (instead of `dispatch_runtime_spawn`) by the per-frame debug drive
// because it needs the `World`, not the backend. Non-`CameraSet` variants are
// ignored: the caller only routes `CameraSet` here.
pub(crate) fn dispatch_camera_set(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::CameraSet { args, reply } = cmd else {
        return;
    };
    let _ = reply.send(apply_camera_set(&args, world));
}

// Apply a drained `QualitySet` command by sending a `SettingCommand` into the
// ECS, exactly as `UiInputSystem` does for a settings-menu toggle. The
// `GraphicsSystem` reads it on its next step and applies the change live
// (`apply_quality_settings`), so this exercises the real toggle path rather
// than a duplicate. Routed here (like `CameraSet`) because it mutates the ECS,
// not the backend. `cn debug` only.
pub(crate) fn dispatch_quality_set(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::QualitySet { setting, op, reply } = cmd else {
        return;
    };
    world
        .events_mut::<crate::assets::SettingCommand>()
        .send(crate::assets::SettingCommand {
            setting,
            op,
            value_label: None,
            persist: true,
        });
    let _ = reply.send(Ok(()));
}

// Apply a drained `Rebind` command by sending a `Rebind` `SettingCommand` into
// the ECS, exactly as `UiInputSystem` does after a capture. `GraphicsSystem`
// reads it on its next step and applies the rebind live (swap + `set_keymap` +
// persist + label refresh via its registry, which is why `value_label` is left
// `None` here). Routed here (like `QualitySet`) because it mutates the ECS.
pub(crate) fn dispatch_rebind(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::Rebind {
        setting,
        key,
        reply,
    } = cmd
    else {
        return;
    };
    world
        .events_mut::<crate::assets::SettingCommand>()
        .send(crate::assets::SettingCommand {
            setting,
            op: crate::assets::SettingOp::Rebind(key),
            value_label: None,
            persist: true,
        });
    let _ = reply.send(Ok(()));
}

// Apply a drained `Despawn` command by resolving the placement name to its
// AssetId and sending a `DespawnRequest` event into the ECS. GraphicsSystem
// reads it on its next step, resolves the name to its entity, hides the entity's
// draw slots, and despawns it and its descendants. Routed here (like
// `CameraSet` / `QualitySet`) because it mutates the ECS, not the backend. The
// reply fires once the event is queued; an unknown name is a clean error. The
// the despawn is applied by the GraphicsSystem on its next step.
pub(crate) fn dispatch_despawn(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::Despawn { name, reply } = cmd else {
        return;
    };
    let Some(id) = crate::ecs::asset_id::lookup(&name) else {
        let _ = reply.send(Err(format!("despawn: name '{name}' not found")));
        return;
    };
    world
        .events_mut::<crate::assets::DespawnRequest>()
        .send(crate::assets::DespawnRequest { target: id.into() });
    let _ = reply.send(Ok(()));
}

// Apply a drained `Reparent` command by resolving the child + parent names to
// AssetIds and sending a `ReparentRequest` event into the ECS. GraphicsSystem
// reads it on its next step, resolves the names to entities, and re-points the
// child's Parent edge. Routed here (like `Despawn`) because it mutates the ECS,
// not the backend. The reply fires once the event is queued; an unknown name is
// a clean error.
pub(crate) fn dispatch_reparent(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::Reparent {
        child,
        parent,
        reply,
    } = cmd
    else {
        return;
    };
    let resolve = crate::ecs::asset_id::lookup;
    let Some(child_id) = resolve(&child) else {
        let _ = reply.send(Err(format!("reparent: child '{child}' not found")));
        return;
    };
    let parent_id = match &parent {
        Some(p) => match resolve(p) {
            Some(id) => Some(id),
            None => {
                let _ = reply.send(Err(format!("reparent: parent '{p}' not found")));
                return;
            }
        },
        None => None,
    };
    world
        .events_mut::<crate::assets::ReparentRequest>()
        .send(crate::assets::ReparentRequest {
            child: child_id.into(),
            parent: parent_id.map(Into::into),
        });
    let _ = reply.send(Ok(()));
}

// Apply a drained `Spawn` command by resolving the template name to its
// AssetId, interning the new instance name, and sending a `SpawnRequest` event
// into the ECS. GraphicsSystem reads it on its next step, clones the template's
// draw slots into recycled slots, and builds the new entity. Routed here (like
// `Despawn`) because it mutates the ECS, not the backend. The reply fires once
// the event is queued; an unknown template is a clean error.
pub(crate) fn dispatch_spawn(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::Spawn {
        template,
        name,
        position,
        rotation_deg,
        scale,
        lifetime,
        reply,
    } = cmd
    else {
        return;
    };
    let Some(template_id) = crate::ecs::asset_id::lookup(&template) else {
        let _ = reply.send(Err(format!("spawn: template '{template}' not found")));
        return;
    };
    // A zero scale (the array default when the request omits it) would make the
    // instance invisible; treat it as unit scale.
    let scale = if scale == [0.0; 3] { [1.0; 3] } else { scale };
    let name_id = crate::ecs::asset_id::intern(&name);
    world
        .events_mut::<crate::assets::SpawnRequest>()
        .send(crate::assets::SpawnRequest {
            template: template_id,
            name: Some(name_id),
            transform: crate::assets::Transform {
                position,
                rotation_deg,
                scale,
            },
            lifetime_secs: lifetime,
        });
    let _ = reply.send(Ok(()));
}

// Apply a drained `Story` command by sending a `StoryCommand` event into the
// ECS, exactly as `UiInputSystem` does for a `story:*` action. The story
// system reads it on its next step and moves through its graph. Routed here
// (like `Despawn`) because it mutates the ECS, not the backend. The reply
// fires once the event is queued; a world without a story simply ignores it.
pub(crate) fn dispatch_story(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::Story { command, reply } = cmd else {
        return;
    };
    world
        .events_mut::<crate::assets::StoryCommand>()
        .send(command);
    let _ = reply.send(Ok(()));
}

// Write a new pose onto the active `Camera3D` and zero the controller velocity.
// The free-fly controller integrates a smoothed velocity onto the camera
// position every step, so a leftover velocity (or held key) would drift the
// teleport away on the next step; zeroing it makes the new pose hold. The
// debug tick runs before the world step, so the controller sees the new pose
// the same frame, and with velocity zeroed (and no input in an unfocused
// window) it leaves it untouched.
pub(crate) fn apply_camera_set(
    args: &CameraSetArgs,
    world: &mut crate::ecs::World,
) -> Result<(), String> {
    use crate::assets::Camera3D;
    let Some(camera) = world.query_mut::<Camera3D>().next() else {
        return Err("camera-set: no Camera3D in world".to_string());
    };
    camera.position = args.position;
    camera.yaw = args.yaw;
    camera.pitch = args.pitch;
    if let Some(fov) = args.fov_y_degrees {
        camera.fov_y_degrees = fov;
    }
    camera.view_matrix = crate::gfx::camera::view_matrix(camera.position, camera.yaw, camera.pitch);

    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::Camera3DSystem(c) = system {
            c.reset_velocity();
        }
    }
    Ok(())
}

// Apply one camera-move step against the live ECS: advance the active
// `Camera3D`'s pose by the motion deltas, refresh its view matrix, and zero the
// controller velocity (same reason as `apply_camera_set`: keep free-fly from
// fighting the externally driven pose). Returns `false` when the world has no
// `Camera3D`, so the caller drops the motion instead of spinning forever.
pub(crate) fn apply_camera_move_step(motion: &CameraMotion, world: &mut crate::ecs::World) -> bool {
    use crate::assets::Camera3D;
    let Some(camera) = world.query_mut::<Camera3D>().next() else {
        return false;
    };
    let (pos, yaw, pitch) = advance_pose(camera.position, camera.yaw, camera.pitch, motion);
    camera.position = pos;
    camera.yaw = yaw;
    camera.pitch = pitch;
    camera.view_matrix = crate::gfx::camera::view_matrix(pos, yaw, pitch);

    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::Camera3DSystem(c) = system {
            c.reset_velocity();
        }
    }
    true
}

// Resolve an optional Texture asset name to its pool slot index. `None` (no
// texture authored on the spawn request) maps to slot 0 (the renderer's
// white fallback) so the tint / colour gradient still stamps. An unknown
// name returns `Err`; the WS client gets a clear error rather than a
// silent fallback. Texture-name resolution leans on the init-time
// `world_reload.texture_name_to_slot` snapshot, so it only succeeds under
// `cn debug` worlds: that matches the current runtime-spawn use case
// (debug WS, headless tests).
fn resolve_texture_slot(
    texture: Option<&str>,
    world_reload: Option<&WorldReloadState>,
) -> Result<usize, String> {
    let Some(name) = texture else {
        return Ok(0);
    };
    let id = crate::ecs::asset_id::lookup(name)
        .ok_or_else(|| format!("texture '{}' not found in interner", name))?;
    let reload = world_reload.ok_or_else(|| {
        "texture-name resolution requires cn debug (world_reload missing)".to_string()
    })?;
    reload
        .texture_name_to_slot
        .get(&id)
        .copied()
        .ok_or_else(|| format!("texture '{}' is not in the live texture pool", name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;

    #[test]
    fn enqueue_drain_round_trip() {
        // The command queue is a process-global static; serialize against the
        // other queue tests so a concurrent drain cannot steal our commands.
        let _guard = test_support::lock();
        // Drain any leftovers from a panicked earlier test in this process.
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::DecalRemove { id: 7, reply: tx });
        let cmds = drain();
        assert_eq!(cmds.len(), 1);
        match cmds.into_iter().next().unwrap() {
            RuntimeCommand::DecalRemove { id, .. } => assert_eq!(id, 7),
            _ => panic!("wrong variant"),
        }
        // Second drain is empty.
        assert!(drain().is_empty());
    }

    #[test]
    fn despawn_enqueue_drain_round_trip() {
        let _guard = test_support::lock();
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::Despawn {
            name: "crate_a".to_string(),
            reply: tx,
        });
        let cmds = drain();
        assert_eq!(cmds.len(), 1);
        match cmds.into_iter().next().unwrap() {
            RuntimeCommand::Despawn { name, .. } => assert_eq!(name, "crate_a"),
            _ => panic!("wrong variant"),
        }
        assert!(drain().is_empty());
    }

    #[test]
    fn reparent_enqueue_drain_round_trip() {
        let _guard = test_support::lock();
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::Reparent {
            child: "box_a".to_string(),
            parent: Some("frame".to_string()),
            reply: tx,
        });
        let cmds = drain();
        assert_eq!(cmds.len(), 1);
        match cmds.into_iter().next().unwrap() {
            RuntimeCommand::Reparent { child, parent, .. } => {
                assert_eq!(child, "box_a");
                assert_eq!(parent.as_deref(), Some("frame"));
            }
            _ => panic!("wrong variant"),
        }
        assert!(drain().is_empty());
    }

    #[test]
    fn decal_spawn_args_defaults() {
        let a = DecalSpawnArgs::default();
        assert_eq!(a.size, [1.0, 1.0, 1.0]);
        assert_eq!(a.tint, [1.0, 1.0, 1.0, 1.0]);
        assert!(a.texture.is_none());
    }

    #[test]
    fn emitter_spawn_args_defaults() {
        let a = EmitterSpawnArgs::default();
        assert_eq!(a.direction, [0.0, 1.0, 0.0]);
        assert_eq!(a.max_particles, 256);
        assert!((a.spread_deg - 15.0).abs() < 1e-6);
    }

    #[test]
    fn camera_set_args_defaults() {
        let a = CameraSetArgs::default();
        assert_eq!(a.position, [0.0, 0.0, 0.0]);
        assert_eq!(a.yaw, 0.0);
        assert_eq!(a.pitch, 0.0);
        assert!(a.fov_y_degrees.is_none());
    }

    #[test]
    fn camera_set_enqueue_drain_round_trip() {
        let _guard = test_support::lock();
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::CameraSet {
            args: CameraSetArgs {
                position: [1.0, 2.0, 3.0],
                yaw: 0.5,
                pitch: -0.25,
                fov_y_degrees: Some(60.0),
            },
            reply: tx,
        });
        let cmds = drain();
        assert_eq!(cmds.len(), 1);
        match cmds.into_iter().next().unwrap() {
            RuntimeCommand::CameraSet { args, .. } => {
                assert_eq!(args.position, [1.0, 2.0, 3.0]);
                assert_eq!(args.fov_y_degrees, Some(60.0));
            }
            _ => panic!("wrong variant"),
        }
        assert!(drain().is_empty());
    }

    // A world with a controlled Camera3D builds a Camera3DSystem at `start`;
    // `apply_camera_set` must write the pose, refresh the view matrix, and
    // succeed (the velocity reset runs over the constructed system).
    #[test]
    fn apply_camera_set_writes_active_camera() {
        use crate::assets::{Camera3D, CameraController};
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(Camera3D {
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
            controller: Some(CameraController::default()),
        });
        world.start().unwrap();

        let args = CameraSetArgs {
            position: [10.0, 20.0, 30.0],
            yaw: 1.0,
            pitch: -0.5,
            fov_y_degrees: Some(50.0),
        };
        assert!(apply_camera_set(&args, &mut world).is_ok());

        let cam = world.query::<Camera3D>().next().expect("camera present");
        assert_eq!(cam.position, [10.0, 20.0, 30.0]);
        assert_eq!(cam.yaw, 1.0);
        assert_eq!(cam.pitch, -0.5);
        assert_eq!(cam.fov_y_degrees, 50.0);
        // view_matrix was refreshed from the new pose, no longer all-zero.
        assert_ne!(cam.view_matrix, [[0.0; 4]; 4]);
    }

    // `fov_y_degrees: None` leaves the existing field untouched.
    #[test]
    fn apply_camera_set_keeps_fov_when_none() {
        use crate::assets::{Camera3D, CameraController};
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(Camera3D {
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
            controller: Some(CameraController::default()),
        });
        world.start().unwrap();

        let args = CameraSetArgs {
            position: [1.0, 1.0, 1.0],
            yaw: 0.0,
            pitch: 0.0,
            fov_y_degrees: None,
        };
        assert!(apply_camera_set(&args, &mut world).is_ok());
        let cam = world.query::<Camera3D>().next().expect("camera present");
        assert_eq!(cam.fov_y_degrees, 75.0);
    }

    // No Camera3D in the world is a clean error, not a panic.
    #[test]
    fn apply_camera_set_errors_without_camera() {
        let mut world = crate::ecs::World::new_empty();
        let args = CameraSetArgs::default();
        assert!(apply_camera_set(&args, &mut world).is_err());
    }

    #[test]
    fn camera_move_args_defaults_are_zero_hold() {
        let a = CameraMoveArgs::default();
        assert_eq!(a.forward, 0.0);
        assert_eq!(a.right, 0.0);
        assert_eq!(a.up, 0.0);
        assert_eq!(a.yaw, 0.0);
        assert_eq!(a.pitch, 0.0);
        // frames == 0 is the indefinite-hold sentinel.
        assert_eq!(a.frames, 0);
    }

    #[test]
    fn camera_motion_from_args_maps_zero_frames_to_indefinite_hold() {
        let hold = CameraMotion::from_args(&CameraMoveArgs {
            forward: 1.0,
            frames: 0,
            ..CameraMoveArgs::default()
        });
        assert_eq!(hold.frames_left, None);
        let finite = CameraMotion::from_args(&CameraMoveArgs {
            forward: 1.0,
            frames: 5,
            ..CameraMoveArgs::default()
        });
        assert_eq!(finite.frames_left, Some(5));
    }

    #[test]
    fn camera_motion_advanced_counts_down_then_stops() {
        let m = CameraMotion {
            forward: 1.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames_left: Some(3),
        };
        // Three frames remain: countdown 3 -> 2 -> 1 -> exhausted.
        let m = m.advanced().expect("2 frames remain");
        assert_eq!(m.frames_left, Some(2));
        let m = m.advanced().expect("1 frame remains");
        assert_eq!(m.frames_left, Some(1));
        assert!(m.advanced().is_none(), "last frame exhausts the motion");
    }

    #[test]
    fn camera_motion_advanced_holds_indefinitely() {
        let m = CameraMotion {
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.1,
            pitch: 0.0,
            frames_left: None,
        };
        // An indefinite hold returns itself unchanged forever.
        let next = m.clone().advanced().expect("hold never exhausts");
        assert_eq!(next, m);
    }

    #[test]
    fn advance_pose_moves_along_look_basis() {
        // yaw = 0, pitch = 0 looks down -Z; forward delta moves -Z, right moves
        // +X, up moves +Y.
        let motion = CameraMotion {
            forward: 2.0,
            right: 3.0,
            up: 4.0,
            yaw: 0.0,
            pitch: 0.0,
            frames_left: Some(1),
        };
        let (pos, yaw, pitch) = advance_pose([0.0, 0.0, 0.0], 0.0, 0.0, &motion);
        assert!((pos[0] - 3.0).abs() < 1e-5, "right -> +X");
        assert!((pos[1] - 4.0).abs() < 1e-5, "up -> +Y");
        assert!((pos[2] + 2.0).abs() < 1e-5, "forward -> -Z");
        assert_eq!(yaw, 0.0);
        assert_eq!(pitch, 0.0);
    }

    #[test]
    fn advance_pose_accumulates_yaw_and_clamps_pitch() {
        let motion = CameraMotion {
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.5,
            // A huge pitch delta must clamp, not flip the camera over.
            pitch: 100.0,
            frames_left: Some(1),
        };
        let (_, yaw, pitch) = advance_pose([0.0, 0.0, 0.0], 1.0, 0.0, &motion);
        assert!((yaw - 1.5).abs() < 1e-6);
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        assert!(
            (pitch - limit).abs() < 1e-5,
            "pitch clamps to near-vertical"
        );
    }

    #[test]
    fn camera_move_enqueue_drain_round_trip() {
        let _guard = test_support::lock();
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::CameraMove {
            args: CameraMoveArgs {
                forward: 1.5,
                frames: 30,
                ..CameraMoveArgs::default()
            },
            reply: tx,
        });
        let (stx, _srx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::CameraStop { reply: stx });
        let cmds = drain();
        assert_eq!(cmds.len(), 2);
        let mut it = cmds.into_iter();
        match it.next().unwrap() {
            RuntimeCommand::CameraMove { args, .. } => {
                assert_eq!(args.forward, 1.5);
                assert_eq!(args.frames, 30);
            }
            _ => panic!("wrong variant"),
        }
        assert!(matches!(
            it.next().unwrap(),
            RuntimeCommand::CameraStop { .. }
        ));
        assert!(drain().is_empty());
    }

    // apply_camera_move_step advances the active camera and refreshes its view
    // matrix; a sequence of steps accumulates displacement (sustained motion).
    #[test]
    fn apply_camera_move_step_advances_active_camera() {
        use crate::assets::{Camera3D, CameraController};
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(Camera3D {
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
            controller: Some(CameraController::default()),
        });
        world.start().unwrap();

        let motion = CameraMotion {
            forward: 1.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames_left: Some(2),
        };
        assert!(apply_camera_move_step(&motion, &mut world));
        assert!(apply_camera_move_step(&motion, &mut world));

        let cam = world.query::<Camera3D>().next().expect("camera present");
        // Two forward steps of 1.0 along -Z accumulate to -2.0.
        assert!((cam.position[2] + 2.0).abs() < 1e-5);
        assert_ne!(cam.view_matrix, [[0.0; 4]; 4]);
    }

    // No Camera3D: the step is a clean `false`, not a panic, so the drive drops
    // the motion.
    #[test]
    fn apply_camera_move_step_false_without_camera() {
        let mut world = crate::ecs::World::new_empty();
        let motion = CameraMotion {
            forward: 1.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames_left: None,
        };
        assert!(!apply_camera_move_step(&motion, &mut world));
    }

    // A do-nothing RenderBackend for driving `dispatch_runtime_spawn` without a
    // GPU. It overrides none of the runtime decal / emitter / screenshot hooks,
    // so those fall through to the trait's default `Err` bodies -- exactly the
    // failure arms the dispatch reply surfaces to a WS client.
    struct StubBackend;

    impl crate::gfx::scene_flow::SceneControl for StubBackend {
        fn update_visibility(&mut self, _draw_idx: usize, _visible: bool) {}
        fn set_fade(&mut self, _fade: f32) {}
    }

    impl crate::gfx::backend::RenderBackend for StubBackend {
        fn window_closed(&mut self) -> bool {
            false
        }
        fn capture_cursor(&mut self) {}
        fn take_input(&mut self) -> crate::gfx::input::RenderInput {
            crate::gfx::input::RenderInput::default()
        }
        fn wait_idle(&self) {}
        fn draw_frame(
            &mut self,
            _params: crate::gfx::backend::FrameParams<'_>,
        ) -> crate::gfx::error::RenderResult<()> {
            Ok(())
        }
        fn update_view(&mut self, _matrix: [[f32; 4]; 4]) {}
        fn update_model(&mut self, _index: usize, _model: [[f32; 4]; 4]) {}
        fn retire_draw_object(&mut self, _draw_idx: usize) {}
        fn upload_skinned(
            &mut self,
            _vertices: &[crate::gfx::mesh_payload::SkinnedVertex],
            _indices: &[u16],
            _draw_objects: Vec<crate::gfx::render_types::SkinnedDrawObject>,
            _vert_bytes: &[u8],
            _frag_bytes: &[u8],
            _shadow_bytes: &[u8],
        ) -> crate::gfx::error::RenderResult<()> {
            Ok(())
        }
        fn update_skinned_pose(&mut self, _skinned_index: usize, _matrices: &[[[f32; 4]; 4]]) {}
        fn evict_texture_slot(&mut self, _slot: usize) -> Result<(), String> {
            Ok(())
        }
        fn update_texture_slot(
            &mut self,
            _slot: usize,
            _image: &concinnity_core::build::texture::TextureImage,
        ) -> crate::gfx::error::RenderResult<()> {
            Ok(())
        }
        fn evict_mesh(&mut self, _draw_idx: usize, _retire_frame: u64) -> Result<(), String> {
            Ok(())
        }
        fn upload_mesh(
            &mut self,
            _draw_idx: usize,
            _verts: &[crate::gfx::mesh_payload::Vertex],
            _idxs: &[u16],
            _frame: u64,
        ) -> crate::gfx::error::RenderResult<()> {
            Ok(())
        }
        fn setup_chunk_streaming(
            &mut self,
            _chunk_vtx_bytes: usize,
            _chunk_idx_bytes: usize,
            _texture_slot: usize,
            _normal_map_slot: usize,
        ) -> crate::gfx::error::RenderResult<()> {
            Ok(())
        }
        fn add_chunk_mesh(
            &mut self,
            _mesh: crate::gfx::backend::ChunkMesh<'_>,
        ) -> crate::gfx::error::RenderResult<usize> {
            Ok(0)
        }
        fn remove_chunk_mesh(
            &mut self,
            _draw_idx: usize,
            _retire_frame: u64,
        ) -> Result<(), String> {
            Ok(())
        }
        fn set_chunk_model(
            &mut self,
            _draw_idx: usize,
            _model: [[f32; 4]; 4],
        ) -> Result<(), String> {
            Ok(())
        }
    }

    // Drive one runtime-spawn command whose reply is `Result<(), String>` (the
    // ECS-side variants) through the backend dispatch and return its reply.
    fn dispatch_spawn_unit(
        build: impl FnOnce(std::sync::mpsc::SyncSender<Result<(), String>>) -> RuntimeCommand,
    ) -> Result<(), String> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut backend = StubBackend;
        dispatch_runtime_spawn(build(tx), None, &mut backend);
        rx.recv().expect("dispatch replied")
    }

    // Every ECS-side command that reaches the backend dispatch is a misroute:
    // the per-frame drive should have partitioned it to its ECS dispatcher. Each
    // arm replies with a clear error that names itself.
    #[test]
    fn backend_dispatch_reports_misroute_for_ecs_commands() {
        let cases: [(&str, Result<(), String>); 9] = [
            (
                "camera-set",
                dispatch_spawn_unit(|reply| RuntimeCommand::CameraSet {
                    args: CameraSetArgs::default(),
                    reply,
                }),
            ),
            (
                "camera-move",
                dispatch_spawn_unit(|reply| RuntimeCommand::CameraMove {
                    args: CameraMoveArgs::default(),
                    reply,
                }),
            ),
            (
                "camera-stop",
                dispatch_spawn_unit(|reply| RuntimeCommand::CameraStop { reply }),
            ),
            (
                "quality-set",
                dispatch_spawn_unit(|reply| RuntimeCommand::QualitySet {
                    setting: "ssao".to_string(),
                    op: crate::assets::SettingOp::Next,
                    reply,
                }),
            ),
            (
                "rebind",
                dispatch_spawn_unit(|reply| RuntimeCommand::Rebind {
                    setting: "key_forward".to_string(),
                    key: crate::assets::Key::Space,
                    reply,
                }),
            ),
            (
                "despawn",
                dispatch_spawn_unit(|reply| RuntimeCommand::Despawn {
                    name: "x".to_string(),
                    reply,
                }),
            ),
            (
                "reparent",
                dispatch_spawn_unit(|reply| RuntimeCommand::Reparent {
                    child: "x".to_string(),
                    parent: None,
                    reply,
                }),
            ),
            (
                "spawn",
                dispatch_spawn_unit(|reply| RuntimeCommand::Spawn {
                    template: "x".to_string(),
                    name: "y".to_string(),
                    position: [0.0; 3],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                    lifetime: None,
                    reply,
                }),
            ),
            (
                "story",
                dispatch_spawn_unit(|reply| RuntimeCommand::Story {
                    command: crate::assets::StoryCommand::Advance,
                    reply,
                }),
            ),
        ];
        for (label, result) in cases {
            let err = result.expect_err("misroute should be an error");
            assert!(err.contains(label), "expected '{label}' in: {err}");
            assert!(err.contains("misrouted to backend dispatch"), "got: {err}");
        }
    }

    // DecalAdd with no texture resolves to the white slot and, with a
    // non-degenerate size, builds the DecalRecord before handing it to the
    // backend, whose default `add_decal` reports the feature is unimplemented.
    #[test]
    fn backend_dispatch_decal_add_builds_record_then_surfaces_backend_err() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut backend = StubBackend;
        dispatch_runtime_spawn(
            RuntimeCommand::DecalAdd {
                args: DecalSpawnArgs::default(),
                reply: tx,
            },
            None,
            &mut backend,
        );
        let err = rx.recv().unwrap().unwrap_err();
        assert!(err.contains("add_decal"), "got: {err}");
    }

    // A zero size makes the decal model matrix non-invertible, caught before the
    // backend is touched.
    #[test]
    fn backend_dispatch_decal_add_rejects_degenerate_size() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut backend = StubBackend;
        dispatch_runtime_spawn(
            RuntimeCommand::DecalAdd {
                args: DecalSpawnArgs {
                    size: [0.0, 0.0, 0.0],
                    ..DecalSpawnArgs::default()
                },
                reply: tx,
            },
            None,
            &mut backend,
        );
        let err = rx.recv().unwrap().unwrap_err();
        assert!(err.contains("degenerate size"), "got: {err}");
    }

    #[test]
    fn backend_dispatch_decal_remove_surfaces_backend_err() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut backend = StubBackend;
        dispatch_runtime_spawn(
            RuntimeCommand::DecalRemove { id: 3, reply: tx },
            None,
            &mut backend,
        );
        let err = rx.recv().unwrap().unwrap_err();
        assert!(err.contains("remove_decal"), "got: {err}");
    }

    // EmitterAdd normalizes / clamps the request into a ParticleEmitterRecord
    // before the backend's default `add_emitter` reports it unimplemented. The
    // two configs cover both the normalize branch (unit direction) and the
    // degenerate-direction fallback plus every out-of-range clamp.
    #[test]
    fn backend_dispatch_emitter_add_normalizes_then_surfaces_backend_err() {
        let configs = [
            EmitterSpawnArgs::default(),
            EmitterSpawnArgs {
                direction: [0.0, 0.0, 0.0],
                spread_deg: 500.0,
                speed_min: -5.0,
                speed_max: -10.0,
                lifetime_min: -1.0,
                lifetime_max: -2.0,
                max_particles: 0,
                spawn_rate: -1.0,
                size_start: -1.0,
                size_end: -1.0,
                ..EmitterSpawnArgs::default()
            },
        ];
        for args in configs {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let mut backend = StubBackend;
            dispatch_runtime_spawn(
                RuntimeCommand::EmitterAdd { args, reply: tx },
                None,
                &mut backend,
            );
            let err = rx.recv().unwrap().unwrap_err();
            assert!(err.contains("add_emitter"), "got: {err}");
        }
    }

    #[test]
    fn backend_dispatch_emitter_remove_surfaces_backend_err() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut backend = StubBackend;
        dispatch_runtime_spawn(
            RuntimeCommand::EmitterRemove { id: 5, reply: tx },
            None,
            &mut backend,
        );
        let err = rx.recv().unwrap().unwrap_err();
        assert!(err.contains("remove_emitter"), "got: {err}");
    }

    #[test]
    fn backend_dispatch_screenshot_surfaces_backend_err() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut backend = StubBackend;
        dispatch_runtime_spawn(
            RuntimeCommand::Screenshot {
                path: "shot.png".to_string(),
                reply: tx,
            },
            None,
            &mut backend,
        );
        let err = rx.recv().unwrap().unwrap_err();
        assert!(
            err.contains("screenshot capture not supported"),
            "got: {err}"
        );
    }

    // resolve_texture_slot: the no-texture fallback, the three failure cases,
    // and the live-pool hit. Touches the thread-local interner, so it takes the
    // shared lock and resets the interner first, like the queue tests.
    #[test]
    fn resolve_texture_slot_covers_every_case() {
        let _guard = test_support::lock();
        crate::ecs::asset_id::reset_interner();

        // None -> the renderer's white fallback slot, no interner or reload.
        assert_eq!(resolve_texture_slot(None, None).unwrap(), 0);

        // A name absent from the interner is a clear error.
        let err = resolve_texture_slot(Some("ghost"), None).unwrap_err();
        assert!(err.contains("not found in interner"), "got: {err}");

        // Interned name but no world_reload (not `cn debug`): unavailable.
        crate::ecs::asset_id::intern_all(&["grid"]);
        let err = resolve_texture_slot(Some("grid"), None).unwrap_err();
        assert!(err.contains("world_reload missing"), "got: {err}");

        // Interned name, reload present, but the name is not in the pool map.
        let empty = WorldReloadState {
            texture_name_to_slot: std::collections::HashMap::new(),
        };
        let err = resolve_texture_slot(Some("grid"), Some(&empty)).unwrap_err();
        assert!(err.contains("not in the live texture pool"), "got: {err}");

        // Interned name present in the pool map -> its resolved slot.
        let mut map = std::collections::HashMap::new();
        map.insert(crate::ecs::asset_id::AssetId(0), 5usize);
        let reload = WorldReloadState {
            texture_name_to_slot: map,
        };
        assert_eq!(
            resolve_texture_slot(Some("grid"), Some(&reload)).unwrap(),
            5
        );
    }

    // Each ECS dispatcher is only ever handed its own variant; a mismatched
    // command hits the `let else { return }` guard and drops the reply channel
    // without answering (so the receiver observes a disconnect). This never
    // reaches the name table, so no interner setup is needed.
    #[test]
    fn ecs_dispatchers_ignore_mismatched_variants() {
        fn dropped(rx: std::sync::mpsc::Receiver<Result<(), String>>) -> bool {
            rx.recv().is_err()
        }
        let mut world = crate::ecs::World::new_empty();

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_camera_set(RuntimeCommand::CameraStop { reply: tx }, &mut world);
        assert!(dropped(rx));

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_quality_set(RuntimeCommand::CameraStop { reply: tx }, &mut world);
        assert!(dropped(rx));

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_rebind(RuntimeCommand::CameraStop { reply: tx }, &mut world);
        assert!(dropped(rx));

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_despawn(RuntimeCommand::CameraStop { reply: tx }, &mut world);
        assert!(dropped(rx));

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_reparent(RuntimeCommand::CameraStop { reply: tx }, &mut world);
        assert!(dropped(rx));

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_spawn(RuntimeCommand::CameraStop { reply: tx }, &mut world);
        assert!(dropped(rx));

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_story(RuntimeCommand::CameraStop { reply: tx }, &mut world);
        assert!(dropped(rx));
    }

    fn controlled_camera() -> crate::assets::Camera3D {
        use crate::assets::{Camera3D, CameraController};
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
            controller: Some(CameraController::default()),
        }
    }

    // The CameraSet wrapper applies the pose against the live ECS and replies Ok.
    #[test]
    fn dispatch_camera_set_applies_pose_and_replies_ok() {
        use crate::assets::Camera3D;
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(controlled_camera());
        world.start().unwrap();

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_camera_set(
            RuntimeCommand::CameraSet {
                args: CameraSetArgs {
                    position: [1.0, 2.0, 3.0],
                    yaw: 0.5,
                    pitch: -0.25,
                    fov_y_degrees: Some(50.0),
                },
                reply: tx,
            },
            &mut world,
        );
        assert!(rx.recv().unwrap().is_ok());
        let cam = world.query::<Camera3D>().next().expect("camera present");
        assert_eq!(cam.position, [1.0, 2.0, 3.0]);
        assert_eq!(cam.fov_y_degrees, 50.0);
    }

    #[test]
    fn dispatch_quality_set_sends_setting_command() {
        let mut world = crate::ecs::World::new_empty();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_quality_set(
            RuntimeCommand::QualitySet {
                setting: "ssao".to_string(),
                op: crate::assets::SettingOp::Next,
                reply: tx,
            },
            &mut world,
        );
        assert!(rx.recv().unwrap().is_ok());

        let events = world
            .events::<crate::assets::SettingCommand>()
            .expect("setting command queued");
        let mut cursor = crate::ecs::EventCursor::default();
        let seen: Vec<_> = events.read(&mut cursor).collect();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].setting, "ssao");
        assert_eq!(seen[0].op, crate::assets::SettingOp::Next);
        assert!(seen[0].persist);
        assert!(seen[0].value_label.is_none());
    }

    #[test]
    fn dispatch_rebind_sends_rebind_setting_command() {
        let mut world = crate::ecs::World::new_empty();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_rebind(
            RuntimeCommand::Rebind {
                setting: "key_forward".to_string(),
                key: crate::assets::Key::Space,
                reply: tx,
            },
            &mut world,
        );
        assert!(rx.recv().unwrap().is_ok());

        let events = world
            .events::<crate::assets::SettingCommand>()
            .expect("setting command queued");
        let mut cursor = crate::ecs::EventCursor::default();
        let seen: Vec<_> = events.read(&mut cursor).collect();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].setting, "key_forward");
        assert_eq!(
            seen[0].op,
            crate::assets::SettingOp::Rebind(crate::assets::Key::Space)
        );
    }

    #[test]
    fn dispatch_story_forwards_the_command() {
        let mut world = crate::ecs::World::new_empty();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_story(
            RuntimeCommand::Story {
                command: crate::assets::StoryCommand::Choose(2),
                reply: tx,
            },
            &mut world,
        );
        assert!(rx.recv().unwrap().is_ok());

        let events = world
            .events::<crate::assets::StoryCommand>()
            .expect("story command queued");
        let mut cursor = crate::ecs::EventCursor::default();
        let seen: Vec<_> = events.read(&mut cursor).collect();
        assert_eq!(seen.len(), 1);
        assert_eq!(*seen[0], crate::assets::StoryCommand::Choose(2));
    }

    #[test]
    fn dispatch_despawn_resolves_name_and_reports_unknown() {
        let _guard = test_support::lock();
        crate::ecs::asset_id::reset_interner();
        crate::ecs::asset_id::intern_all(&["crate_a", "crate_b"]);
        let mut world = crate::ecs::World::new_empty();

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_despawn(
            RuntimeCommand::Despawn {
                name: "crate_b".to_string(),
                reply: tx,
            },
            &mut world,
        );
        assert!(rx.recv().unwrap().is_ok());
        let events = world
            .events::<crate::assets::DespawnRequest>()
            .expect("despawn request queued");
        let mut cursor = crate::ecs::EventCursor::default();
        let seen: Vec<_> = events.read(&mut cursor).collect();
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].target.name().unwrap(),
            crate::ecs::asset_id::AssetId(1)
        );

        // An unknown name is a clean error.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_despawn(
            RuntimeCommand::Despawn {
                name: "ghost".to_string(),
                reply: tx,
            },
            &mut world,
        );
        let err = rx.recv().unwrap().unwrap_err();
        assert!(err.contains("'ghost' not found"), "got: {err}");
    }

    #[test]
    fn dispatch_reparent_resolves_names_and_reports_errors() {
        let _guard = test_support::lock();
        crate::ecs::asset_id::reset_interner();
        crate::ecs::asset_id::intern_all(&["box_a", "frame"]);
        let mut world = crate::ecs::World::new_empty();
        let mut cursor = crate::ecs::EventCursor::default();

        // Both names known -> queued with both ids.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_reparent(
            RuntimeCommand::Reparent {
                child: "box_a".to_string(),
                parent: Some("frame".to_string()),
                reply: tx,
            },
            &mut world,
        );
        assert!(rx.recv().unwrap().is_ok());
        {
            let events = world
                .events::<crate::assets::ReparentRequest>()
                .expect("reparent request queued");
            let seen: Vec<_> = events.read(&mut cursor).collect();
            assert_eq!(seen.len(), 1);
            assert_eq!(
                seen[0].child.name().unwrap(),
                crate::ecs::asset_id::AssetId(0)
            );
            assert_eq!(
                seen[0].parent.and_then(|p| p.name()),
                Some(crate::ecs::asset_id::AssetId(1))
            );
        }

        // A None parent detaches the child to a root.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_reparent(
            RuntimeCommand::Reparent {
                child: "box_a".to_string(),
                parent: None,
                reply: tx,
            },
            &mut world,
        );
        assert!(rx.recv().unwrap().is_ok());
        {
            let events = world.events::<crate::assets::ReparentRequest>().unwrap();
            let seen: Vec<_> = events.read(&mut cursor).collect();
            assert_eq!(seen.len(), 1);
            assert!(seen[0].parent.is_none());
        }

        // Unknown child.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_reparent(
            RuntimeCommand::Reparent {
                child: "ghost".to_string(),
                parent: Some("frame".to_string()),
                reply: tx,
            },
            &mut world,
        );
        let err = rx.recv().unwrap().unwrap_err();
        assert!(err.contains("child 'ghost' not found"), "got: {err}");

        // Known child, unknown parent.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_reparent(
            RuntimeCommand::Reparent {
                child: "box_a".to_string(),
                parent: Some("void".to_string()),
                reply: tx,
            },
            &mut world,
        );
        let err = rx.recv().unwrap().unwrap_err();
        assert!(err.contains("parent 'void' not found"), "got: {err}");
    }

    #[test]
    fn dispatch_spawn_resolves_template_interns_name_and_defaults_scale() {
        let _guard = test_support::lock();
        crate::ecs::asset_id::reset_interner();
        crate::ecs::asset_id::intern_all(&["template_a"]);
        let mut world = crate::ecs::World::new_empty();
        let mut cursor = crate::ecs::EventCursor::default();

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_spawn(
            RuntimeCommand::Spawn {
                template: "template_a".to_string(),
                name: "instance_1".to_string(),
                position: [1.0, 2.0, 3.0],
                rotation_deg: [0.0, 90.0, 0.0],
                scale: [2.0, 2.0, 2.0],
                lifetime: Some(5.0),
                reply: tx,
            },
            &mut world,
        );
        assert!(rx.recv().unwrap().is_ok());
        {
            let events = world
                .events::<crate::assets::SpawnRequest>()
                .expect("spawn request queued");
            let seen: Vec<_> = events.read(&mut cursor).collect();
            assert_eq!(seen.len(), 1);
            assert_eq!(seen[0].template, crate::ecs::asset_id::AssetId(0));
            // The new instance name was interned to the next id.
            assert_eq!(seen[0].name, Some(crate::ecs::asset_id::AssetId(1)));
            assert_eq!(seen[0].transform.position, [1.0, 2.0, 3.0]);
            assert_eq!(seen[0].transform.scale, [2.0, 2.0, 2.0]);
            assert_eq!(seen[0].lifetime_secs, Some(5.0));
        }

        // A zero scale (the array default when omitted) is treated as unit scale
        // so the spawned copy is visible.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_spawn(
            RuntimeCommand::Spawn {
                template: "template_a".to_string(),
                name: "instance_2".to_string(),
                position: [0.0; 3],
                rotation_deg: [0.0; 3],
                scale: [0.0; 3],
                lifetime: None,
                reply: tx,
            },
            &mut world,
        );
        assert!(rx.recv().unwrap().is_ok());
        {
            let events = world.events::<crate::assets::SpawnRequest>().unwrap();
            let seen: Vec<_> = events.read(&mut cursor).collect();
            assert_eq!(seen.len(), 1);
            assert_eq!(seen[0].transform.scale, [1.0, 1.0, 1.0]);
        }

        // An unknown template is a clean error.
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_spawn(
            RuntimeCommand::Spawn {
                template: "missing".to_string(),
                name: "instance_3".to_string(),
                position: [0.0; 3],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
                lifetime: None,
                reply: tx,
            },
            &mut world,
        );
        let err = rx.recv().unwrap().unwrap_err();
        assert!(err.contains("template 'missing' not found"), "got: {err}");
    }
}
