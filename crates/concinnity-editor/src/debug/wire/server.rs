// src/debug/wire/server.rs
//
// The localhost WebSocket debug server: `DebugServer` (the `DebugHook` the run
// loop ticks), the accept / per-connection threads, and the per-frame drive of
// the asset / shader / world.jsonl hot-reload passes. The shared snapshot lives
// in `super::super::state`; the query-command dispatcher is
// `super::super::dispatch::handle_request`; spawn / crossfade command handlers
// live in `super::super::commands`.

use crate::debug_hook::DebugHook;
use crate::ecs::{SystemAsset, World};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use concinnity_core::shutdown::ShutdownToken;

use tokio_tungstenite::tungstenite::{Message, accept};

use crate::debug::dispatch::handle_request;
use crate::debug::state::{AssetEntry, CameraSnapshot, DebugState};
use crate::debug::{hot_reload, runtime_spawn};

// How often `tick` rebuilds the asset/system snapshot (in frames). The frame
// counter still advances every tick; only the heavier lists are throttled.
const SNAPSHOT_INTERVAL: u64 = 30;

// A running debug server. Implements `DebugHook`, so the run loop owns it as
// `Box<dyn DebugHook>` and ticks it each frame.
pub struct DebugServer {
    shared: Arc<Mutex<DebugState>>,
    frame: u64,
    // Asset / shader / world.jsonl reload state, built lazily on the first
    // tick that sees a `GraphicsSystem` carrying init-captured sources (i.e.
    // `cn debug` with a file-backed asset / world.jsonl). Owns the filesystem
    // watcher + in-flight decode handles. `None` otherwise: `cn run` never
    // reaches `tick`, and a world with no file-backed asset never builds it.
    hot_reload: Option<hot_reload::AssetHotReloadState>,
    // Active camera-move motion installed by a `camera-move` command, advanced
    // once per frame by `drive_hot_reload` until exhausted or cleared by a
    // `camera-stop`. `None` when no motion is in progress. Main-thread only.
    camera_motion: Option<runtime_spawn::CameraMotion>,
}

impl DebugServer {
    // Bind a localhost WebSocket server on `port` and spawn its accept thread.
    // Binds `127.0.0.1` only: the debug surface is never exposed off-box.
    pub fn start(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let shared = Arc::new(Mutex::new(DebugState::default()));

        let shared_for_thread = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("debug-server".to_string())
            .spawn(move || serve(listener, shared_for_thread))?;

        tracing::info!("debug server listening on ws://127.0.0.1:{port}");
        Ok(Self {
            shared,
            frame: 0,
            hot_reload: None,
            camera_motion: None,
        })
    }
}

impl DebugServer {
    // Run the asset / shader / world.jsonl hot-reload passes once per frame and
    // apply their ECS side-effects. `cn debug` only: the matching drive used
    // to sit at the top of `GraphicsSystem::run_step` / `AnimationSystem::step`.
    // The reload state is built lazily from the `GraphicsSystem`'s init-captured
    // sources on the first tick that finds them, then driven against the
    // backend + Prop-tracking handle (`hot_reload_apply_parts`). The passes
    // return ECS edits (skeleton-shape changes + added Props) applied here, once
    // the system borrow is released. A world with no captured sources never
    // builds `self.hot_reload` and this stays a cheap no-op.
    fn drive_hot_reload(&mut self, world: &mut World) {
        let mut effects = None;
        // ECS-side commands (camera-set / camera-move / camera-stop, plus
        // quality-set) mutate the ECS or this server's motion slot, not the
        // backend, so they cannot be applied inside the systems borrow
        // below. Collect them here and apply them once that borrow ends.
        let mut deferred_ecs_cmds: Vec<runtime_spawn::RuntimeCommand> = Vec::new();
        // The backend lives in the world's parked slot (disjoint from the
        // system list), so both are borrowed at once for the apply passes.
        let (systems, mut backend) = world.systems_and_render_backend();
        for system in systems {
            match system {
                SystemAsset::GraphicsSystem(gs) => {
                    // Lazily build the reload state from the init-captured
                    // sources (must precede the apply-parts borrow of `gs`).
                    if self.hot_reload.is_none()
                        && let Some(sources) = gs.take_hot_reload_sources()
                    {
                        self.hot_reload =
                            Some(hot_reload::AssetHotReloadState::from_sources(sources));
                    }
                    if let Some(backend) = backend.take() {
                        let mut apply = gs.hot_reload_apply_parts(backend);
                        // Runtime decal / emitter spawn: independent of the
                        // hot-reload state, available in any `cn debug` world.
                        // CameraSet is deferred; everything else hits the
                        // backend now.
                        for cmd in runtime_spawn::drain() {
                            if matches!(
                                cmd,
                                runtime_spawn::RuntimeCommand::CameraSet { .. }
                                    | runtime_spawn::RuntimeCommand::CameraMove { .. }
                                    | runtime_spawn::RuntimeCommand::CameraStop { .. }
                                    | runtime_spawn::RuntimeCommand::QualitySet { .. }
                                    | runtime_spawn::RuntimeCommand::Rebind { .. }
                                    | runtime_spawn::RuntimeCommand::Despawn { .. }
                                    | runtime_spawn::RuntimeCommand::Reparent { .. }
                                    | runtime_spawn::RuntimeCommand::Spawn { .. }
                                    | runtime_spawn::RuntimeCommand::Story { .. }
                            ) {
                                deferred_ecs_cmds.push(cmd);
                            } else {
                                runtime_spawn::dispatch_runtime_spawn(
                                    cmd,
                                    apply.world_reload.as_ref(),
                                    apply.backend,
                                );
                            }
                        }
                        // Asset / shader / world.jsonl reload passes, only when
                        // the reload state was armed at init.
                        if let Some(state) = self.hot_reload.as_mut() {
                            effects = Some(hot_reload::run_frame(state, &mut apply));
                        }
                    }
                }
                SystemAsset::AnimationSystem(anim) => {
                    crate::anim_reload::reload_clips_if_pending(anim);
                    anim.apply_runtime_commands();
                }
                _ => {}
            }
        }

        // Apply deferred ECS commands now the `systems_mut` borrow is
        // released. tick() runs before the world step, so the Camera3DSystem
        // step this frame sees the new pose; the velocity reset inside keeps
        // free-fly from drifting it. camera-move / camera-stop install or clear
        // the motion slot; the actual per-frame advance happens just below so a
        // freshly installed motion also steps this same frame. quality-set
        // sends a `SettingCommand` the GraphicsSystem reads on its next step.
        for cmd in deferred_ecs_cmds {
            match cmd {
                runtime_spawn::RuntimeCommand::CameraSet { .. } => {
                    runtime_spawn::dispatch_camera_set(cmd, world);
                }
                runtime_spawn::RuntimeCommand::QualitySet { .. } => {
                    runtime_spawn::dispatch_quality_set(cmd, world);
                }
                runtime_spawn::RuntimeCommand::Rebind { .. } => {
                    runtime_spawn::dispatch_rebind(cmd, world);
                }
                runtime_spawn::RuntimeCommand::Despawn { .. } => {
                    runtime_spawn::dispatch_despawn(cmd, world);
                }
                runtime_spawn::RuntimeCommand::Reparent { .. } => {
                    runtime_spawn::dispatch_reparent(cmd, world);
                }
                runtime_spawn::RuntimeCommand::Spawn { .. } => {
                    runtime_spawn::dispatch_spawn(cmd, world);
                }
                runtime_spawn::RuntimeCommand::Story { .. } => {
                    runtime_spawn::dispatch_story(cmd, world);
                }
                runtime_spawn::RuntimeCommand::CameraMove { args, reply } => {
                    // Accept the motion only when a camera exists, so the client
                    // gets a clean error in a camera-less world. The reply fires
                    // on acceptance, not completion.
                    if world.query::<crate::assets::Camera3D>().next().is_some() {
                        self.camera_motion = Some(runtime_spawn::CameraMotion::from_args(&args));
                        let _ = reply.send(Ok(()));
                    } else {
                        let _ = reply.send(Err("camera-move: no Camera3D in world".to_string()));
                    }
                }
                runtime_spawn::RuntimeCommand::CameraStop { reply } => {
                    self.camera_motion = None;
                    let _ = reply.send(Ok(()));
                }
                // Only ECS-side variants are routed into `deferred_ecs_cmds`.
                _ => {}
            }
        }

        // Advance an in-progress camera-move one step. Runs every frame (before
        // the world step) so the renderer sees sustained motion across temporal
        // passes. A finite motion counts itself down to None; a vanished camera
        // (world swap) drops the motion rather than spinning.
        if let Some(motion) = self.camera_motion.take()
            && runtime_spawn::apply_camera_move_step(&motion, world)
        {
            self.camera_motion = motion.advanced();
        }

        let Some(effects) = effects else {
            return;
        };

        // Splice any skeleton-shape changes into the ECS-owned `SkeletonPose`
        // components so `AnimationSystem` produces right-sized output going
        // forward.
        if !effects.skeleton_updates.is_empty() {
            let index_to_new: std::collections::HashMap<usize, crate::gfx::skinning::Skeleton> =
                effects
                    .skeleton_updates
                    .into_iter()
                    .map(|u| (u.skinned_index, u.new_skeleton))
                    .collect();
            let mut applied = 0usize;
            for pose in world.query_mut::<crate::assets::SkeletonPose>() {
                if let Some(new_skel) = index_to_new.get(&pose.skinned_index) {
                    pose.skeleton = new_skel.clone();
                    pose.joint_matrices = pose.skeleton.bind_skinning_matrices();
                    applied += 1;
                }
            }
            tracing::info!(
                "asset hot-reload: applied skeleton-shape change to {} SkeletonPose component(s)",
                applied
            );
        }

        // Hand freshly re-compiled story graphs to the story system, which
        // swaps them in while keeping the play position. tick() runs before
        // the world step, so the swap lands the same frame.
        for story in effects.story_updates {
            world
                .events_mut::<crate::assets::StoryReload>()
                .send(crate::assets::StoryReload { story });
        }
    }
}

impl DebugHook for DebugServer {
    fn tick(&mut self, world: &mut World) {
        self.frame += 1;

        // Drive the asset / shader / world.jsonl hot-reload passes. This is the
        // `cn debug`-only half of the reload machinery that used to run inside
        // `GraphicsSystem::run_step` / `AnimationSystem::step`; it lives here so
        // a `cn run` (no debug hook) never touches it.
        self.drive_hot_reload(world);

        let mut state = match self.shared.lock() {
            Ok(s) => s,
            // A panicked client thread should never take down the engine.
            Err(poisoned) => poisoned.into_inner(),
        };
        state.frame = self.frame;

        // Streaming counts change every frame in the early load-in, so refresh
        // them every tick -- `streaming_stats` is just a few small count loops
        // over the parked StreamingState (StreamingSystem owns the pools).
        state.streaming = world.streaming_stats().unwrap_or_default();
        // Opportunistically pick up the shader-reload flag the backend exposes
        // (Some only under `cn debug` on hot-reload backends); once captured,
        // the `reload-shaders` command can fire the flag. The backend sits in
        // the world's parked slot between ticks.
        if state.shader_reload.is_none()
            && let Some(flag) = world
                .resource::<crate::ecs::ActiveRenderBackend>()
                .and_then(|slot| slot.0.as_ref())
                .and_then(|backend| backend.shader_reload_flag())
        {
            state.shader_reload = Some(flag);
        }
        // The asset-reload flag lives on the debug-owned `AssetHotReloadState`
        // (built lazily by `drive_hot_reload` above), not on `GraphicsSystem`.
        // Capture its `pending` Arc so the `reload-assets` command thread can
        // flip it.
        if state.asset_reload.is_none()
            && let Some(h) = self.hot_reload.as_ref()
        {
            state.asset_reload = Some(std::sync::Arc::clone(&h.pending));
        }

        // The profiler snapshot is small (one entry per system + a handful of
        // render counters), so refresh it every tick like the streaming stats.
        let profile = world.profile();
        state.profile_systems = profile
            .system_timings()
            .iter()
            .map(|&(name, micros)| (name.to_string(), micros))
            .collect();
        state.profile_render = profile.render;

        // Active-camera pose for `camera-get`. One component read, so refresh
        // every tick like the streaming / profiler snapshots above.
        state.camera = world
            .query::<crate::assets::Camera3D>()
            .next()
            .map(|c| CameraSnapshot {
                position: c.position,
                yaw: c.yaw,
                pitch: c.pitch,
                fov_y_degrees: c.fov_y_degrees,
                near: c.near,
                far: c.far,
            });

        if self.frame % SNAPSHOT_INTERVAL == 1 {
            state.system_count = world.system_count();
            state.component_count = world.component_count();
            state.systems = world
                .systems()
                .iter()
                .map(|s| s.name().to_string())
                .collect();
            state.assets = world
                .component_tags()
                .into_iter()
                .map(|tag| AssetEntry {
                    kind: "Component".to_string(),
                    discriminant: tag,
                })
                .collect();
            // The AssetId -> name table is the build interner snapshot; it is
            // stable once the world is built, so capture it just once.
            if state.names.is_empty() {
                state.names = crate::ecs::asset_id::name_table();
            }
        }
    }

    fn attach_shutdown(&mut self, shutdown: ShutdownToken) {
        let mut state = match self.shared.lock() {
            Ok(s) => s,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.shutdown_token = Some(shutdown);
    }
}

// Accept loop. Each client is handled on its own thread so a slow or stuck
// client never blocks another. Errors are logged and dropped: a debug client
// disconnecting is routine, not a fault.
fn serve(listener: TcpListener, shared: Arc<Mutex<DebugState>>) {
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!("debug server accept error: {e}");
                continue;
            }
        };
        let shared = Arc::clone(&shared);
        std::thread::spawn(move || {
            if let Err(e) = handle_conn(stream, shared) {
                tracing::debug!("debug client closed: {e}");
            }
        });
    }
}

fn handle_conn(stream: TcpStream, shared: Arc<Mutex<DebugState>>) -> Result<(), String> {
    let mut ws = accept(stream).map_err(|e| e.to_string())?;
    loop {
        match ws.read().map_err(|e| e.to_string())? {
            Message::Text(text) => {
                let reply = handle_request(&text, &shared);
                ws.send(Message::Text(reply)).map_err(|e| e.to_string())?;
            }
            Message::Ping(payload) => {
                ws.send(Message::Pong(payload)).map_err(|e| e.to_string())?;
            }
            Message::Close(_) => return Ok(()),
            _ => {}
        }
    }
}
