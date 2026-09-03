// src/debug/wire/server.rs
//
// The localhost debug listener: `DebugServer` (the `DebugHook` the run loop
// ticks), the accept / per-connection threads, and the per-frame drive of the
// runtime commands plus the owned hot-reload driver. Each connection is handed
// to `crate::mcp::AppServer`, which parses the HTTP request and answers the
// MCP message it carried. The shared snapshot lives in `super::super::state`;
// the query-command dispatcher `AppServer` runs each call against is
// `super::super::dispatch::handle_request`; spawn / crossfade command handlers
// live in `super::super::commands`.

use crate::debug_hook::DebugHook;
use crate::ecs::World;
use crate::gfx::animation::AnimationSystem;
use crate::gfx::graphics_system::GraphicsSystem;
use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use concinnity_engine::shutdown::ShutdownToken;

use crate::debug::state::{AssetEntry, CameraSnapshot, DebugState};
use crate::debug::{hot_reload, runtime_spawn};
use crate::mcp::AppServer;

// Bound each connection's reads so a client that opens a socket and stalls
// mid-request releases its thread instead of holding it for the session.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);

// How often `tick` rebuilds the asset/system snapshot (in frames). The frame
// counter still advances every tick; only the heavier lists are throttled.
const SNAPSHOT_INTERVAL: u64 = 30;

// A running debug server. Implements `DebugHook`, so the run loop owns it as
// `Box<dyn DebugHook>` and ticks it each frame.
pub(crate) struct DebugServer {
    shared: Arc<Mutex<DebugState>>,
    frame: u64,
    // The asset / shader / world.jsonl reload drive. The server owns the
    // session's one driver so the `reload-assets` command can reach its
    // pending flag; the drive itself is shared with the plain `cn editor`
    // path (see `crate::debug::hot_reload::HotReloadDriver`).
    reload: hot_reload::HotReloadDriver,
    // Active camera-move motion installed by a `camera-move` command, advanced
    // once per frame by `drive_runtime_commands` until exhausted or cleared by
    // a `camera-stop`. `None` when no motion is in progress. Main-thread only.
    camera_motion: Option<runtime_spawn::CameraMotion>,
}

impl DebugServer {
    // Bind the localhost MCP endpoint on `port` and spawn its accept thread.
    // Binds `127.0.0.1` only: the debug surface is never exposed off-box.
    pub(crate) fn start(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let shared = Arc::new(Mutex::new(DebugState::default()));

        let shared_for_thread = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("debug-server".to_string())
            .spawn(move || serve(listener, shared_for_thread))?;

        tracing::info!("debug server listening on http://127.0.0.1:{port}/mcp");
        Ok(Self {
            shared,
            frame: 0,
            reload: hot_reload::HotReloadDriver::new(),
            camera_motion: None,
        })
    }

    // Report reload results through an editor session's toast queue as well as
    // the log.
    pub(crate) fn with_notifier(mut self, notifier: crate::editor::notify::Notifier) -> Self {
        self.reload = self.reload.with_notifier(notifier);
        self
    }
}

impl DebugServer {
    // Apply the debug runtime commands once per frame: decal / emitter
    // spawn against the backend, plus the deferred ECS-side commands and the
    // per-frame camera-move advance. The asset / shader / world.jsonl reload
    // passes live on `self.reload`, driven separately by `tick`.
    fn drive_runtime_commands(&mut self, world: &mut World) {
        // ECS-side commands (camera-set / camera-move / camera-stop, plus
        // quality-set) mutate the ECS or this server's motion slot, not the
        // backend, so they cannot be applied inside the systems borrow
        // below. Collect them here and apply them once that borrow ends.
        let mut deferred_ecs_cmds: Vec<runtime_spawn::RuntimeCommand> = Vec::new();
        // The backend lives in the world's parked slot (disjoint from the
        // system list), so both are borrowed at once for the apply passes.
        let (systems, mut backend) = concinnity_engine::ecs::systems_and_render_backend(world);
        for system in systems {
            if let Some(gs) = system.downcast_mut::<GraphicsSystem>() {
                if let Some(backend) = backend.take() {
                    let apply = gs.hot_reload_apply_parts(backend);
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
                }
            } else if let Some(anim) = system.downcast_mut::<AnimationSystem>() {
                anim.apply_runtime_commands();
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
                    if world
                        .query::<crate::components::Camera3D>()
                        .next()
                        .is_some()
                    {
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
    }
}

impl DebugHook for DebugServer {
    fn tick(&mut self, world: &mut World) {
        self.frame += 1;

        // The runtime commands, then the asset / shader / world.jsonl
        // reload passes. Both run only from this hook, so a `cn run` (no
        // debug hook) never touches them.
        self.drive_runtime_commands(world);
        self.reload.drive(world);

        let mut state = match self.shared.lock() {
            Ok(s) => s,
            // A panicked client thread should never take down the engine.
            Err(poisoned) => poisoned.into_inner(),
        };
        state.frame = self.frame;

        // Streaming counts change every frame in the early load-in, so refresh
        // them every tick -- `streaming_stats` is just a few small count loops
        // over the parked StreamingState (StreamingSystem owns the pools).
        state.streaming = concinnity_engine::ecs::streaming_stats(world).unwrap_or_default();
        state.scratch = world.scratch_stats();
        // Live RAM back-off pressure, refreshed alongside the streaming counts.
        state.streaming_pressure = concinnity_engine::ecs::streaming_pressure(world).map(|p| {
            crate::debug::state::PressureSnapshot {
                rss_bytes: p.rss_bytes,
                budget_bytes: p.budget_bytes,
                under_pressure: p.under_pressure,
            }
        });

        // Process thread + memory budgets (fixed at start) plus the live RSS
        // (one cheap syscall per tick, dev-only), for the `budget` query.
        if let (Some(threads), Some(memory)) = (
            concinnity_engine::ecs::thread_budget(world),
            concinnity_engine::ecs::memory_budget(world),
        ) {
            state.budget = Some(crate::debug::state::BudgetSnapshot {
                total_cores: threads.total_cores,
                job_threads: threads.job_threads,
                total_ram_mib: memory.total_ram_bytes.map(|b| b / (1024 * 1024)),
                budget_mib: memory.budget_mib(),
                overridden: memory.overridden,
                rss_mib: concinnity_engine::app::sysmem::process_resident_bytes()
                    .map(|b| b / (1024 * 1024)),
            });
        }
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
        // The asset-reload flag lives on the reload driver's state, not on
        // `GraphicsSystem`. Refresh the `pending` Arc every tick so the
        // `reload-assets` command thread flips the current flag even after a
        // world rebuild re-armed the driver with a fresh one.
        if let Some(pending) = self.reload.pending() {
            state.asset_reload = Some(pending);
        }

        // The profiler snapshot is small (one entry per system + a handful of
        // render counters), so refresh it every tick like the streaming stats.
        let profile = world.profile();
        state.profile_systems = profile
            .system_timings()
            .iter()
            .map(|&(name, micros)| (name.to_string(), micros))
            .collect();
        state.profile_allocs = profile
            .system_allocs()
            .iter()
            .map(|&(name, allocs)| (name.to_string(), allocs))
            .collect();
        state.profile_frame_allocs = profile.frame_allocs();
        state.profile_render = profile.render;

        // Active-camera pose for `camera-get`. One component read, so refresh
        // every tick like the streaming / profiler snapshots above.
        state.camera = world
            .query::<crate::components::Camera3D>()
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
                .component_census()
                .into_iter()
                .map(|(discriminant, count)| AssetEntry {
                    discriminant,
                    count,
                })
                .collect();
            // The AssetId -> name table is the build interner snapshot; it is
            // stable once the world is built, so capture it just once.
            if state.names.is_empty() {
                state.names = std::sync::Arc::new(crate::ecs::asset_id::name_table());
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
// client never blocks another, and both threads are detached: the listener
// blocks in `accept` for the life of the process, which exits out from under
// it. Errors are logged and dropped: a debug client disconnecting is routine,
// not a fault.
fn serve(listener: TcpListener, shared: Arc<Mutex<DebugState>>) {
    let server = Arc::new(AppServer::new(shared));
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!("debug server accept error: {e}");
                continue;
            }
        };
        let server = Arc::clone(&server);
        std::thread::spawn(move || {
            if let Err(e) = handle_conn(stream, &server) {
                tracing::debug!("debug client closed: {e}");
            }
        });
    }
}

// One request, one response, then the connection closes with the stream.
fn handle_conn(stream: TcpStream, server: &AppServer) -> std::io::Result<()> {
    stream.set_read_timeout(Some(CONNECTION_TIMEOUT))?;
    stream.set_write_timeout(Some(CONNECTION_TIMEOUT))?;
    let mut input = BufReader::new(stream.try_clone()?);
    let mut output = stream;
    server.serve(&mut input, &mut output)
}
