// src/debug/dispatch.rs
//
// The query-command dispatcher. `handle_request` takes one raw JSON request and
// the shared world snapshot and returns the JSON reply string. It is socket-free
// (`&str` in, `String` out over a `DebugState`), so the whole command surface is
// unit-testable against a hand-built snapshot without a live engine or a live
// socket. Each MCP tool call reaches it through `crate::mcp::AppServer`, which
// the connection loop in `super::wire::server` feeds; the spawn / crossfade
// command handlers live in `super::commands`.

use std::sync::{Arc, Mutex};

use super::commands::{
    error_reply, handle_anim_crossfade, handle_anim_param, handle_anim_state, handle_camera_move,
    handle_camera_set, handle_camera_stop, handle_decal_add, handle_decal_remove, handle_despawn,
    handle_emitter_add, handle_emitter_remove, handle_quality_set, handle_rebind, handle_reparent,
    handle_screenshot, handle_spawn, handle_story,
};
use super::hot_reload;
use super::state::DebugState;

#[derive(serde::Deserialize)]
struct Request {
    cmd: String,
}

// Dispatch one request against the shared snapshot and return a JSON reply.
pub(crate) fn handle_request(text: &str, shared: &Arc<Mutex<DebugState>>) -> String {
    let cmd = match serde_json::from_str::<Request>(text) {
        Ok(r) => r.cmd,
        Err(e) => return error_reply(&format!("malformed request: {e}")),
    };

    let state = match shared.lock() {
        Ok(s) => s,
        Err(poisoned) => poisoned.into_inner(),
    };

    let body = match cmd.as_str() {
        "ping" => serde_json::json!({ "ok": true, "pong": true }),
        "state" => serde_json::json!({
            "ok": true,
            "frame": state.frame,
            "system_count": state.system_count,
            "component_count": state.component_count,
            "systems": state.systems,
        }),
        "assets" => serde_json::json!({
            "ok": true,
            "frame": state.frame,
            "assets": state.assets,
        }),
        "names" => serde_json::json!({
            "ok": true,
            "names": &*state.names,
        }),
        "streaming" => {
            // Each pool is null when it is not streaming, else its
            // (resident, pending, unloaded) counts plus resident bytes and the
            // active byte budget (`byte_budget` is 0 when the pool runs
            // count-only, with no byte budget).
            let pool = |s: &Option<(usize, usize, usize)>, bytes: &Option<(u64, u64)>| match s {
                Some((resident, pending, unloaded)) => {
                    let (resident_bytes, byte_budget) = bytes.unwrap_or((0, 0));
                    serde_json::json!({
                        "resident": resident,
                        "pending": pending,
                        "unloaded": unloaded,
                        "resident_bytes": resident_bytes,
                        "byte_budget": byte_budget,
                    })
                }
                None => serde_json::Value::Null,
            };
            // The chunk pool has no `unloaded` count -- an infinite world has
            // no bounded set of not-yet-loaded chunks. Its resident bytes +
            // byte budget ride alongside so the byte clamp is observable.
            let chunk_pool = |s: &Option<(usize, usize)>, bytes: &Option<(u64, u64)>| match s {
                Some((resident, pending)) => {
                    let (resident_bytes, byte_budget) = bytes.unwrap_or((0, 0));
                    serde_json::json!({
                        "resident": resident,
                        "pending": pending,
                        "resident_bytes": resident_bytes,
                        "byte_budget": byte_budget,
                    })
                }
                None => serde_json::Value::Null,
            };
            // The RAM back-off valve reading rides alongside the pool counts,
            // or null before StreamingSystem's first sample / when inert.
            let pressure = match &state.streaming_pressure {
                Some(p) => serde_json::json!({
                    "rss_bytes": p.rss_bytes,
                    "budget_bytes": p.budget_bytes,
                    "under_pressure": p.under_pressure,
                }),
                None => serde_json::Value::Null,
            };
            serde_json::json!({
                "ok": true,
                "frame": state.frame,
                "texture": pool(&state.streaming.texture, &state.streaming.texture_bytes),
                "mesh": pool(&state.streaming.mesh, &state.streaming.mesh_bytes),
                "chunk": chunk_pool(&state.streaming.chunk, &state.streaming.chunk_bytes),
                "pressure": pressure,
            })
        }
        // The allocation layer's own readout: heap counters, the per-tag
        // ledger, and the size-class histogram. Global rather than snapshot
        // state, so it is read here at reply time.
        "memory" => super::memory::report(
            state.frame,
            concinnity_core::memory::stats(),
            &concinnity_core::memory::ledger().snapshot(),
            concinnity_core::memory::size_classes().and_then(|c| c.busiest()),
            state.scratch,
        ),
        "budget" => match &state.budget {
            Some(b) => serde_json::json!({
                "ok": true,
                "frame": state.frame,
                "threads": {
                    "total_cores": b.total_cores,
                    "job_threads": b.job_threads,
                },
                "memory": {
                    "total_ram_mib": b.total_ram_mib,
                    "budget_mib": b.budget_mib,
                    "overridden": b.overridden,
                    "rss_mib": b.rss_mib,
                },
            }),
            // App::start publishes the budgets before the loop runs, so a None
            // here means `tick` has not run since startup yet.
            None => serde_json::json!({
                "ok": false,
                "error": "budgets not published yet (App::start has not run)",
            }),
        },
        "profile" => {
            let r = &state.profile_render;
            // Per-system alloc counts share the timing list's order when the
            // frame loop sampled them (dev builds); absent otherwise, so the
            // field is omitted rather than reported as a misleading zero.
            let systems: Vec<_> = state
                .profile_systems
                .iter()
                .enumerate()
                .map(|(i, (name, micros))| {
                    let mut entry = serde_json::json!({ "name": name, "micros": micros });
                    if let Some((_, allocs)) = state.profile_allocs.get(i) {
                        entry["allocs"] = serde_json::json!(allocs);
                    }
                    entry
                })
                .collect();
            // Skip empty-name slots so the JSON reflects only the passes the
            // active backend actually populated. Per-pass GPU timing lands on
            // Metal + DirectX + Vulkan; backends that don't time a given pass
            // leave its slot at ("", 0), which the name filter drops.
            let passes: Vec<_> = r
                .pass_times_us
                .iter()
                .filter(|(name, _)| !name.is_empty())
                .map(|(name, micros)| serde_json::json!({ "name": name, "micros": micros }))
                .collect();
            serde_json::json!({
                "ok": true,
                "frame": state.frame,
                "frame_allocs": state.profile_frame_allocs,
                "systems": systems,
                "render": {
                    "draw_calls": r.draw_calls,
                    "objects": r.objects,
                    "skinned_visible": r.skinned_visible,
                    "skinned_pool_free": r.skinned_pool_free,
                    "gpu_frame_us": r.gpu_frame_us,
                    "vram_bytes": r.vram_bytes,
                    "transient_pool_bytes": r.transient_pool_bytes,
                    "auto_exposure_ev": r.auto_exposure_ev,
                    "max_edr": r.max_edr,
                    "passes": passes,
                },
            })
        }
        "camera-get" => match &state.camera {
            Some(c) => serde_json::json!({
                "ok": true,
                "frame": state.frame,
                "position": c.position,
                "yaw": c.yaw,
                "pitch": c.pitch,
                "fov_y_degrees": c.fov_y_degrees,
                "near": c.near,
                "far": c.far,
            }),
            // No camera snapshot yet: either the world has no Camera3D or
            // `tick` has not run since startup.
            None => serde_json::json!({
                "ok": false,
                "error": "no Camera3D snapshot (world has no camera, or tick has not run yet)",
            }),
        },
        "shutdown" => {
            match &state.shutdown_token {
                Some(token) => {
                    token.cancel();
                    serde_json::json!({ "ok": true, "shutdown": true })
                }
                // attach_shutdown runs before the loop starts, so a None here
                // means the run loop has not been entered yet.
                None => serde_json::json!({
                    "ok": false,
                    "error": "shutdown token not attached yet",
                }),
            }
        }
        "reload-shaders" => {
            match &state.shader_reload {
                Some(flag) => {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    serde_json::json!({ "ok": true, "reload_queued": true })
                }
                // `tick` captures the flag once the backend exposes it, so a
                // `None` here means either hot-reload is off (`cn run` /
                // unsupported backend) or `tick` has not run yet.
                None => serde_json::json!({
                    "ok": false,
                    "error": "shader hot-reload not available (cn debug only)",
                }),
            }
        }
        "reload-assets" => {
            match &state.asset_reload {
                Some(flag) => {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    // AnimationSystem, the GraphicsSystem world-reload pass,
                    // and the world-loaded shader reload pass each
                    // listen on their own sibling flags. Fire all four here
                    // so a single WS command reloads every hot-reloadable
                    // surface in one shot.
                    crate::app::dev_flags::set_pending_animations();
                    hot_reload::set_pending_world();
                    hot_reload::set_pending_shader_stages();
                    serde_json::json!({ "ok": true, "reload_queued": true })
                }
                // `tick` captures the flag once `GraphicsSystem` exposes it,
                // so a `None` here means `cn run`, an all-procedural world,
                // or `tick` has not yet seen GraphicsSystem.
                None => serde_json::json!({
                    "ok": false,
                    "error": "asset hot-reload not available (cn debug only; no file-backed textures captured yet)",
                }),
            }
        }
        "decal-add" => {
            // Drop the snapshot lock before blocking on the engine reply:
            // the main thread will need to acquire it for the next tick.
            drop(state);
            return handle_decal_add(text);
        }
        "decal-remove" => {
            drop(state);
            return handle_decal_remove(text);
        }
        "emitter-add" => {
            drop(state);
            return handle_emitter_add(text);
        }
        "emitter-remove" => {
            drop(state);
            return handle_emitter_remove(text);
        }
        "anim-crossfade" => {
            // The handler needs the names table to resolve the target
            // SkinnedMesh, so capture it before dropping the snapshot lock.
            let names = std::sync::Arc::clone(&state.names);
            drop(state);
            return handle_anim_crossfade(text, &names);
        }
        "anim-param" => {
            let names = std::sync::Arc::clone(&state.names);
            drop(state);
            return handle_anim_param(text, &names);
        }
        "anim-state" => {
            let names = std::sync::Arc::clone(&state.names);
            drop(state);
            return handle_anim_state(text, &names);
        }
        "screenshot" => {
            // Drop the snapshot lock before blocking on the engine reply: the
            // render thread needs it for the next tick (which performs the
            // capture).
            drop(state);
            return handle_screenshot(text);
        }
        "camera-set" => {
            // Runtime mutation: drop the snapshot lock before blocking on the
            // engine reply, like the spawn commands above.
            drop(state);
            return handle_camera_set(text);
        }
        "quality-set" => {
            // Runtime mutation (live quality toggle): drop the snapshot lock
            // before blocking on the engine reply, like the spawn commands above.
            drop(state);
            return handle_quality_set(text);
        }
        "rebind" => {
            // Runtime mutation (live key rebind): drop the snapshot lock before
            // blocking on the engine reply, like the spawn commands above.
            drop(state);
            return handle_rebind(text);
        }
        "camera-move" => {
            // Sustained-motion mutation: same drop-then-block shape as
            // camera-set. The reply fires when the motion is accepted, not when
            // it finishes, so even a long move stays inside the WS timeout.
            drop(state);
            return handle_camera_move(text);
        }
        "camera-stop" => {
            drop(state);
            return handle_camera_stop();
        }
        "despawn" => {
            // Runtime mutation (remove an authored placement): drop the snapshot
            // lock before blocking on the engine reply, like the camera / quality
            // commands above.
            drop(state);
            return handle_despawn(text);
        }
        "reparent" => {
            // Runtime mutation (move an authored placement under a new parent):
            // drop the snapshot lock before blocking, like `despawn` above.
            drop(state);
            return handle_reparent(text);
        }
        "spawn" => {
            // Runtime mutation (instantiate a copy of an authored placement):
            // drop the snapshot lock before blocking, like `despawn` above.
            drop(state);
            return handle_spawn(text);
        }
        "story" => {
            // Runtime mutation (drive the story system): drop the snapshot
            // lock before blocking, like `despawn` above.
            drop(state);
            return handle_story(text);
        }
        // The catalog is the verb table, so an unknown verb answers with what
        // the server does accept.
        other => {
            return error_reply(&format!(
                "unknown cmd '{other}' (known: {})",
                super::catalog::verb_list()
            ));
        }
    };

    body.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::state::{AssetEntry, CameraSnapshot};
    use crate::gfx::profile::RenderStats;
    use crate::gfx::streaming_system::StreamingStats;
    use std::sync::atomic::{AtomicBool, Ordering};

    // Run one request against a hand-built snapshot and parse the reply.
    fn reply(text: &str, state: DebugState) -> serde_json::Value {
        let shared = Arc::new(Mutex::new(state));
        serde_json::from_str(&handle_request(text, &shared)).expect("reply is valid JSON")
    }

    #[test]
    fn ping_pongs() {
        let r = reply(r#"{"cmd":"ping"}"#, DebugState::default());
        assert_eq!(r["ok"], true);
        assert_eq!(r["pong"], true);
    }

    #[test]
    fn state_reports_counts_and_systems() {
        let st = DebugState {
            frame: 42,
            system_count: 3,
            component_count: 7,
            systems: vec!["GraphicsSystem".into(), "PhysicsSystem".into()],
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"state"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["frame"], 42);
        assert_eq!(r["system_count"], 3);
        assert_eq!(r["component_count"], 7);
        assert_eq!(r["systems"][0], "GraphicsSystem");
    }

    #[test]
    fn budget_reports_threads_and_memory() {
        use crate::debug::state::BudgetSnapshot;
        let st = DebugState {
            frame: 9,
            budget: Some(BudgetSnapshot {
                total_cores: 10,
                job_threads: 9,
                total_ram_mib: Some(65536),
                budget_mib: 16384,
                overridden: true,
                rss_mib: Some(512),
            }),
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"budget"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["threads"]["total_cores"], 10);
        assert_eq!(r["threads"]["job_threads"], 9);
        assert_eq!(r["memory"]["total_ram_mib"], 65536);
        assert_eq!(r["memory"]["budget_mib"], 16384);
        assert_eq!(r["memory"]["overridden"], true);
        assert_eq!(r["memory"]["rss_mib"], 512);
    }

    #[test]
    fn budget_before_start_is_not_ready() {
        let r = reply(r#"{"cmd":"budget"}"#, DebugState::default());
        assert_eq!(r["ok"], false);
    }

    #[test]
    fn assets_lists_discriminant_and_count() {
        let st = DebugState {
            frame: 1,
            assets: vec![AssetEntry {
                discriminant: 5,
                count: 12,
            }],
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"assets"}"#, st);
        assert_eq!(r["assets"][0]["discriminant"], 5);
        assert_eq!(r["assets"][0]["count"], 12);
    }

    #[test]
    fn names_returns_id_table() {
        let st = DebugState {
            names: std::sync::Arc::new(vec!["hero".into(), "floor".into()]),
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"names"}"#, st);
        assert_eq!(r["names"][1], "floor");
    }

    #[test]
    fn streaming_pools_are_null_when_absent() {
        let r = reply(r#"{"cmd":"streaming"}"#, DebugState::default());
        assert_eq!(r["ok"], true);
        assert!(r["texture"].is_null());
        assert!(r["chunk"].is_null());
    }

    #[test]
    fn profile_reports_system_timings() {
        let st = DebugState {
            profile_systems: vec![("GraphicsSystem".into(), 1234)],
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"profile"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["systems"][0]["name"], "GraphicsSystem");
        assert_eq!(r["systems"][0]["micros"], 1234);
        assert!(r["render"]["passes"].is_array());
        // An unsampled build omits the alloc fields rather than reporting
        // zeroes that read as "this frame allocated nothing".
        assert!(r["systems"][0].get("allocs").is_none());
        assert!(r["frame_allocs"].is_null());
    }

    #[test]
    fn profile_reports_alloc_counts_when_sampled() {
        let st = DebugState {
            profile_systems: vec![("SpawnSystem".into(), 10), ("GraphicsSystem".into(), 1234)],
            profile_allocs: vec![("SpawnSystem".into(), 0), ("GraphicsSystem".into(), 17)],
            profile_frame_allocs: Some(29),
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"profile"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["frame_allocs"], 29);
        assert_eq!(r["systems"][0]["allocs"], 0);
        assert_eq!(r["systems"][1]["allocs"], 17);
    }

    #[test]
    fn camera_get_reports_pose_when_present() {
        let st = DebugState {
            frame: 2,
            camera: Some(CameraSnapshot {
                position: [1.0, 2.0, 3.0],
                yaw: 0.5,
                pitch: -0.2,
                fov_y_degrees: 60.0,
                near: 0.1,
                far: 100.0,
            }),
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"camera-get"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["position"][0], 1.0);
        assert_eq!(r["fov_y_degrees"], 60.0);
    }

    #[test]
    fn camera_get_errors_when_absent() {
        let r = reply(r#"{"cmd":"camera-get"}"#, DebugState::default());
        assert_eq!(r["ok"], false);
        assert!(r["error"].is_string());
    }

    #[test]
    fn shutdown_errors_without_token() {
        let r = reply(r#"{"cmd":"shutdown"}"#, DebugState::default());
        assert_eq!(r["ok"], false);
    }

    #[test]
    fn reload_shaders_errors_without_flag() {
        let r = reply(r#"{"cmd":"reload-shaders"}"#, DebugState::default());
        assert_eq!(r["ok"], false);
    }

    #[test]
    fn unknown_cmd_is_rejected() {
        let r = reply(r#"{"cmd":"bogus"}"#, DebugState::default());
        assert_eq!(r["ok"], false);
        // The reply lists what the server does accept, so a call naming a
        // typo names the right verb without a round trip through the docs.
        let error = r["error"].as_str().unwrap();
        assert!(error.contains("unknown cmd"));
        assert!(error.contains("camera-get"), "{error}");
    }

    #[test]
    fn malformed_request_is_rejected() {
        let r = reply(r#"{"no_cmd":1}"#, DebugState::default());
        assert_eq!(r["ok"], false);
        assert!(r["error"].as_str().unwrap().contains("malformed request"));
    }

    #[test]
    fn streaming_reports_populated_pools() {
        let st = DebugState {
            frame: 9,
            streaming: StreamingStats {
                texture: Some((10, 2, 1)),
                mesh: Some((4, 0, 3)),
                chunk: Some((7, 5)),
                texture_bytes: Some((2048, 4096)),
                mesh_bytes: Some((1024, 0)),
                chunk_bytes: Some((3072, 8192)),
            },
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"streaming"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["frame"], 9);
        assert_eq!(r["texture"]["resident"], 10);
        assert_eq!(r["texture"]["pending"], 2);
        assert_eq!(r["texture"]["unloaded"], 1);
        // Resident bytes + the derived byte budget ride alongside the counts.
        assert_eq!(r["texture"]["resident_bytes"], 2048);
        assert_eq!(r["texture"]["byte_budget"], 4096);
        assert_eq!(r["mesh"]["resident"], 4);
        // A 0 byte_budget flags a count-only pool (no byte budget set).
        assert_eq!(r["mesh"]["resident_bytes"], 1024);
        assert_eq!(r["mesh"]["byte_budget"], 0);
        // The chunk pool reports resident + pending but carries no unloaded count.
        assert_eq!(r["chunk"]["resident"], 7);
        assert_eq!(r["chunk"]["pending"], 5);
        assert!(r["chunk"]["unloaded"].is_null());
        // The chunk byte clamp's resident bytes + derived budget ride alongside.
        assert_eq!(r["chunk"]["resident_bytes"], 3072);
        assert_eq!(r["chunk"]["byte_budget"], 8192);
        // No pressure sample published yet: the valve reading is null.
        assert!(r["pressure"].is_null());
    }

    #[test]
    fn streaming_reports_ram_back_off_pressure() {
        let st = DebugState {
            frame: 3,
            streaming_pressure: Some(crate::debug::state::PressureSnapshot {
                rss_bytes: 900,
                budget_bytes: 1000,
                under_pressure: true,
            }),
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"streaming"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["pressure"]["rss_bytes"], 900);
        assert_eq!(r["pressure"]["budget_bytes"], 1000);
        assert_eq!(r["pressure"]["under_pressure"], true);
    }

    #[test]
    fn profile_reports_populated_render_passes() {
        let mut render = RenderStats {
            draw_calls: 128,
            objects: 64,
            ..Default::default()
        };
        // One populated pass; the remaining empty-name slots must be filtered out.
        render.pass_times_us[0] = ("shadow", 900);
        let st = DebugState {
            profile_render: render,
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"profile"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["render"]["draw_calls"], 128);
        assert_eq!(r["render"]["objects"], 64);
        let passes = r["render"]["passes"].as_array().expect("passes array");
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0]["name"], "shadow");
        assert_eq!(passes[0]["micros"], 900);
    }

    #[test]
    fn reload_shaders_queues_and_flips_the_captured_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        let st = DebugState {
            shader_reload: Some(Arc::clone(&flag)),
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"reload-shaders"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["reload_queued"], true);
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn reload_assets_queues_the_flag_and_raises_sibling_reloads() {
        // The handler flips the primary asset flag AND raises the process-global
        // sibling reload flags; serialize on the shared lock and drain those
        // flags so the side effects do not leak into other tests.
        let _guard = crate::test_support::lock();
        hot_reload::take_pending_world();
        hot_reload::take_pending_shader_stages();
        hot_reload::take_pending_stories();
        crate::app::dev_flags::take_pending_animations();

        let flag = Arc::new(AtomicBool::new(false));
        let st = DebugState {
            asset_reload: Some(Arc::clone(&flag)),
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"reload-assets"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["reload_queued"], true);
        assert!(flag.load(Ordering::SeqCst));
        // The world, Shader-stage, and animation reload surfaces were signalled;
        // drain them so they do not leak.
        assert!(hot_reload::take_pending_world());
        assert!(hot_reload::take_pending_shader_stages());
        assert!(crate::app::dev_flags::take_pending_animations());
        // Stories reload only on their own `.md` watch, so reload-assets leaves
        // that flag clear.
        assert!(!hot_reload::take_pending_stories());
    }

    #[test]
    fn reload_assets_errors_without_flag() {
        let r = reply(r#"{"cmd":"reload-assets"}"#, DebugState::default());
        assert_eq!(r["ok"], false);
        assert!(r["error"].is_string());
    }

    #[test]
    fn shutdown_cancels_the_attached_token() {
        use concinnity_engine::shutdown::ShutdownToken;
        let token = ShutdownToken::new();
        let st = DebugState {
            shutdown_token: Some(token.clone()),
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"shutdown"}"#, st);
        assert_eq!(r["ok"], true);
        assert_eq!(r["shutdown"], true);
        assert!(
            token.is_cancelled(),
            "the run loop's token must be cancelled"
        );
    }

    // Each runtime-mutation command forwards to a handler in `super::commands`.
    // A payload that fails the handler's own parse / validation returns an
    // error reply before anything is enqueued, so these exercise the dispatch
    // forwarding arm (and the handler's rejection path) without a live engine
    // or the process-global command queue.
    #[test]
    fn runtime_mutation_commands_forward_and_reject_bad_input() {
        let cases = [
            (r#"{"cmd":"decal-add","position":"x"}"#, "decal-add"),
            (r#"{"cmd":"decal-remove"}"#, "decal-remove"),
            (r#"{"cmd":"emitter-add","position":"x"}"#, "emitter-add"),
            (r#"{"cmd":"emitter-remove"}"#, "emitter-remove"),
            (
                r#"{"cmd":"anim-crossfade","target":"ghost"}"#,
                "anim-crossfade",
            ),
            (r#"{"cmd":"anim-param"}"#, "anim-param"),
            (r#"{"cmd":"anim-state"}"#, "anim-state"),
            (r#"{"cmd":"screenshot"}"#, "screenshot"),
            (r#"{"cmd":"camera-set","yaw":"x"}"#, "camera-set"),
            (r#"{"cmd":"camera-move","frames":"x"}"#, "camera-move"),
            (r#"{"cmd":"quality-set"}"#, "quality-set"),
            (r#"{"cmd":"rebind"}"#, "rebind"),
            (r#"{"cmd":"despawn"}"#, "despawn"),
            (r#"{"cmd":"reparent"}"#, "reparent"),
            (r#"{"cmd":"spawn"}"#, "spawn"),
            (r#"{"cmd":"story","action":"twirl"}"#, "story"),
        ];
        for (payload, needle) in cases {
            let r = reply(payload, DebugState::default());
            assert_eq!(r["ok"], false, "payload {payload} should be rejected");
            assert!(
                r["error"].as_str().unwrap_or_default().contains(needle),
                "reply for {payload} should name '{needle}': {r}"
            );
        }
    }

    // `camera-stop` carries no payload to fail on, so it always forwards to the
    // engine. Drive the process-global queue in a worker while the dispatcher
    // blocks so the forwarding arm's reply lands without the engine timeout.
    #[test]
    fn camera_stop_forwards_and_reports_stopped() {
        use crate::debug::runtime_spawn::{self, RuntimeCommand};
        let _guard = crate::test_support::lock();
        let worker = std::thread::spawn(|| {
            let shared = Arc::new(Mutex::new(DebugState::default()));
            handle_request(r#"{"cmd":"camera-stop"}"#, &shared)
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            for cmd in runtime_spawn::drain() {
                match cmd {
                    RuntimeCommand::CameraStop { reply } => {
                        let _ = reply.send(Ok(()));
                    }
                    other => runtime_spawn::enqueue(other),
                }
            }
            if worker.is_finished() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "camera-stop dispatch never returned"
            );
            std::thread::yield_now();
        }
        let reply = worker.join().expect("dispatch thread panicked");
        assert!(reply.contains(r#""stopped":true"#), "got: {reply}");
    }
}
