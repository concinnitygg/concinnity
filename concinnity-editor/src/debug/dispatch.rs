// src/debug/dispatch.rs
//
// The query-command dispatcher. `handle_request` takes one raw JSON request and
// the shared world snapshot and returns the JSON reply string. It is socket-free
// (`&str` in, `String` out over a `DebugState`), so the whole command surface is
// unit-testable against a hand-built snapshot without a live engine or a real
// WebSocket. The connection loop that feeds it lives in `super::wire::server`;
// the spawn / crossfade command handlers live in `super::commands`.

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
pub(super) fn handle_request(text: &str, shared: &Arc<Mutex<DebugState>>) -> String {
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
            "names": state.names,
        }),
        "streaming" => {
            // Each pool is null when it is not streaming, else its
            // (resident, pending, unloaded) counts.
            let pool = |s: &Option<(usize, usize, usize)>| match s {
                Some((resident, pending, unloaded)) => serde_json::json!({
                    "resident": resident,
                    "pending": pending,
                    "unloaded": unloaded,
                }),
                None => serde_json::Value::Null,
            };
            // The chunk pool has no `unloaded` count -- an infinite world has
            // no bounded set of not-yet-loaded chunks.
            let chunk_pool = |s: &Option<(usize, usize)>| match s {
                Some((resident, pending)) => serde_json::json!({
                    "resident": resident,
                    "pending": pending,
                }),
                None => serde_json::Value::Null,
            };
            serde_json::json!({
                "ok": true,
                "frame": state.frame,
                "texture": pool(&state.streaming.texture),
                "normal_map": pool(&state.streaming.normal_map),
                "mesh": pool(&state.streaming.mesh),
                "chunk": chunk_pool(&state.streaming.chunk),
            })
        }
        "profile" => {
            let r = &state.profile_render;
            let systems: Vec<_> = state
                .profile_systems
                .iter()
                .map(|(name, micros)| serde_json::json!({ "name": name, "micros": micros }))
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
                "systems": systems,
                "render": {
                    "draw_calls": r.draw_calls,
                    "objects": r.objects,
                    "skinned_visible": r.skinned_visible,
                    "skinned_pool_free": r.skinned_pool_free,
                    "gpu_frame_us": r.gpu_frame_us,
                    "vram_bytes": r.vram_bytes,
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
                    "error": "shader hot-reload not available (cn debug only, Metal-only today)",
                }),
            }
        }
        "reload-assets" => {
            match &state.asset_reload {
                Some(flag) => {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    // AnimationSystem, the GraphicsSystem world-reload pass,
                    // and the world-loaded ShaderStage reload pass each
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
            let names = state.names.clone();
            drop(state);
            return handle_anim_crossfade(text, &names);
        }
        "anim-param" => {
            let names = state.names.clone();
            drop(state);
            return handle_anim_param(text, &names);
        }
        "anim-state" => {
            let names = state.names.clone();
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
        other => return error_reply(&format!("unknown cmd '{other}'")),
    };

    body.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::state::{AssetEntry, CameraSnapshot};

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
    fn assets_lists_kind_and_discriminant() {
        let st = DebugState {
            frame: 1,
            assets: vec![AssetEntry {
                kind: "Texture".into(),
                discriminant: 5,
            }],
            ..Default::default()
        };
        let r = reply(r#"{"cmd":"assets"}"#, st);
        assert_eq!(r["assets"][0]["kind"], "Texture");
        assert_eq!(r["assets"][0]["discriminant"], 5);
    }

    #[test]
    fn names_returns_id_table() {
        let st = DebugState {
            names: vec!["hero".into(), "floor".into()],
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
        assert!(r["error"].as_str().unwrap().contains("unknown cmd"));
    }

    #[test]
    fn malformed_request_is_rejected() {
        let r = reply(r#"{"no_cmd":1}"#, DebugState::default());
        assert_eq!(r["ok"], false);
        assert!(r["error"].as_str().unwrap().contains("malformed request"));
    }
}
