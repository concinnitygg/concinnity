//! The verbs answered from the per-frame snapshot, or from a handle it holds,
//! without waiting on the engine.

use serde::Serialize;
use serde_json::{Value, json};
use std::sync::atomic::Ordering;

use crate::debug::call::Call;
use crate::debug::memory;
use crate::debug::verb::{Access, Args, Reply, Verb};

pub(in crate::debug) const VERBS: &[Verb] = &[
    Verb {
        name: "ping",
        description: "Check the debug server is alive; replies with a pong.",
        access: Access::ReadOnly,
        params: &[],
        run: ping,
    },
    Verb {
        name: "state",
        description: "Report the current frame, the system and component counts, and the running system names.",
        access: Access::ReadOnly,
        params: &[],
        run: state,
    },
    Verb {
        name: "assets",
        description: "Report how many instances of each component the running world holds, keyed by discriminant.",
        access: Access::ReadOnly,
        params: &[],
        run: assets,
    },
    Verb {
        name: "names",
        description: "Report the build's asset id to handle table, indexed by id: each asset's `$id`, or `<Type>#<ordinal>` for an anonymous one.",
        access: Access::ReadOnly,
        params: &[],
        run: names,
    },
    Verb {
        name: "streaming",
        description: "Report streaming residency for the texture, mesh, and chunk pools, plus the memory back-off reading.",
        access: Access::ReadOnly,
        params: &[],
        run: streaming,
    },
    Verb {
        name: "memory",
        description: "Report the allocation layer's heap counters, per-tag ledger, and busiest size class.",
        access: Access::ReadOnly,
        params: &[],
        run: memory,
    },
    Verb {
        name: "budget",
        description: "Report the resolved thread and memory budgets alongside the current resident set size.",
        access: Access::ReadOnly,
        params: &[],
        run: budget,
    },
    Verb {
        name: "profile",
        description: "Report last-frame CPU time per system plus render draw-call, object, per-pass GPU timings, and the CPU's blocked-on-GPU time.",
        access: Access::ReadOnly,
        params: &[],
        run: profile,
    },
    Verb {
        name: "camera-get",
        description: "Report the active camera's position, yaw, pitch, vertical field of view, near plane, and view distance (null when unlimited).",
        access: Access::ReadOnly,
        params: &[],
        run: camera_get,
    },
    Verb {
        name: "shutdown",
        description: "Cancel the run loop's shutdown token so the engine exits cleanly on its next iteration.",
        access: Access::Mutating,
        params: &[],
        run: shutdown,
    },
    Verb {
        name: "reload-shaders",
        description: "Queue a rebuild of every built-in render pipeline from disk-resident shader source.",
        access: Access::Mutating,
        params: &[],
        run: reload_shaders,
    },
    Verb {
        name: "reload-assets",
        description: "Queue a re-decode of file-backed textures and a reload of world, animation, and shader stage sources.",
        access: Access::Mutating,
        params: &[],
        run: reload_assets,
    },
];

// A snapshot struct's fields, after the frame it was taken on.
#[derive(Serialize)]
struct Framed<'a, T> {
    frame: u64,
    #[serde(flatten)]
    snapshot: &'a T,
}

fn framed(frame: u64, snapshot: &impl Serialize) -> Reply {
    serde_json::to_value(Framed { frame, snapshot }).map_err(|e| e.to_string())
}

fn ping(_: &Call, _: Args) -> Reply {
    Ok(json!({ "pong": true }))
}

fn state(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    Ok(json!({
        "frame": s.frame,
        "system_count": s.system_count,
        "component_count": s.component_count,
        "systems": s.systems,
    }))
}

fn assets(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    Ok(json!({ "frame": s.frame, "assets": s.assets }))
}

fn names(call: &Call, _: Args) -> Reply {
    Ok(json!({ "names": &*call.snapshot().names }))
}

fn streaming(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    let pools = &s.streaming;
    Ok(json!({
        "frame": s.frame,
        "texture": pool(pools.texture, pools.texture_bytes),
        "mesh": pool(pools.mesh, pools.mesh_bytes),
        "chunk": chunk_pool(pools.chunk, pools.chunk_bytes),
        "pressure": s.streaming_pressure,
    }))
}

// A streaming pool's (resident, pending, unloaded) counts with its resident
// bytes and byte budget (0 for a count-only pool), or null when it is not
// streaming.
fn pool(counts: Option<(usize, usize, usize)>, bytes: Option<(u64, u64)>) -> Value {
    let Some((resident, pending, unloaded)) = counts else {
        return Value::Null;
    };
    let (resident_bytes, byte_budget) = bytes.unwrap_or((0, 0));
    json!({
        "resident": resident,
        "pending": pending,
        "unloaded": unloaded,
        "resident_bytes": resident_bytes,
        "byte_budget": byte_budget,
    })
}

// The chunk pool has no `unloaded` count: an infinite world has no bounded set
// of chunks still to load.
fn chunk_pool(counts: Option<(usize, usize)>, bytes: Option<(u64, u64)>) -> Value {
    let Some((resident, pending)) = counts else {
        return Value::Null;
    };
    let (resident_bytes, byte_budget) = bytes.unwrap_or((0, 0));
    json!({
        "resident": resident,
        "pending": pending,
        "resident_bytes": resident_bytes,
        "byte_budget": byte_budget,
    })
}

// The allocation layer's counters are global rather than snapshot state, so
// they are read at reply time.
fn memory(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    Ok(memory::report(
        s.frame,
        concinnity_core::memory::stats(),
        &concinnity_core::memory::ledger().snapshot(),
        concinnity_core::memory::size_classes().and_then(|c| c.busiest()),
        s.scratch,
    ))
}

fn budget(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    let budget = s
        .budget
        .as_ref()
        .ok_or("budgets not published yet (Runtime::start has not run)")?;
    framed(s.frame, budget)
}

fn profile(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    let r = &s.profile_render;
    // Alloc counts share the timing list's order in builds that sample them,
    // and are omitted otherwise rather than reported as a misleading zero.
    let systems: Vec<_> = s
        .profile_systems
        .iter()
        .enumerate()
        .map(|(i, (name, micros))| {
            let mut entry = json!({ "name": name, "micros": micros });
            if let Some((_, allocs)) = s.profile_allocs.get(i) {
                entry["allocs"] = json!(allocs);
            }
            entry
        })
        .collect();
    // A pass slot the backend does not time keeps an empty name.
    let passes: Vec<_> = r
        .pass_times_us
        .iter()
        .filter(|(name, _)| !name.is_empty())
        .map(|(name, micros)| json!({ "name": name, "micros": micros }))
        .collect();
    Ok(json!({
        "frame": s.frame,
        "frame_allocs": s.profile_frame_allocs,
        "systems": systems,
        "render": {
            "draw_calls": r.draw_calls,
            "objects": r.objects,
            "skinned_visible": r.skinned_visible,
            "skinned_pool_free": r.skinned_pool_free,
            "gpu_frame_us": r.gpu_frame_us,
            "gpu_wait_us": r.gpu_wait_us,
            "vram_bytes": r.vram_bytes,
            "transient_pool_bytes": r.transient_pool_bytes,
            "auto_exposure_ev": r.auto_exposure_ev,
            "max_edr": r.max_edr,
            "passes": passes,
        },
    }))
}

fn camera_get(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    let camera = s
        .camera
        .as_ref()
        .ok_or("no Camera3D snapshot (world has no camera, or tick has not run yet)")?;
    framed(s.frame, camera)
}

// The token is attached before the run loop starts, so a missing one means the
// loop has not been entered yet.
fn shutdown(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    let token = s
        .shutdown_token
        .as_ref()
        .ok_or("shutdown token not attached yet")?;
    token.cancel();
    Ok(json!({ "shutdown": true }))
}

// `tick` captures the flag once the backend exposes it: never under `cn run` or
// a backend without shader hot-reload.
fn reload_shaders(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    let flag = s
        .shader_reload
        .as_ref()
        .ok_or("shader hot-reload not available (cn debug only)")?;
    flag.store(true, Ordering::SeqCst);
    Ok(json!({ "reload_queued": true }))
}

// `tick` captures the signals once the reload driver is armed, which needs
// `cn debug` and a graphics init that parked its reload sources.
fn reload_assets(call: &Call, _: Args) -> Reply {
    let s = call.snapshot();
    let signals = s.reload.as_ref().ok_or(
        "asset hot-reload not available (cn debug only; no file-backed textures captured yet)",
    )?;
    signals.request_all();
    Ok(json!({ "reload_queued": true }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::hot_reload::ReloadSignals;
    use crate::debug::state::{
        AssetEntry, BudgetMemory, BudgetSnapshot, BudgetThreads, CameraSnapshot, DebugState,
        PressureSnapshot,
    };
    use crate::debug::verbs::testing::ask;
    use concinnity_core::profile::RenderStats;
    use concinnity_engine::gfx::streaming::system::StreamingStats;
    use concinnity_engine::shutdown::ShutdownToken;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    fn answer(name: &str, state: DebugState) -> Value {
        ask(state, name).expect("the snapshot answers")
    }

    #[test]
    fn ping_pongs() {
        assert_eq!(
            answer("ping", DebugState::default()),
            json!({ "pong": true })
        );
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
        let r = answer("state", st);
        assert_eq!(r["frame"], 42);
        assert_eq!(r["system_count"], 3);
        assert_eq!(r["component_count"], 7);
        assert_eq!(r["systems"][0], "GraphicsSystem");
    }

    #[test]
    fn budget_reports_threads_and_memory() {
        let st = DebugState {
            frame: 9,
            budget: Some(BudgetSnapshot {
                threads: BudgetThreads {
                    total_cores: 10,
                    job_threads: 9,
                },
                memory: BudgetMemory {
                    total_ram_mib: Some(65536),
                    budget_mib: 16384,
                    overridden: true,
                    rss_mib: Some(512),
                },
            }),
            ..Default::default()
        };
        assert_eq!(
            answer("budget", st),
            json!({
                "frame": 9,
                "threads": { "total_cores": 10, "job_threads": 9 },
                "memory": {
                    "total_ram_mib": 65536,
                    "budget_mib": 16384,
                    "overridden": true,
                    "rss_mib": 512,
                },
            })
        );
    }

    #[test]
    fn budget_before_start_is_not_ready() {
        assert!(ask(DebugState::default(), "budget").is_err());
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
        let r = answer("assets", st);
        assert_eq!(r["assets"][0]["discriminant"], 5);
        assert_eq!(r["assets"][0]["count"], 12);
    }

    #[test]
    fn names_returns_id_table() {
        let st = DebugState {
            names: Arc::new(vec!["hero".into(), "floor".into()]),
            ..Default::default()
        };
        assert_eq!(answer("names", st)["names"][1], "floor");
    }

    #[test]
    fn streaming_pools_are_null_when_absent() {
        let r = answer("streaming", DebugState::default());
        assert!(r["texture"].is_null());
        assert!(r["chunk"].is_null());
        assert!(r["pressure"].is_null());
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
        let r = answer("streaming", st);
        assert_eq!(r["frame"], 9);
        assert_eq!(
            r["texture"],
            json!({
                "resident": 10,
                "pending": 2,
                "unloaded": 1,
                "resident_bytes": 2048,
                "byte_budget": 4096,
            })
        );
        // A 0 byte_budget flags a count-only pool.
        assert_eq!(r["mesh"]["byte_budget"], 0);
        assert_eq!(
            r["chunk"],
            json!({
                "resident": 7,
                "pending": 5,
                "resident_bytes": 3072,
                "byte_budget": 8192,
            })
        );
    }

    #[test]
    fn streaming_reports_ram_back_off_pressure() {
        let st = DebugState {
            streaming_pressure: Some(PressureSnapshot {
                rss_bytes: 900,
                budget_bytes: 1000,
                under_pressure: true,
            }),
            ..Default::default()
        };
        assert_eq!(
            answer("streaming", st)["pressure"],
            json!({ "rss_bytes": 900, "budget_bytes": 1000, "under_pressure": true })
        );
    }

    #[test]
    fn profile_reports_system_timings() {
        let st = DebugState {
            profile_systems: vec![("GraphicsSystem".into(), 1234)],
            ..Default::default()
        };
        let r = answer("profile", st);
        assert_eq!(r["systems"][0]["name"], "GraphicsSystem");
        assert_eq!(r["systems"][0]["micros"], 1234);
        assert!(r["render"]["passes"].is_array());
        assert_eq!(r["render"]["gpu_wait_us"], 0);
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
        let r = answer("profile", st);
        assert_eq!(r["frame_allocs"], 29);
        assert_eq!(r["systems"][0]["allocs"], 0);
        assert_eq!(r["systems"][1]["allocs"], 17);
    }

    #[test]
    fn profile_reports_only_the_populated_render_passes() {
        let mut render = RenderStats {
            draw_calls: 128,
            objects: 64,
            ..Default::default()
        };
        render.pass_times_us[0] = ("shadow", 900);
        let st = DebugState {
            profile_render: render,
            ..Default::default()
        };
        let r = answer("profile", st);
        assert_eq!(r["render"]["draw_calls"], 128);
        assert_eq!(r["render"]["objects"], 64);
        assert_eq!(
            r["render"]["passes"],
            json!([{ "name": "shadow", "micros": 900 }])
        );
    }

    #[test]
    fn camera_get_reports_the_pose_after_the_frame() {
        let st = DebugState {
            frame: 2,
            camera: Some(CameraSnapshot {
                position: [1.0, 2.0, 3.0],
                yaw: 0.5,
                pitch: -0.25,
                fov_y_degrees: 60.0,
                near: 0.125,
                view_distance: None,
            }),
            ..Default::default()
        };
        assert_eq!(
            answer("camera-get", st),
            json!({
                "frame": 2,
                "position": [1.0, 2.0, 3.0],
                "yaw": 0.5,
                "pitch": -0.25,
                "fov_y_degrees": 60.0,
                "near": 0.125,
                "view_distance": null,
            })
        );
    }

    #[test]
    fn camera_get_errors_when_absent() {
        assert!(ask(DebugState::default(), "camera-get").is_err());
    }

    #[test]
    fn shutdown_cancels_the_attached_token() {
        let token = ShutdownToken::new();
        let st = DebugState {
            shutdown_token: Some(token.clone()),
            ..Default::default()
        };
        assert_eq!(answer("shutdown", st), json!({ "shutdown": true }));
        assert!(token.is_canceled(), "the run loop's token must be canceled");
    }

    #[test]
    fn a_handle_the_snapshot_has_not_captured_is_an_error() {
        for verb in ["shutdown", "reload-shaders", "reload-assets"] {
            assert!(ask(DebugState::default(), verb).is_err(), "{verb}");
        }
    }

    #[test]
    fn reload_shaders_flips_the_captured_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        let st = DebugState {
            shader_reload: Some(Arc::clone(&flag)),
            ..Default::default()
        };
        assert_eq!(
            answer("reload-shaders", st),
            json!({ "reload_queued": true })
        );
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn reload_assets_requests_every_reload() {
        let signals = Arc::new(ReloadSignals::default());
        let st = DebugState {
            reload: Some(Arc::clone(&signals)),
            ..Default::default()
        };
        assert_eq!(
            answer("reload-assets", st),
            json!({ "reload_queued": true })
        );
        assert!(signals.take_assets());
        assert!(signals.take_world());
        assert!(signals.take_shaders().all);
        assert!(signals.take_sdf_volumes().all);
        assert!(signals.take_animations());
        // Stories reload only on their own `.md` watch.
        assert!(!signals.take_stories());
    }
}
