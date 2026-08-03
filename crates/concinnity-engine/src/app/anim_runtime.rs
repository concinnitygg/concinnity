// src/app/anim_runtime.rs
//
// Process-wide command queue for runtime animation control (crossfades,
// graph parameter writes, graph state queries). Mirrors the shape of
// `crate::debug::runtime_spawn`, but separate so the AnimationSystem can
// drain its own commands without contending with GraphicsSystem's decal /
// particle queue.
//
// The debug WebSocket server (binary-only, off the engine thread) pushes
// commands here; the editor's per-frame debug drive drains them via
// `AnimationSystem::apply_runtime_commands` every frame -- including while a
// menu pauses playback, so a blocked WS client always gets its reply. Each
// command carries a reply channel the drain fulfils synchronously.

use std::sync::Mutex;

use crate::ecs::asset_id::AssetId;

// One queued crossfade request. `target` is the `SkinnedMesh` asset id the
// command applies to; `weights` must match the clip count registered for
// that target. `duration_secs == 0` snaps to the new weights on the next
// frame. Rejected when the target is graph-driven (use `SetParam`).
//
// `dead_code` allow: the only constructor is the binary-only debug module,
// so `cargo check --lib` sees the struct as unconstructed.
#[derive(Debug)]
pub struct CrossfadeRequest {
    pub target: AssetId,
    pub weights: Vec<f32>,
    pub duration_secs: f32,
}

// One queued graph parameter write. `target` is the `SkinnedMesh` whose
// graph declares the parameter; the value lands in the target's `AnimParams`
// component on the next animation step.
//
// `dead_code` allow: same rationale as `CrossfadeRequest` above.
#[derive(Debug)]
pub struct SetParamRequest {
    pub target: AssetId,
    pub name: String,
    pub value: f32,
}

// Snapshot of a graph target's live state, answered synchronously to the
// `anim-state` debug command. Parameter values are as of the last completed
// animation step (a pending `SetParam` shows up after the next step).
#[derive(Debug, Clone)]
pub struct GraphStateReport {
    pub state: String,
    pub clock_secs: f32,
    pub fading_from: Option<String>,
    pub fade_progress: Option<f32>,
    // One weight per blendspace member (point / grid order); None when the
    // active state plays a single clip.
    pub blend_weights: Option<Vec<f32>>,
    pub params: Vec<(String, f32)>,
}

// One runtime command pushed onto [`enqueue`] by the debug WS server and
// drained by `AnimationSystem::apply_runtime_commands`.
//
// `dead_code` allow: the only producer is the binary-only `crate::debug`
// module (declared by main.rs, not lib.rs), so `cargo check --lib` sees the
// variants as unconstructed.
pub enum AnimCommand {
    Crossfade {
        req: CrossfadeRequest,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    SetParam {
        req: SetParamRequest,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    QueryState {
        target: AssetId,
        reply: std::sync::mpsc::SyncSender<Result<GraphStateReport, String>>,
    },
}

static QUEUE: Mutex<Vec<AnimCommand>> = Mutex::new(Vec::new());

// Serialises the tests that drive the queue. It is process-wide and `drain`
// takes all of it, so two tests enqueuing at once would steal each other's
// commands; every such test holds this for its enqueue + drain.
#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

// Push a command onto the animation runtime queue. The caller blocks on its
// own reply receiver to get the result. A poisoned mutex is recovered and
// used regardless (an unrelated panic must not silently drop commands).
//
// `dead_code` allow: the only producer is the binary-only debug module.
pub fn enqueue(cmd: AnimCommand) {
    let mut q = match QUEUE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    q.push(cmd);
}

// Take every queued command. Drained by `AnimationSystem::apply_runtime_commands`,
// which the `cn debug` drive (`DebugHook::tick`) calls each frame. The
// returned `Vec` is the live list: the queue is reset to empty.
pub(crate) fn drain() -> Vec<AnimCommand> {
    let mut q = match QUEUE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    std::mem::take(&mut *q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_drain_round_trip() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(AnimCommand::Crossfade {
            req: CrossfadeRequest {
                target: AssetId::default(),
                weights: vec![1.0, 0.0],
                duration_secs: 0.5,
            },
            reply: tx,
        });
        let cmds = drain();
        assert_eq!(cmds.len(), 1);
        assert!(drain().is_empty());
    }

    // A poisoned queue is recovered rather than swallowing commands: an
    // unrelated panic must not silently break runtime control.
    #[test]
    fn a_poisoned_queue_still_enqueues_and_drains() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = drain();
        // Poison the queue's own mutex from a panicking thread.
        let _ = std::thread::spawn(|| {
            let _held = QUEUE.lock().unwrap();
            panic!("poison");
        })
        .join();
        assert!(QUEUE.is_poisoned());

        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(AnimCommand::QueryState {
            target: AssetId::default(),
            reply: tx,
        });
        assert_eq!(drain().len(), 1);
        QUEUE.clear_poison();
    }
}
