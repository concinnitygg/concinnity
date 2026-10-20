//! What a verb's handler is handed: the world snapshot to read, and a way to run
//! work on the engine thread and wait for its result.

use concinnity_core::ecs::World;
use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::world_load::source_maps::TextureNameSlots;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use super::queue::{BackendJob, RuntimeQueue};
use super::state::DebugState;
use super::verbs::camera::CameraMotion;

/// How long a call waits for the engine to run its job. The drive runs once per
/// frame, so a healthy engine answers within a frame; the headroom covers a slow
/// boot frame without leaving a client hanging on a stalled engine.
pub(super) const REPLY_TIMEOUT: Duration = Duration::from_secs(1);

/// The wait for a job that idles the GPU and copies a resource back.
pub(super) const READBACK_TIMEOUT: Duration = Duration::from_secs(5);

/// One call in flight.
pub(super) struct Call<'a> {
    verb: &'static str,
    shared: &'a Mutex<DebugState>,
    queue: RuntimeQueue,
}

impl<'a> Call<'a> {
    pub(super) fn new(verb: &'static str, shared: &'a Mutex<DebugState>) -> Self {
        let queue = lock(shared).queue.clone();
        Self {
            verb,
            shared,
            queue,
        }
    }

    /// The world snapshot `tick` refreshes each frame.
    pub(super) fn snapshot(&self) -> MutexGuard<'a, DebugState> {
        lock(self.shared)
    }

    /// Run `apply` against the live world at the next frame start and return
    /// its result.
    #[expect(
        clippy::let_underscore_must_use,
        reason = "the caller may have stopped waiting, which is not the engine's error"
    )]
    pub(super) fn on_world<T: Send + 'static>(
        &self,
        apply: impl FnOnce(&mut World, &mut Option<CameraMotion>) -> Result<T, String> + Send + 'static,
    ) -> Result<T, String> {
        let (reply, answer) = sync_channel(1);
        self.queue.push_world(Box::new(move |world, motion| {
            let _ = reply.send(apply(world, motion));
        }));
        wait(self.verb, &answer, REPLY_TIMEOUT)
    }

    /// Run `apply` against the parked render backend once one exists, waiting
    /// up to `timeout` for its result.
    #[expect(
        clippy::let_underscore_must_use,
        reason = "the caller may have stopped waiting, which is not the engine's error"
    )]
    pub(super) fn on_backend<T: Send + 'static>(
        &self,
        timeout: Duration,
        apply: impl FnOnce(&mut dyn RenderBackend, Option<&TextureNameSlots>) -> Result<T, String>
        + Send
        + 'static,
    ) -> Result<T, String> {
        let (reply, answer) = sync_channel(1);
        self.queue.push_backend(Box::new(move |backend, slots| {
            let _ = reply.send(apply(backend, slots));
        }));
        wait(self.verb, &answer, timeout)
    }

    /// Run `apply` against the parked render backend on each of `count`
    /// consecutive frames, handing it the run's index, and return each result
    /// beside the frame it ran on. Each run re-queues the next, so no frame is
    /// skipped; the first error ends the run, and each frame gets `timeout`.
    pub(super) fn on_backend_frames<T: Send + 'static>(
        &self,
        count: usize,
        timeout: Duration,
        apply: impl FnMut(&mut dyn RenderBackend, usize) -> Result<T, String> + Send + 'static,
    ) -> Result<Vec<(u64, T)>, String> {
        let (reply, answer) = sync_channel(count.max(1));
        let first = frame_job(self.queue.clone(), reply, apply, 0, count);
        self.queue.push_backend(first);
        (0..count)
            .map(|_| wait(self.verb, &answer, timeout))
            .collect()
    }
}

// One run of a per-frame job: answer, then queue the next run for the next
// frame. A refused send means the caller stopped waiting, so the chain ends.
fn frame_job<T: Send + 'static>(
    queue: RuntimeQueue,
    reply: SyncSender<Result<(u64, T), String>>,
    mut apply: impl FnMut(&mut dyn RenderBackend, usize) -> Result<T, String> + Send + 'static,
    index: usize,
    count: usize,
) -> BackendJob {
    Box::new(move |backend, _| {
        let result = apply(backend, index).map(|value| (queue.frame(), value));
        let failed = result.is_err();
        if reply.send(result).is_ok() && !failed && index + 1 < count {
            let next = frame_job(queue.clone(), reply, apply, index + 1, count);
            queue.push_backend(next);
        }
    })
}

// A panicked connection thread must not take the snapshot down with it.
fn lock(shared: &Mutex<DebugState>) -> MutexGuard<'_, DebugState> {
    shared
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn wait<T>(
    verb: &str,
    answer: &Receiver<Result<T, String>>,
    timeout: Duration,
) -> Result<T, String> {
    answer
        .recv_timeout(timeout)
        .unwrap_or_else(|_| Err(format!("{verb}: timed out waiting for engine")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::render::backend::NullBackend;
    use std::sync::Arc;

    // Stand in for the engine: stamp each frame and run what it queued until
    // the call on the worker returns.
    fn drive<T: Send + 'static>(
        call: impl FnOnce(&Call<'_>) -> T + Send + 'static,
    ) -> (T, RuntimeQueue) {
        let shared = Arc::new(Mutex::new(DebugState::default()));
        let queue = lock(&shared).queue.clone();
        let worker = {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || call(&Call::new("burst", &shared)))
        };
        let mut frame = 100;
        while !worker.is_finished() {
            queue.set_frame(frame);
            for job in queue.take().backend {
                job(&mut NullBackend, None);
            }
            frame += 1;
            std::thread::yield_now();
        }
        (worker.join().expect("call panicked"), queue)
    }

    #[test]
    fn a_per_frame_job_runs_once_on_each_consecutive_frame() {
        let (runs, _) =
            drive(|call| call.on_backend_frames(5, Duration::from_secs(5), |_, index| Ok(index)));
        let runs = runs.expect("every run answered");
        assert_eq!(
            runs.iter().map(|r| r.1).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
        assert!(
            runs.windows(2).all(|w| w[1].0 == w[0].0 + 1),
            "frames {runs:?}"
        );
    }

    #[test]
    fn a_failed_run_ends_the_per_frame_job() {
        let (runs, queue) = drive(|call| {
            call.on_backend_frames(4, Duration::from_secs(5), |_, index| {
                if index == 1 {
                    Err("burst: frame 1 failed".to_string())
                } else {
                    Ok(index)
                }
            })
        });
        assert_eq!(runs, Err("burst: frame 1 failed".to_string()));
        assert!(queue.take().backend.is_empty(), "nothing re-queued");
    }

    #[test]
    fn an_unanswered_job_times_out_naming_the_verb() {
        let (_reply, answer) = sync_channel::<Result<(), String>>(1);
        let result = wait("camera-stop", &answer, Duration::from_millis(5));
        assert_eq!(
            result,
            Err("camera-stop: timed out waiting for engine".into())
        );
    }

    #[test]
    fn a_dropped_job_is_reported_like_a_timeout() {
        let (reply, answer) = sync_channel::<Result<(), String>>(1);
        drop(reply);
        let result = wait("spawn", &answer, Duration::from_secs(5));
        assert_eq!(result, Err("spawn: timed out waiting for engine".into()));
    }
}
