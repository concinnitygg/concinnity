//! What a verb's handler is handed: the world snapshot to read, and a way to run
//! work on the engine thread and wait for its result.

use concinnity_core::ecs::World;
use concinnity_core::render::backend::RenderBackend;
use concinnity_engine::live_edit::parked::TextureNameSlots;
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use super::queue::RuntimeQueue;
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
    pub(super) fn on_world<T: Send + 'static>(
        &self,
        apply: impl FnOnce(&mut World, &mut Option<CameraMotion>) -> Result<T, String> + Send + 'static,
    ) -> Result<T, String> {
        let (reply, answer) = sync_channel(1);
        self.queue.push_world(Box::new(move |world, motion| {
            // The caller may have stopped waiting, which is not the engine's error.
            let _ = reply.send(apply(world, motion));
        }));
        wait(self.verb, &answer, REPLY_TIMEOUT)
    }

    /// Run `apply` against the parked render backend once one exists, waiting
    /// up to `timeout` for its result.
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
}

// A panicked connection thread must not take the snapshot down with it.
fn lock(shared: &Mutex<DebugState>) -> MutexGuard<'_, DebugState> {
    shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
