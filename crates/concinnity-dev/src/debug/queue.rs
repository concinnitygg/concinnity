//! The work a verb hands to the engine thread.
//!
//! A handler runs on a connection thread, but the world, the animation system
//! and the render backend live on the engine thread. A verb that needs one of
//! them queues a closure here; the per-frame debug drive in `super::wire` runs
//! it at the next frame start, before the world step. A backend job waits in
//! the queue until a backend is parked, so a call made before the render loop
//! exists is applied once it does; a world job never waits on one.

use concinnity_core::ecs::World;
use concinnity_core::render::backend::RenderBackend;
use concinnity_engine::live_edit::parked::TextureNameSlots;
use std::sync::{Arc, Mutex, MutexGuard};

use super::verbs::camera::CameraMotion;

/// Work applied against the live ECS, with the debug server's camera-motion
/// slot beside it.
pub(super) type WorldJob = Box<dyn FnOnce(&mut World, &mut Option<CameraMotion>) + Send>;

/// Work applied against the parked render backend, with the texture-name table
/// graphics init captured.
pub(super) type BackendJob =
    Box<dyn FnOnce(&mut dyn RenderBackend, Option<&TextureNameSlots>) + Send>;

/// Every job waiting for the next frame, each kind in queue order.
#[derive(Default)]
pub(super) struct Jobs {
    pub(super) world: Vec<WorldJob>,
    pub(super) backend: Vec<BackendJob>,
}

/// The debug server's job queue. Handlers push onto a clone; the per-frame
/// drive takes everything queued. A poisoned mutex is recovered and used
/// regardless, so an unrelated panic in another thread cannot drop jobs.
#[derive(Clone, Default)]
pub(super) struct RuntimeQueue(Arc<Mutex<Jobs>>);

impl RuntimeQueue {
    pub(super) fn push_world(&self, job: WorldJob) {
        self.jobs().world.push(job);
    }

    pub(super) fn push_backend(&self, job: BackendJob) {
        self.jobs().backend.push(job);
    }

    /// Put backend jobs that could not run back at the front of the queue,
    /// ahead of anything queued since they were taken.
    pub(super) fn requeue_backend(&self, jobs: Vec<BackendJob>) {
        self.jobs().backend.splice(0..0, jobs);
    }

    /// Take every queued job, leaving the queue empty.
    pub(super) fn take(&self) -> Jobs {
        std::mem::take(&mut *self.jobs())
    }

    fn jobs(&self) -> MutexGuard<'_, Jobs> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::render::backend::NullBackend;
    use std::sync::mpsc;

    fn world_job(tx: &mpsc::Sender<&'static str>, tag: &'static str) -> WorldJob {
        let tx = tx.clone();
        Box::new(move |_, _| tx.send(tag).unwrap())
    }

    fn backend_job(tx: &mpsc::Sender<&'static str>, tag: &'static str) -> BackendJob {
        let tx = tx.clone();
        Box::new(move |_, _| tx.send(tag).unwrap())
    }

    // A clone is a handle to the same queue: the drive sees what a handler
    // pushed through its own clone, each kind in order.
    #[test]
    fn a_take_returns_every_job_of_each_kind_in_queue_order() {
        let queue = RuntimeQueue::default();
        let handle = queue.clone();
        let (tx, rx) = mpsc::channel();
        handle.push_world(world_job(&tx, "w1"));
        handle.push_backend(backend_job(&tx, "b1"));
        handle.push_world(world_job(&tx, "w2"));

        let jobs = queue.take();
        assert_eq!((jobs.world.len(), jobs.backend.len()), (2, 1));
        let mut world = World::new();
        for job in jobs.world {
            job(&mut world, &mut None);
        }
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), ["w1", "w2"]);
        let taken = handle.take();
        assert!(taken.world.is_empty() && taken.backend.is_empty());
    }

    #[test]
    fn requeued_backend_jobs_run_ahead_of_later_ones() {
        let queue = RuntimeQueue::default();
        let (tx, rx) = mpsc::channel();
        queue.push_backend(backend_job(&tx, "early"));
        let early = queue.take().backend;
        queue.push_backend(backend_job(&tx, "late"));
        queue.requeue_backend(early);

        let mut backend = NullBackend;
        for job in queue.take().backend {
            job(&mut backend, None);
        }
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), ["early", "late"]);
    }

    // A poisoned queue is recovered rather than swallowing jobs: an unrelated
    // panic must not silently break runtime control.
    #[test]
    fn a_poisoned_queue_still_takes_jobs() {
        let queue = RuntimeQueue::default();
        let poisoner = queue.clone();
        let _ = std::thread::spawn(move || {
            let _held = poisoner.0.lock().unwrap();
            panic!("poison");
        })
        .join();
        assert!(queue.0.is_poisoned());

        let (tx, _rx) = mpsc::channel();
        queue.push_world(world_job(&tx, "w"));
        assert_eq!(queue.take().world.len(), 1);
    }
}
