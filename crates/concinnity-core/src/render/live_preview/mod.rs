//! Live edits of a running world's renderer state, for an editor previewing a
//! change without a rebuild. Each seam records its backend calls into the
//! frame's op queue, where submission replays them before the next draw, and
//! keeps the world's own copy of the edited state in step.

pub mod draw;
pub mod lighting;
pub mod material;
pub mod shape;

use crate::ecs::World;
use crate::render::frame::ops::RenderOps;
use crate::render::frame::queues::ActiveRenderQueues;

// Run `f` against the frame's op queue. `None` when the world has no queue to
// record into, which leaves the backend exactly as it was.
fn with_ops<R>(world: &mut World, f: impl FnOnce(&mut RenderOps) -> R) -> Option<R> {
    ActiveRenderQueues::with(world.context().resources, |queues| f(&mut queues.ops))
}

// Whether the world has an op queue to record into.
fn has_queue(world: &World) -> bool {
    world
        .resource::<ActiveRenderQueues>()
        .is_some_and(|slot| slot.0.is_some())
}

#[cfg(test)]
mod test_support {
    use alloc::vec::Vec;

    use crate::ecs::World;
    use crate::render::backend::recording::{
        Call, RecordingBackend, SharedRecording, recording_backend,
    };
    use crate::render::frame::ops::RenderOps;
    use crate::render::frame::queues::{ActiveRenderQueues, RenderQueues};
    use crate::render::slots::render_slots::RenderSlots;

    fn with_queue(world: &mut World) {
        world.insert_resource(ActiveRenderQueues::new(RenderQueues {
            ops: RenderOps::default(),
            slots: RenderSlots::new(0, true, &[]),
        }));
    }

    /// A world with an empty op queue, and a backend to replay it into.
    pub(super) struct Fixture {
        pub(super) world: World,
        pub(super) backend: RecordingBackend,
        pub(super) calls: SharedRecording,
    }

    impl Fixture {
        pub(super) fn new(mut world: World) -> Self {
            with_queue(&mut world);
            let (calls, backend) = recording_backend();
            Self {
                world,
                backend,
                calls,
            }
        }

        /// Replay what the queue recorded, park the queue again, and report
        /// every call the backend has seen.
        pub(super) fn replay(&mut self) -> Vec<Call> {
            let backend = &mut self.backend;
            ActiveRenderQueues::with(self.world.context().resources, |queues| {
                queues.ops.replay(backend)
            })
            .expect("the queue is parked again");
            self.calls.lock().calls.clone()
        }

        /// Forget the calls replayed so far.
        pub(super) fn clear(&self) {
            self.calls.lock().calls.clear();
        }
    }
}
