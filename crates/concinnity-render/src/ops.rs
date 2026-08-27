//! Recorded backend effects: simulation systems queue their GPU mutations as
//! ops instead of calling the backend directly, and the submit path replays
//! them in record order before the frame's draw. Ordering across systems is
//! preserved by the single queue, so the GPU-visible result matches the old
//! direct calls exactly. Ops own their payloads, so a queue can cross a thread
//! boundary with the snapshot that carries it.

use crate::backend::RenderBackend;
use alloc::boxed::Box;
use alloc::vec::Vec;
use concinnity_core::gfx::chunk_coord::ChunkCoord;

type BackendOp = Box<dyn FnOnce(&mut dyn RenderBackend, &mut ReplayOutcome) + Send>;

/// An op whose failure the simulation side must observe and roll back;
/// fire-and-forget ops log at replay instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpFailure {
    /// A streamed-mesh upload was refused (transient region exhaustion); the
    /// streamer rolls the mesh back to unloaded and retries later.
    MeshUpload {
        /// The streamed mesh that failed to upload.
        stream_id: usize,
    },
    /// A chunk-mesh add failed; the chunk's tracking and draw slot roll back.
    ChunkAdd {
        /// The chunk whose mesh add failed.
        coord: ChunkCoord,
    },
}

/// What one queue replay produced, for the simulation side.
#[derive(Debug, Default)]
pub struct ReplayOutcome {
    /// Failures the simulation side must roll back.
    pub failures: Vec<OpFailure>,
    /// An op hit device-memory exhaustion; feeds the streaming valve.
    pub memory_pressure: bool,
}

/// Backend effects recorded by simulation systems, replayed in order by the
/// submit path. The buffer keeps its capacity across frames.
#[derive(Default)]
pub struct RenderOps {
    ops: Vec<BackendOp>,
}

impl core::fmt::Debug for RenderOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RenderOps")
            .field("len", &self.ops.len())
            .finish()
    }
}

impl RenderOps {
    /// Record a fire-and-forget backend effect.
    pub fn record(&mut self, op: impl FnOnce(&mut dyn RenderBackend) + Send + 'static) {
        self.ops.push(Box::new(move |backend, _| op(backend)));
    }

    /// Record an effect that reports into the replay outcome (failure
    /// identity, memory pressure).
    pub fn record_with(
        &mut self,
        op: impl FnOnce(&mut dyn RenderBackend, &mut ReplayOutcome) + Send + 'static,
    ) {
        self.ops.push(Box::new(op));
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Ops recorded so far.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Move this queue's ops onto the end of `dst`, leaving this queue empty
    /// with its capacity intact.
    pub fn drain_into(&mut self, dst: &mut RenderOps) {
        dst.ops.append(&mut self.ops);
    }

    /// Replay every op in record order, draining the queue.
    pub fn replay(&mut self, backend: &mut dyn RenderBackend) -> ReplayOutcome {
        let mut outcome = ReplayOutcome::default();
        for op in self.ops.drain(..) {
            op(backend, &mut outcome);
        }
        outcome
    }

    /// Drop every recorded op, keeping the buffer's capacity.
    pub fn clear(&mut self) {
        self.ops.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    // Replay hands ops the backend in record order and drains the queue.
    // The mock records call order through the outcome's failure list.
    #[test]
    fn replay_runs_ops_in_record_order_and_drains() {
        let mut ops = RenderOps::default();
        for i in 0..3 {
            ops.record_with(move |_, out| {
                out.failures.push(OpFailure::MeshUpload { stream_id: i });
            });
        }
        assert_eq!(ops.len(), 3);
        let mut backend = crate::backend::test_stub::StubBackend;
        let outcome = ops.replay(&mut backend);
        let order: Vec<usize> = outcome
            .failures
            .iter()
            .map(|f| match f {
                OpFailure::MeshUpload { stream_id } => *stream_id,
                _ => usize::MAX,
            })
            .collect();
        assert_eq!(order, vec![0, 1, 2]);
        assert!(ops.is_empty(), "replay drains the queue");
    }

    #[test]
    fn drain_into_appends_preserving_order() {
        let mut a = RenderOps::default();
        let mut b = RenderOps::default();
        a.record_with(|_, out| out.failures.push(OpFailure::MeshUpload { stream_id: 1 }));
        b.record_with(|_, out| out.failures.push(OpFailure::MeshUpload { stream_id: 2 }));
        a.drain_into(&mut b);
        assert!(a.is_empty());
        let mut backend = crate::backend::test_stub::StubBackend;
        let outcome = b.replay(&mut backend);
        assert_eq!(
            outcome.failures,
            vec![
                OpFailure::MeshUpload { stream_id: 2 },
                OpFailure::MeshUpload { stream_id: 1 },
            ]
        );
    }
}
