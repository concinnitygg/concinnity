// The background thread that builds a planet's ground patches, so sampling
// tens of thousands of heights never lands on a frame.

use concinnity_core::planet::{GroundPatch, LocalFrame, PlanetShape, build_patch};

use crate::worker::{Job, Worker};

// One patch to build: around local point `around` of `frame`, tagged `id`.
pub(super) struct PatchRequest {
    pub(super) id: u64,
    pub(super) frame: LocalFrame,
    pub(super) around: [f32; 3],
}

// What came back for request `id`: the patch, or `None` where the ground
// cannot be built there (a camera off the planet).
pub(super) struct PatchResult {
    pub(super) id: u64,
    pub(super) patch: Option<GroundPatch>,
}

impl Job for PatchRequest {
    type Key = u64;

    fn key(&self) -> u64 {
        self.id
    }
}

// The thread building patches.
pub(super) struct GroundWorker {
    worker: Worker<PatchRequest, PatchResult>,
}

impl std::fmt::Debug for GroundWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroundWorker").finish_non_exhaustive()
    }
}

impl GroundWorker {
    pub(super) fn spawn(shape: PlanetShape) -> std::io::Result<Self> {
        let worker = Worker::spawn("cn-planet-ground", 1, move |request: PatchRequest| {
            PatchResult {
                id: request.id,
                patch: build(&shape, &request),
            }
        })?;
        Ok(Self { worker })
    }

    // Queue a patch. `false` when the worker is gone.
    pub(super) fn request(&mut self, request: PatchRequest) -> bool {
        self.worker.send(request).is_ok()
    }

    // A finished patch, if one is waiting. A build that panicked comes back
    // with no patch.
    pub(super) fn try_recv(&self) -> Option<PatchResult> {
        self.worker
            .try_recv()
            .map(|answer| answer.unwrap_or_else(|id| PatchResult { id, patch: None }))
    }
}

// The patch for `request`, built on the calling thread.
pub(super) fn build(shape: &PlanetShape, request: &PatchRequest) -> Option<GroundPatch> {
    build_patch(shape, &request.frame, request.around)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_worker_builds_what_it_is_asked_for() {
        let shape = PlanetShape {
            center: [0.0, -5_000.0, 0.0],
            radius: 5_000.0,
            amplitude: 5.0,
            feature_size: 200.0,
            octaves: 3,
            seed: 2,
        };
        let mut worker = GroundWorker::spawn(shape).expect("thread");
        let request = PatchRequest {
            id: 7,
            frame: LocalFrame::AUTHORED,
            around: [10.0, 0.0, -4.0],
        };
        let expected = build(&shape, &request);
        assert!(worker.request(request));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let result = loop {
            if let Some(r) = worker.try_recv() {
                break r;
            }
            assert!(std::time::Instant::now() < deadline, "no patch came back");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(result.id, 7);
        assert_eq!(result.patch, expected);
        assert!(result.patch.is_some());
    }
}
