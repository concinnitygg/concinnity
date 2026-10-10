// The background thread that builds a planet's ground patches, so sampling
// tens of thousands of heights never lands on a frame.

use std::sync::mpsc::{Receiver, Sender};

use concinnity_core::planet::{GroundPatch, LocalFrame, PlanetShape, ground_patch};

use crate::gfx::streaming::worker::Worker;

/// Cells along each side of a ground patch.
pub(super) const PATCH_CELLS: u32 = 128;

/// Half the width of a ground patch, in meters: one-meter cells.
pub(super) const PATCH_HALF_WIDTH: f32 = 64.0;

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

// The worker and the channel its results come back on.
pub(super) struct GroundWorker {
    worker: Worker<PatchRequest>,
    results: Receiver<PatchResult>,
}

impl std::fmt::Debug for GroundWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroundWorker").finish_non_exhaustive()
    }
}

impl GroundWorker {
    pub(super) fn spawn(shape: PlanetShape) -> Self {
        let (request_tx, request_rx) = std::sync::mpsc::channel::<PatchRequest>();
        let (result_tx, results) = std::sync::mpsc::channel::<PatchResult>();
        let worker = Worker::spawn("cn-planet-ground", request_rx, request_tx, move |rx| {
            work(shape, rx, result_tx)
        });
        Self { worker, results }
    }

    // Queue a patch. `false` when the worker is gone.
    pub(super) fn request(&self, request: PatchRequest) -> bool {
        self.worker.send(request)
    }

    // A finished patch, if one is waiting.
    pub(super) fn try_recv(&self) -> Option<PatchResult> {
        self.results.try_recv().ok()
    }
}

// The patch for `request`, built on the calling thread.
pub(super) fn build(shape: &PlanetShape, request: &PatchRequest) -> Option<GroundPatch> {
    ground_patch(
        shape,
        &request.frame,
        request.around,
        PATCH_CELLS,
        PATCH_HALF_WIDTH,
    )
}

fn work(shape: PlanetShape, requests: Receiver<PatchRequest>, results: Sender<PatchResult>) {
    while let Ok(request) = requests.recv() {
        let patch = build(&shape, &request);
        if results
            .send(PatchResult {
                id: request.id,
                patch,
            })
            .is_err()
        {
            break;
        }
    }
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
        let worker = GroundWorker::spawn(shape);
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
