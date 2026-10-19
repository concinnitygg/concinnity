// The background thread that builds a planet's ground patches, so sampling
// tens of thousands of heights never lands on a frame.

use concinnity_core::planet::{GroundBuilder, PatchRequest, PatchResult, PlanetShape};

use crate::worker::{Job, Worker};

impl Job for PatchRequest {
    type Key = u64;

    fn key(&self) -> u64 {
        self.id
    }
}

// The ground worker for a planet of `shape`, or `None` (with a warning) when its
// thread cannot start.
pub(crate) fn start_ground_worker(shape: PlanetShape) -> Option<Box<dyn GroundBuilder>> {
    let spawned = Worker::spawn("cn-planet-ground", 1, move |request: PatchRequest| {
        request.build(&shape)
    });
    match spawned {
        Ok(worker) => Some(Box::new(worker)),
        Err(e) => {
            tracing::warn!("planet: no ground worker, the ground stays put: {e}");
            None
        }
    }
}

impl GroundBuilder for Worker<PatchRequest, PatchResult> {
    fn request(&mut self, request: PatchRequest) -> bool {
        self.send(request).is_ok()
    }

    // A build that panicked comes back with no patch.
    fn try_recv(&mut self) -> Option<PatchResult> {
        Worker::try_recv(self)
            .map(|answer| answer.unwrap_or_else(|id| PatchResult { id, patch: None }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::planet::LocalFrame;

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
        let mut worker = start_ground_worker(shape).expect("thread");
        let request = PatchRequest {
            id: 7,
            frame: LocalFrame::AUTHORED,
            around: [10.0, 0.0, -4.0],
        };
        let expected = request.build(&shape);
        assert!(worker.request(request));
        let result = concinnity_testing::eventually("the patch", || worker.try_recv());
        assert_eq!(result, expected);
        assert!(result.patch.is_some());
    }
}
