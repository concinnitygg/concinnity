// The background thread that builds a planet's ground patches, so sampling
// tens of thousands of heights never lands on a frame.

use concinnity_engine::planet::{GroundBuilder, PatchRequest, PatchResult, PlanetShape};

use concinnity_host::thread::worker::Worker;

// The ground worker for a planet of `shape`, or `None` (with a warning) when its
// thread cannot start.
pub(super) fn start_ground_worker(shape: PlanetShape) -> Option<Box<dyn GroundBuilder>> {
    let spawned = Worker::spawn_keyed(
        "cn-planet-ground",
        1,
        |request: &PatchRequest| request.id,
        move |request: PatchRequest| request.build(&shape),
    );
    match spawned {
        Ok(worker) => Some(Box::new(GroundWorker(worker))),
        Err(e) => {
            tracing::warn!("planet: no ground worker, the ground stays put: {e}");
            None
        }
    }
}

#[derive(Debug)]
struct GroundWorker(Worker<PatchRequest, PatchResult, u64>);

impl GroundBuilder for GroundWorker {
    fn request(&mut self, request: PatchRequest) -> bool {
        self.0.send(request).is_ok()
    }

    // A build that panicked comes back with no patch.
    fn try_recv(&mut self) -> Option<PatchResult> {
        self.0
            .try_recv()
            .map(|answer| answer.unwrap_or_else(|id| PatchResult { id, patch: None }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::planet::LocalFrame;

    #[test]
    fn the_worker_builds_what_it_is_asked_for() {
        let shape = PlanetShape {
            center: [0.0, -5_000.0, 0.0],
            radius: 5_000.0,
            amplitude: 5.0,
            feature_size: 200.0,
            octaves: 3,
            seed: 2,
            continents: concinnity_engine::planet::Continents::NONE,
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
