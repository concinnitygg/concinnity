//! Calls answered the way a running engine answers them, for verb tests.

use concinnity_core::ecs::World;
use concinnity_core::render::backend::NullBackend;
use serde_json::{Map, Value};
use std::sync::{Arc, Mutex};

use super::camera::CameraMotion;
use crate::debug::catalog;
use crate::debug::state::DebugState;
use crate::debug::verb::Reply;

/// The arguments a test writes as a JSON object, or none for `null`.
pub(in crate::debug) fn arguments(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(fields) => fields,
        _ => Map::new(),
    }
}

/// Answer a call from `state` alone, for verbs that never wait on the engine.
pub(in crate::debug) fn ask(state: DebugState, name: &str) -> Reply {
    catalog::run(name, Map::new(), &Mutex::new(state))
}

/// A world and the debug server's camera-motion slot, standing in for the
/// engine thread: a call runs on a worker thread while this one applies each
/// queued job, world jobs to `world` and backend jobs to a `NullBackend`.
pub(in crate::debug) struct Engine {
    pub(in crate::debug) world: World,
    pub(in crate::debug) motion: Option<CameraMotion>,
}

impl Engine {
    pub(in crate::debug) fn new(world: World) -> Self {
        Self {
            world,
            motion: None,
        }
    }

    // A loaded test host can stall either thread past the call's wait; a call
    // that timed out is retried, so a test asserts what the verb answers
    // rather than the scheduler.
    pub(in crate::debug) fn call(&mut self, name: &str, args: Value) -> Reply {
        for _ in 0..5 {
            let reply = self.call_once(name, args.clone());
            if !reply
                .as_ref()
                .is_err_and(|e| e.ends_with("timed out waiting for engine"))
            {
                return reply;
            }
        }
        panic!("{name} kept timing out under load");
    }

    fn call_once(&mut self, name: &str, args: Value) -> Reply {
        let shared = Arc::new(Mutex::new(DebugState::default()));
        let queue = shared.lock().unwrap().queue.clone();
        let name = name.to_string();
        let worker = std::thread::spawn(move || catalog::run(&name, arguments(args), &shared));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !worker.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "the call never returned"
            );
            let jobs = queue.take();
            for job in jobs.world {
                job(&mut self.world, &mut self.motion);
            }
            for job in jobs.backend {
                job(&mut NullBackend, None);
            }
            std::thread::yield_now();
        }
        worker.join().expect("call thread panicked")
    }
}
