//! Off-thread recompiles. Each request runs on its own worker so the frame loop
//! keeps drawing while dxc works, and results are tagged with the request's
//! generation: only a key's newest request is applied, so an older compile that
//! finishes late never overwrites a newer save.

use concinnity_host::thread::{ThreadRole, set_current_thread_role};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::mpsc::{Receiver, Sender, channel};

// One worker's result, tagged with the request it answers.
#[derive(Debug)]
pub(super) struct Finished<K, T> {
    pub key: K,
    pub generation: u64,
    pub result: T,
}

// The generation each key's newest request was given. A result answers the
// newest request only when its generation matches.
#[derive(Debug)]
pub(super) struct Generations<K> {
    latest: HashMap<K, u64>,
    next: u64,
}

impl<K> Default for Generations<K> {
    fn default() -> Self {
        Self {
            latest: HashMap::new(),
            next: 0,
        }
    }
}

impl<K: Eq + Hash + Clone> Generations<K> {
    // Start a request for `key`, superseding any still in flight.
    pub(super) fn begin(&mut self, key: K) -> u64 {
        self.next += 1;
        self.latest.insert(key, self.next);
        self.next
    }

    // The result to apply, or `None` when a newer request for the key has
    // started since this one.
    pub(super) fn accept<T>(&self, finished: Finished<K, T>) -> Option<(K, T)> {
        (self.latest.get(&finished.key) == Some(&finished.generation))
            .then_some((finished.key, finished.result))
    }
}

// The in-flight recompiles and the channel their results come back on.
pub(super) struct CompileQueue<K, T> {
    generations: Generations<K>,
    tx: Sender<Finished<K, T>>,
    rx: Receiver<Finished<K, T>>,
}

impl<K, T> CompileQueue<K, T>
where
    K: Eq + Hash + Clone + Debug + Send + 'static,
    T: Send + 'static,
{
    pub(super) fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            generations: Generations::default(),
            tx,
            rx,
        }
    }

    // Run `job` on a worker as the newest request for `key`.
    pub(super) fn submit(
        &mut self,
        key: K,
        job: impl FnOnce() -> T + Send + 'static,
    ) -> Result<(), String> {
        let generation = self.generations.begin(key.clone());
        let tx = self.tx.clone();
        std::thread::Builder::new()
            .name("cn-shader-reload".into())
            .spawn(move || {
                set_current_thread_role(ThreadRole::Background);
                // A dropped receiver means the reload state was rebuilt for
                // another world; the result is no longer wanted.
                let _ = tx.send(Finished {
                    key,
                    generation,
                    result: job(),
                });
            })
            .map(drop)
            .map_err(|e| format!("could not spawn the compile worker: {e}"))
    }

    // Every result that has arrived and still answers its key's newest
    // request. Never blocks.
    pub(super) fn drain(&mut self) -> Vec<(K, T)> {
        let generations = &self.generations;
        self.rx
            .try_iter()
            .filter_map(|finished| {
                let generation = finished.generation;
                let key = finished.key.clone();
                let accepted = generations.accept(finished);
                if accepted.is_none() {
                    tracing::debug!(
                        "hot-reload: dropped a superseded compile ({key:?}, request {generation})"
                    );
                }
                accepted
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn finished(key: u32, generation: u64) -> Finished<u32, ()> {
        Finished {
            key,
            generation,
            result: (),
        }
    }

    // Whatever order two compiles of one key finish in, only the newer applies.
    #[test]
    fn only_a_keys_newest_request_is_accepted() {
        let mut generations = Generations::default();
        let older = generations.begin(1);
        let newer = generations.begin(1);
        assert!(generations.accept(finished(1, older)).is_none());
        assert!(generations.accept(finished(1, newer)).is_some());
        // A late older result after the newer one applied is still stale.
        assert!(generations.accept(finished(1, older)).is_none());
    }

    // Requests for different keys never supersede each other.
    #[test]
    fn requests_for_different_keys_are_independent() {
        let mut generations = Generations::default();
        let water = generations.begin(1);
        let cave = generations.begin(2);
        assert!(generations.accept(finished(1, water)).is_some());
        assert!(generations.accept(finished(2, cave)).is_some());
        assert!(generations.accept(finished(3, cave)).is_none());
    }

    // On real workers: an older compile that finishes after a newer one is
    // dropped instead of overwriting it.
    #[test]
    fn a_stale_compile_finishing_late_is_dropped() {
        let mut queue = CompileQueue::new();
        let (release_old, old_gate) = channel::<()>();
        let (old_done, old_finished) = channel::<()>();
        queue
            .submit(7, move || {
                let _ = old_gate.recv();
                let _ = old_done.send(());
                "old"
            })
            .unwrap();
        queue.submit(7, || "new").unwrap();

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut applied = Vec::new();
        while applied.is_empty() && Instant::now() < deadline {
            applied = queue.drain();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(applied, [(7, "new")]);

        release_old.send(()).unwrap();
        old_finished
            .recv_timeout(Duration::from_secs(10))
            .expect("the old compile ran");
        std::thread::sleep(Duration::from_millis(50));
        assert!(queue.drain().is_empty());
    }
}
