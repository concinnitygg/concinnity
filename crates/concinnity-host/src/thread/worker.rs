//! Background threads that answer requests off the frame thread: a request
//! channel per thread, handed work round robin, and one channel the answers come
//! back on. A load that panics is answered with its request's key, so whoever
//! queued it can retire it rather than wait forever. Drop cancels what is still
//! queued, hangs up and joins, so a world rebuild never leaks a thread or waits
//! out a backlog.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

use super::{ThreadRole, set_current_thread_role};

/// What came back for a request: its result, or its key when the load panicked.
pub type Answer<Res, K> = Result<Res, K>;

struct Thread<Req> {
    requests: Option<Sender<Req>>,
    handle: Option<JoinHandle<()>>,
}

/// Threads answering `Req` with `Res`; a request that panicked comes back as
/// its key `K`.
pub struct Worker<Req, Res, K = Req> {
    threads: Vec<Thread<Req>>,
    answers: Receiver<Answer<Res, K>>,
    cancel: Arc<AtomicBool>,
    next: usize,
}

impl<Req, Res, K> std::fmt::Debug for Worker<Req, Res, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("threads", &self.threads.len())
            .finish_non_exhaustive()
    }
}

impl<Req, Res> Worker<Req, Res>
where
    Req: Copy + Send + 'static,
    Res: Send + 'static,
{
    /// `threads` threads named `name-N` (or `name` alone for one), each
    /// answering a request with `load`; a request is its own key. Fails when
    /// the OS refuses a thread.
    pub fn spawn(
        name: &str,
        threads: usize,
        load: impl Fn(Req) -> Res + Send + Sync + 'static,
    ) -> std::io::Result<Self> {
        Self::spawn_keyed(name, threads, |request| *request, load)
    }
}

impl<Req, Res, K> Worker<Req, Res, K>
where
    Req: Send + 'static,
    Res: Send + 'static,
    K: Send + 'static,
{
    /// As [`Worker::spawn`], with `key` naming what a panicked request
    /// answers with.
    pub fn spawn_keyed(
        name: &str,
        threads: usize,
        key: fn(&Req) -> K,
        load: impl Fn(Req) -> Res + Send + Sync + 'static,
    ) -> std::io::Result<Self> {
        let load = Arc::new(load);
        let cancel = Arc::new(AtomicBool::new(false));
        let (answer_tx, answers) = std::sync::mpsc::channel();
        let count = threads.max(1);
        let mut worker = Self {
            threads: Vec::with_capacity(count),
            answers,
            cancel,
            next: 0,
        };
        for i in 0..count {
            let thread_name = if count == 1 {
                name.to_string()
            } else {
                format!("{name}-{i}")
            };
            let (request_tx, requests) = std::sync::mpsc::channel::<Req>();
            let (load, answers, cancel) = (
                Arc::clone(&load),
                answer_tx.clone(),
                Arc::clone(&worker.cancel),
            );
            let handle = std::thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    set_current_thread_role(ThreadRole::Background);
                    answer_each(&requests, &answers, &cancel, key, &*load);
                })?;
            worker.threads.push(Thread {
                requests: Some(request_tx),
                handle: Some(handle),
            });
        }
        Ok(worker)
    }

    /// Queue one request on the next live thread, handing it back when every
    /// thread is gone.
    pub fn send(&mut self, request: Req) -> Result<(), Req> {
        let mut request = request;
        for _ in 0..self.threads.len() {
            let thread = &self.threads[self.next % self.threads.len()];
            self.next = self.next.wrapping_add(1);
            if let Some(tx) = &thread.requests {
                match tx.send(request) {
                    Ok(()) => return Ok(()),
                    Err(refused) => request = refused.0,
                }
            }
        }
        Err(request)
    }

    /// One finished answer, if any is waiting. Never blocks.
    pub fn try_recv(&self) -> Option<Answer<Res, K>> {
        self.answers.try_recv().ok()
    }
}

impl<Req, Res, K> Drop for Worker<Req, Res, K> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        for thread in &mut self.threads {
            thread.requests = None;
        }
        for thread in &mut self.threads {
            if let Some(handle) = thread.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

// A thread's loop: answer each request until the channel closes, skipping
// what is still queued once the worker is canceled.
fn answer_each<Req, Res, K>(
    requests: &Receiver<Req>,
    answers: &Sender<Answer<Res, K>>,
    cancel: &AtomicBool,
    key: fn(&Req) -> K,
    load: &dyn Fn(Req) -> Res,
) {
    while let Ok(request) = requests.recv() {
        if cancel.load(Ordering::Relaxed) {
            continue;
        }
        let key = key(&request);
        let answer = catch_unwind(AssertUnwindSafe(|| load(request))).map_err(|_| {
            tracing::warn!("a background load panicked; its request is answered as failed");
            key
        });
        if answers.send(answer).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Every answer waiting within the deadline, until `count` arrived.
    fn collect(worker: &Worker<usize, usize>, count: usize) -> Vec<Answer<usize, usize>> {
        let mut got = Vec::new();
        concinnity_testing::eventually("the answers", || {
            got.extend(std::iter::from_fn(|| worker.try_recv()));
            (got.len() >= count).then_some(())
        });
        got
    }

    #[test]
    fn requests_round_trip_through_the_threads() {
        let mut worker = Worker::spawn("test-load", 2, |n: usize| n * 10).expect("threads");
        for n in 1..=3 {
            assert!(worker.send(n).is_ok());
        }
        let mut got: Vec<usize> = collect(&worker, 3).into_iter().flatten().collect();
        got.sort_unstable();
        assert_eq!(got, [10, 20, 30]);
    }

    // A load that panics is answered with its key, and the thread keeps
    // answering.
    #[test]
    fn a_panicking_load_answers_its_key_and_the_thread_survives() {
        let mut worker = Worker::spawn("test-panic", 1, |n: usize| {
            assert!(n != 7, "a load that cannot finish");
            n
        })
        .expect("thread");
        assert!(worker.send(7).is_ok());
        assert!(worker.send(8).is_ok());
        assert_eq!(collect(&worker, 2), [Err(7), Ok(8)]);
    }

    // A request no live thread takes is handed back.
    #[test]
    fn a_request_with_no_live_thread_is_handed_back() {
        let mut worker = Worker::spawn("test-refuse", 1, |n: usize| n).expect("thread");
        worker.threads[0].requests = None;
        assert_eq!(worker.send(3), Err(3));
    }

    // Dropping the worker skips what is still queued rather than loading it.
    #[test]
    fn drop_skips_the_queued_loads() {
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&loads);
        let mut worker = Worker::spawn("test-cancel", 1, move |n: usize| {
            std::thread::sleep(Duration::from_millis(20));
            counted.fetch_add(1, Ordering::Relaxed);
            n
        })
        .expect("thread");
        for n in 0..50 {
            assert!(worker.send(n).is_ok());
        }
        drop(worker);
        assert!(loads.load(Ordering::Relaxed) < 50);
    }
}
