// Background threads that answer requests off the frame thread: a request
// channel per thread, handed work round robin, and one channel the answers come
// back on. A load that panics is answered with its request's key, so whoever
// queued it can retire it rather than wait forever. Drop cancels what is still
// queued, hangs up and joins, so a world rebuild never leaks a thread or waits
// out a backlog.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

use concinnity_core::planet::TileId;
use concinnity_core::render::stream_pools::{ChunkRequest, LoadPool};
use concinnity_host::thread::{ThreadRole, set_current_thread_role};

// A request a worker answers, and the key a failed answer carries back.
pub(crate) trait Job: Send + 'static {
    type Key: Copy + Send + 'static;

    fn key(&self) -> Self::Key;
}

impl Job for usize {
    type Key = usize;

    fn key(&self) -> usize {
        *self
    }
}

impl Job for TileId {
    type Key = TileId;

    fn key(&self) -> TileId {
        *self
    }
}

impl Job for ChunkRequest {
    type Key = ChunkRequest;

    fn key(&self) -> ChunkRequest {
        *self
    }
}

// What came back for a request: its result, or its key when the load panicked.
pub(crate) type Answer<Req, Res> = Result<Res, <Req as Job>::Key>;

struct Thread<Req> {
    requests: Option<Sender<Req>>,
    handle: Option<JoinHandle<()>>,
}

pub(crate) struct Worker<Req: Job, Res> {
    threads: Vec<Thread<Req>>,
    answers: Receiver<Answer<Req, Res>>,
    cancel: Arc<AtomicBool>,
    next: usize,
}

impl<Req: Job, Res: Send + 'static> Worker<Req, Res> {
    // `threads` threads named `name-N` (or `name` alone for one), each
    // answering a request with `load`. Fails when the OS refuses a thread.
    pub(crate) fn spawn(
        name: &str,
        threads: usize,
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
                    answer_each(&requests, &answers, &cancel, &*load);
                })?;
            worker.threads.push(Thread {
                requests: Some(request_tx),
                handle: Some(handle),
            });
        }
        Ok(worker)
    }

    // Queue one request on the next live thread, handing it back when every
    // thread is gone.
    pub(crate) fn send(&mut self, request: Req) -> Result<(), Req> {
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

    // One finished answer, if any is waiting. Never blocks.
    pub(crate) fn try_recv(&self) -> Option<Answer<Req, Res>> {
        self.answers.try_recv().ok()
    }
}

impl<Req, Res> Worker<Req, Res>
where
    Req: Job<Key = Req>,
    Res: Send + 'static,
{
    // Hand `pool` every finished load; a load that panicked is retired on the
    // pool, which never asks for it again.
    pub(crate) fn deliver<P: LoadPool<Request = Req, Result = Res>>(&self, pool: &mut P) {
        while let Some(answer) = self.try_recv() {
            match answer {
                Ok(result) => pool.mailbox().deliver(result),
                Err(request) => pool.load_failed(request),
            }
        }
    }

    // Send the requests `pool`'s drive queued; one no thread takes is
    // reverted on the pool.
    pub(crate) fn send_requests<P: LoadPool<Request = Req, Result = Res>>(&mut self, pool: &mut P) {
        while let Some(request) = pool.mailbox().pop_request() {
            if let Err(request) = self.send(request) {
                pool.request_refused(request);
            }
        }
    }
}

impl<Req: Job, Res> Drop for Worker<Req, Res> {
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
fn answer_each<Req: Job, Res>(
    requests: &Receiver<Req>,
    answers: &Sender<Answer<Req, Res>>,
    cancel: &AtomicBool,
    load: &dyn Fn(Req) -> Res,
) {
    while let Ok(request) = requests.recv() {
        if cancel.load(Ordering::Relaxed) {
            continue;
        }
        let key = request.key();
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
    use concinnity_core::render::stream_pools::Mailbox;
    use std::time::{Duration, Instant};

    // A pool that records the requests it had reverted or retired.
    #[derive(Default)]
    struct Pool {
        mail: Mailbox<usize, usize>,
        refused: Vec<usize>,
        failed: Vec<usize>,
    }

    impl LoadPool for Pool {
        type Request = usize;
        type Result = usize;

        fn mailbox(&mut self) -> &mut Mailbox<usize, usize> {
            &mut self.mail
        }

        fn request_refused(&mut self, request: usize) {
            self.refused.push(request);
        }

        fn load_failed(&mut self, request: usize) {
            self.failed.push(request);
        }
    }

    // Deliver until `done` holds or ten seconds pass.
    fn deliver_until(worker: &Worker<usize, usize>, pool: &mut Pool, done: impl Fn(&Pool) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !done(pool) {
            assert!(Instant::now() < deadline, "no answers came back");
            worker.deliver(pool);
            std::thread::yield_now();
        }
    }

    #[test]
    fn requests_round_trip_through_the_threads() {
        let mut worker = Worker::spawn("test-load", 2, |n: usize| n * 10).expect("threads");
        let mut pool = Pool::default();
        for n in 1..=3 {
            pool.mail.request(n);
        }
        worker.send_requests(&mut pool);
        let mut got = Vec::new();
        deadline_collect(&worker, &mut pool, &mut got, 3);
        got.sort_unstable();
        assert_eq!(got, [10, 20, 30]);
        assert!(pool.refused.is_empty());
    }

    fn deadline_collect(
        worker: &Worker<usize, usize>,
        pool: &mut Pool,
        got: &mut Vec<usize>,
        count: usize,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while got.len() < count {
            assert!(Instant::now() < deadline, "no answers came back");
            worker.deliver(pool);
            got.extend(pool.mail.take_results());
            std::thread::yield_now();
        }
    }

    // A load that panics leaves the thread answering: its request comes back
    // retired, not refused, and the next request still loads.
    #[test]
    fn a_panicking_load_is_retired_and_the_thread_survives() {
        let mut worker = Worker::spawn("test-panic", 1, |n: usize| {
            assert!(n != 7, "a load that cannot finish");
            n
        })
        .expect("thread");
        let mut pool = Pool::default();
        pool.mail.request(7);
        pool.mail.request(8);
        worker.send_requests(&mut pool);
        deliver_until(&worker, &mut pool, |p| !p.failed.is_empty());
        assert_eq!(pool.failed, [7]);
        assert!(pool.refused.is_empty());
        let mut got = Vec::new();
        deadline_collect(&worker, &mut pool, &mut got, 1);
        assert_eq!(got, [8]);
    }

    // A request no live thread takes comes back refused, for the pool to ask
    // for again, rather than retired.
    #[test]
    fn a_request_with_no_live_thread_is_refused() {
        let mut worker = Worker::spawn("test-refuse", 1, |n: usize| n).expect("thread");
        worker.threads[0].requests = None;
        let mut pool = Pool::default();
        pool.mail.request(3);
        worker.send_requests(&mut pool);
        assert_eq!(pool.refused, [3]);
        assert!(pool.failed.is_empty());
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
