// The background fetch thread every streamer drives: a request channel plus the
// thread that drains it. Owning both together keeps the shutdown order in one
// place -- drop the sender to end the worker's `recv` loop, then join so a world
// rebuild never leaks the thread.

use std::sync::mpsc::Sender;
use std::thread::JoinHandle;

pub(crate) struct Worker<Req> {
    request_tx: Option<Sender<Req>>,
    handle: Option<JoinHandle<()>>,
}

impl<Req: Send + 'static> Worker<Req> {
    // Spawn the named thread running `work`, which should drain `requests`
    // until the channel closes.
    pub(crate) fn spawn(
        name: &str,
        requests: std::sync::mpsc::Receiver<Req>,
        request_tx: Sender<Req>,
        work: impl FnOnce(std::sync::mpsc::Receiver<Req>) + Send + 'static,
    ) -> Self {
        let handle = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || work(requests))
            .unwrap_or_else(|e| panic!("failed to spawn {name} worker: {e}"));
        Self {
            request_tx: Some(request_tx),
            handle: Some(handle),
        }
    }

    // Queue one request. `false` when the worker is gone, so the caller can
    // revert its bookkeeping and retry rather than waiting forever.
    pub(crate) fn send(&self, request: Req) -> bool {
        self.request_tx
            .as_ref()
            .is_some_and(|tx| tx.send(request).is_ok())
    }
}

impl<Req> Drop for Worker<Req> {
    fn drop(&mut self) {
        self.request_tx = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
