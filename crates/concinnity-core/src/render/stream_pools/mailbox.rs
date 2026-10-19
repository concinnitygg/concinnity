//! The seam between a streaming pool and the host's load workers: requests the
//! pool's plan queued this frame, and results the workers have finished.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// A streaming pool whose loads the host runs: the mailbox it trades through,
/// and how it reverts a request that never reached a load.
pub trait LoadPool {
    /// What the pool asks to have loaded.
    type Request;
    /// What a finished load hands back.
    type Result;

    /// The loads requested and the loads finished.
    fn mailbox(&mut self) -> &mut Mailbox<Self::Request, Self::Result>;

    /// A request the host could not run: the pool forgets it so a later plan
    /// asks again rather than leaving it pending.
    fn request_refused(&mut self, request: Self::Request);

    /// A load that cannot succeed (it panicked): the pool retires the request
    /// until the world is rebuilt, so it is never asked for again.
    fn load_failed(&mut self, request: Self::Request);
}

/// One pool's outgoing load requests and incoming load results. The pool
/// writes requests during its drive and reads every delivered result; the host
/// delivers results before the drive and sends the requests after it. Both
/// queues keep their capacity across frames.
#[derive(Debug)]
pub struct Mailbox<Req, Res> {
    outbox: VecDeque<Req>,
    inbox: Vec<Res>,
}

impl<Req, Res> Default for Mailbox<Req, Res> {
    fn default() -> Self {
        Self {
            outbox: VecDeque::new(),
            inbox: Vec::new(),
        }
    }
}

impl<Req, Res> Mailbox<Req, Res> {
    /// Hand a finished load to the pool's next drive.
    pub fn deliver(&mut self, result: Res) {
        self.inbox.push(result);
    }

    // Every request queued since the last call, in order.
    #[cfg(test)]
    pub(super) fn take_requests(&mut self) -> impl Iterator<Item = Req> + '_ {
        self.outbox.drain(..)
    }

    /// The oldest request not yet sent.
    pub fn pop_request(&mut self) -> Option<Req> {
        self.outbox.pop_front()
    }

    // Whether requests wait to be sent.
    #[cfg(test)]
    pub(super) fn has_requests(&self) -> bool {
        !self.outbox.is_empty()
    }

    /// Queue a load request for the host to send.
    pub fn request(&mut self, request: Req) {
        self.outbox.push_back(request);
    }

    /// Every result delivered since the last call, in order.
    pub fn take_results(&mut self) -> impl Iterator<Item = Res> + '_ {
        self.inbox.drain(..)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn requests_and_results_pass_through_in_order_once() {
        let mut mail: Mailbox<u32, &str> = Mailbox::default();
        mail.request(3);
        mail.request(1);
        assert!(mail.has_requests());
        assert_eq!(mail.take_requests().collect::<Vec<_>>(), vec![3, 1]);
        assert!(!mail.has_requests());
        mail.request(5);
        mail.request(6);
        assert_eq!(mail.pop_request(), Some(5));
        assert_eq!(mail.pop_request(), Some(6));
        assert_eq!(mail.pop_request(), None);
        mail.deliver("a");
        mail.deliver("b");
        assert_eq!(mail.take_results().collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!(mail.take_results().count(), 0);
    }
}
