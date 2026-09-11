// src/metal/frame_pacing.rs
//
// Frames-in-flight CPU↔GPU pacing. Without it the render loop's only
// backpressure is `currentDrawable()` blocking, so the CPU can queue frames
// arbitrarily far ahead of the GPU: the per-frame transient buffers (object /
// draw-args / joint / bindless-texture / instance) pile up, and the per-frame
// autorelease pool is the only thing keeping VRAM from running away. A
// counting semaphore seeded to the frames-in-flight depth bounds that queue:
// `draw_frame` acquires a slot before encoding, and the frame command buffer's
// completion handler releases it once the GPU has retired the frame, so at most
// `depth` frames are ever in flight. This is the foundation that lets the
// per-frame buffers move from fresh-allocation to ring-buffered reuse.
//
// "The GPU has retired the frame" is a join over both command queues, not one
// command buffer's completion. The render graph submits its async-compute
// passes on a second queue (`metal/graph_queues.rs`), and the terminal one
// (`HizFinal`, which writes the pyramid the *next* frame's cull reads) is not
// an ancestor of the presenting composite pass, so it can still be running when
// the composite command buffer retires. Every per-frame ring the slot guards
// -- the transient buffers, the argument buffers, the pass-timing sample
// buffers -- is written from both queues, so releasing on the composite alone
// would hand a slot back while the async queue was still reading it.
// [`FrameJoin`] is that join: each participating command buffer registers a
// part before it is committed and arrives from its completion handler, and the
// last arrival runs the frame's completion work and releases the slot exactly
// once. The alternative -- having the composite wait on the async queue's
// terminal event before presenting -- would also be correct, but it puts the
// present behind `HizFinal` and so pays for the slot with latency.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use dispatch2::{DispatchRetained, DispatchSemaphore, DispatchTime};

// Counting semaphore bounding how many frames the CPU may queue ahead of the
// GPU. Seeded to the frames-in-flight depth at construction.
pub(super) struct FrameInFlight {
    semaphore: DispatchRetained<DispatchSemaphore>,
}

impl FrameInFlight {
    // `depth` is the maximum number of frames allowed in flight at once,
    // clamped to ≥1 so a `0` can never deadlock the very first acquire.
    pub(super) fn new(depth: usize) -> Self {
        Self {
            semaphore: DispatchSemaphore::new(depth.max(1) as isize),
        }
    }

    // Block until a frame slot is free, then return an RAII [`FrameSlot`]
    // holding it. Dropping the slot releases it synchronously: the balanced
    // path for a frame abandoned before it is recorded at all. The normal path
    // calls [`FrameSlot::into_join`] to hand the single release to the join over
    // the frame's GPU completion handlers instead.
    pub(super) fn acquire(&self) -> FrameSlot {
        let semaphore = self.semaphore.clone();
        // DISPATCH_TIME_FOREVER: the GPU always eventually retires an in-flight
        // frame and signals, so steady state cannot block here permanently.
        let _ = semaphore.wait(DispatchTime::FOREVER);
        FrameSlot {
            semaphore: Some(semaphore),
        }
    }

    // Non-destructive single-threaded probe: is at least one slot free right
    // now? Decrements then immediately re-signals so the count is unchanged,
    // leaving the semaphore balanced for disposal (libdispatch traps if a
    // semaphore is deallocated below its seed value). Test-only.
    #[cfg(test)]
    fn has_free_slot(&self) -> bool {
        // Low-level `wait` with a NOW timeout returns 0 if it decremented (a
        // slot was free) or non-zero on timeout (none free). Re-signal on
        // success so the probe leaves no net change.
        if self.semaphore.wait(DispatchTime::NOW) == 0 {
            self.semaphore.signal();
            true
        } else {
            false
        }
    }
}

// RAII holder for one acquired frame-in-flight slot. Releases the slot exactly
// once: on `Drop` for a frame abandoned before it is recorded, or (when
// [`Self::into_join`] is called) from the frame's completion join. The release
// is GPU-driven on the normal path so the semaphore paces the CPU against GPU
// *retirement* rather than against CPU encode completion.
pub(super) struct FrameSlot {
    semaphore: Option<DispatchRetained<DispatchSemaphore>>,
}

impl FrameSlot {
    // Take the slot's single release. The handle must be signaled exactly once;
    // after this the guard's `Drop` is a no-op, so the slot is never
    // double-released.
    fn into_gpu_release(mut self) -> DispatchRetained<DispatchSemaphore> {
        self.semaphore
            .take()
            .expect("FrameSlot::into_gpu_release called exactly once")
    }

    // Transfer the slot's single release to a [`FrameJoin`], and return the
    // submission token that keeps the join open while the frame is still being
    // recorded. `on_last` runs on whichever thread makes the final arrival,
    // immediately before the slot is released.
    pub(super) fn into_join(
        self,
        on_last: Box<dyn FnOnce() + Send>,
    ) -> (std::sync::Arc<FrameJoin>, SubmissionToken) {
        let join = std::sync::Arc::new(FrameJoin {
            semaphore: self.into_gpu_release(),
            // The submission token is the first part.
            remaining: AtomicUsize::new(1),
            on_last: Mutex::new(Some(on_last)),
        });
        (std::sync::Arc::clone(&join), SubmissionToken(join))
    }
}

impl Drop for FrameSlot {
    fn drop(&mut self) {
        if let Some(semaphore) = self.semaphore.take() {
            semaphore.signal();
        }
    }
}

// A frame's completion join across every command buffer that carries part of
// it. Holds the frame-in-flight slot's single release and hands it back once
// every registered part has arrived.
pub(super) struct FrameJoin {
    semaphore: DispatchRetained<DispatchSemaphore>,
    remaining: AtomicUsize,
    on_last: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl FrameJoin {
    // Register one more command buffer. Called on the recording thread strictly
    // before that buffer is committed, so the count can never reach zero while
    // the frame still has work to submit: the submission token holds a part
    // until recording is done.
    pub(super) fn add_part(&self) {
        self.remaining.fetch_add(1, Ordering::Relaxed);
    }

    // Report one part complete. The last arrival runs the completion work and
    // releases the frame slot.
    pub(super) fn arrive(&self) {
        if self.remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        if let Some(on_last) = self.on_last.lock().ok().and_then(|mut slot| slot.take()) {
            on_last();
        }
        self.semaphore.signal();
    }
}

// RAII holder for the join's submission part. Dropped once the frame has been
// recorded and committed, whether that path ended in success or an error, so a
// frame abandoned mid-record still converges on a single release.
pub(super) struct SubmissionToken(std::sync::Arc<FrameJoin>);

impl Drop for SubmissionToken {
    fn drop(&mut self) {
        self.0.arrive();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_consumes_seeded_slots() {
        let fif = FrameInFlight::new(2);
        let _a = fif.acquire();
        let _b = fif.acquire();
        // Both seeded slots are now held; a third acquire would block.
        assert!(!fif.has_free_slot());
        // `_a` / `_b` drop here, releasing both slots back to the seed count so
        // the semaphore disposes balanced.
    }

    #[test]
    fn drop_releases_on_abandon() {
        let fif = FrameInFlight::new(1);
        {
            let _slot = fif.acquire();
            assert!(!fif.has_free_slot(), "slot should be held while alive");
        }
        // The abandoned slot's Drop must have released it back.
        assert!(fif.has_free_slot(), "Drop did not release the slot");
    }

    #[test]
    fn gpu_handoff_releases_exactly_once() {
        let fif = FrameInFlight::new(1);
        let slot = fif.acquire();
        let sem = slot.into_gpu_release();
        // Handing the release to the GPU path must suppress the guard's Drop,
        // so no slot is free until the handler signals.
        assert!(
            !fif.has_free_slot(),
            "into_gpu_release double-released via Drop"
        );
        sem.signal();
        // Exactly one slot came back: take it, then confirm none remain (a
        // double-release would leave a second free slot here).
        let taken = fif.acquire();
        assert!(!fif.has_free_slot(), "slot was released more than once");
        drop(taken);
    }

    #[test]
    fn the_join_releases_only_after_every_part_arrives() {
        let fif = FrameInFlight::new(1);
        let ran = std::sync::Arc::new(AtomicUsize::new(0));
        let flag = std::sync::Arc::clone(&ran);
        let (join, token) = fif.acquire().into_join(Box::new(move || {
            flag.fetch_add(1, Ordering::Relaxed);
        }));
        join.add_part();
        join.add_part();
        drop(token);
        join.arrive();
        assert!(!fif.has_free_slot(), "released with a part outstanding");
        assert_eq!(ran.load(Ordering::Relaxed), 0);
        join.arrive();
        assert_eq!(ran.load(Ordering::Relaxed), 1, "completion work ran once");
        assert!(fif.has_free_slot(), "the last arrival did not release");
        // A second acquire must find nothing left: the release happened once.
        let taken = fif.acquire();
        assert!(!fif.has_free_slot(), "slot was released more than once");
        drop(taken);
    }

    #[test]
    fn an_abandoned_frame_releases_through_the_token_alone() {
        // The record path errored before any command buffer was committed, so
        // the submission token is the only part.
        let fif = FrameInFlight::new(1);
        let (_join, token) = fif.acquire().into_join(Box::new(|| {}));
        assert!(!fif.has_free_slot());
        drop(token);
        assert!(fif.has_free_slot(), "the token did not release the slot");
    }

    #[test]
    fn zero_depth_clamps_to_one() {
        let fif = FrameInFlight::new(0);
        assert!(fif.has_free_slot(), "depth 0 should clamp to 1 usable slot");
        let taken = fif.acquire();
        assert!(!fif.has_free_slot());
        drop(taken);
    }
}
