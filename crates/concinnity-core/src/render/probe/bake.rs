//! The sequencing of a renderer's staggered reflection-probe bake, shared by
//! every backend: which probe starts, which cube face renders and which mip
//! convolves this frame, when a capture hands over to its convolution, when a
//! finished cube installs, and when the rest of the queue is abandoned. The
//! backend records the GPU work through [`ProbeBakeDevice`] and owns whatever
//! each step creates.
//!
//! Two slots run side by side: one probe renders its cube faces, one per
//! frame, while the previous probe's capture convolves into its cube, one
//! destination mip per frame. Each slot holds at most one probe, so installs
//! land in placement order and the [`ProbeBook`]'s records stay aligned with
//! the cube array.

use alloc::vec::Vec;

use crate::gfx::cubemap::FACE_BASIS;
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::probe::book::{ProbeBook, ProbeProgress};
use crate::render::probe::capture::{PrefilterPlan, ProbePlacement};

/// Cube faces a capture renders, one per frame.
pub const CAPTURE_FACES: usize = FACE_BASIS.len();

/// The transient-ring slot a capture builds its CPU-written buffers into: one
/// past the `frames_in_flight` slots the frame cycles through, so no frame
/// overwrites them while the capture is in flight. A ring the capture uses
/// holds `frames_in_flight + 1` slots.
pub const fn capture_ring_slot(frames_in_flight: usize) -> usize {
    frames_in_flight
}

/// The GPU half of a probe bake: one renderer's recording and submission of
/// each step [`ProbeBake`] schedules, and its answers about what the GPU has
/// finished.
pub trait ProbeBakeDevice {
    /// The resources of a capture in flight.
    type Capture;
    /// The resources of a convolution in flight.
    type Prefilter;
    /// What the frame hands a capture that starts or renders a face in it,
    /// borrowed for the frame's advance.
    type Frame<'f>;

    /// The probe bookkeeping the bake advances.
    fn book(&mut self) -> &mut ProbeBook;

    /// Whether this renderer can capture probes at all. A `false` answer
    /// abandons the queue, keeping the probes already installed.
    fn capture_supported(&self) -> bool;

    /// Whether a capture can start this frame, given whether a convolution is
    /// still in flight. A `false` answer keeps the queue for a later frame.
    fn capture_ready(&self, prefilter_in_flight: bool) -> bool;

    /// Make room in the cube array for `count` probes.
    fn reserve_cubes(&mut self, count: usize) -> RenderResult<()>;

    /// Build the capture of the probe at `index`, submitting no face yet.
    fn start_capture(
        &mut self,
        frame: &Self::Frame<'_>,
        index: usize,
        placement: ProbePlacement,
    ) -> RenderResult<Self::Capture>;

    /// Record and submit cube face `face` of `capture`.
    fn render_face(
        &mut self,
        frame: &Self::Frame<'_>,
        capture: &mut Self::Capture,
        face: usize,
    ) -> RenderResult<()>;

    /// Whether the GPU has retired every face submitted for `capture`.
    fn capture_retired(&self, capture: &Self::Capture) -> bool;

    /// Turn a retired capture of the probe at `index` into its convolution,
    /// releasing the capture's draw resources. Submits nothing.
    fn begin_prefilter(
        &mut self,
        index: usize,
        capture: Self::Capture,
    ) -> RenderResult<Self::Prefilter>;

    /// Record and submit convolution step `mip`: mip 0 is the clamped copy of
    /// the capture plus its source pyramid, each later mip one GGX convolution.
    fn prefilter_mip(&mut self, prefilter: &mut Self::Prefilter, mip: u32) -> RenderResult<()>;

    /// Whether every convolution step is done with as far as installing the
    /// cube is concerned: its writes are ordered before the frames that sample
    /// it, and nothing the steps recorded is still in use.
    fn prefilter_retired(&self, prefilter: &Self::Prefilter) -> bool;

    /// Release a convolution whose cube is about to install.
    fn finish_prefilter(&mut self, prefilter: Self::Prefilter);

    /// Release bake resources the GPU may still be reading.
    fn abandon(&mut self, capture: Option<Self::Capture>, prefilter: Option<Self::Prefilter>);
}

/// What one advance of the bake did that the renderer reports.
#[derive(Debug, Default)]
#[must_use]
pub struct BakeReport {
    /// The progress after a probe installed.
    pub installed: Option<ProbeProgress>,
    /// The error that abandoned the rest of the queue.
    pub failed: Option<BakeFailure>,
}

/// A step that failed, abandoning every probe not yet installed.
#[derive(Debug)]
pub struct BakeFailure {
    /// What failed.
    pub error: RenderError,
    /// Probes that stay installed.
    pub kept: usize,
}

// One probe in a slot: its placement index, the next face or mip to submit,
// and the backend's resources for it.
struct Slot<T> {
    index: usize,
    next: u32,
    gpu: T,
}

/// The two bake slots of a renderer, generic over the backend's resources for
/// a capture (`C`) and a convolution (`P`).
pub struct ProbeBake<C, P> {
    capture: Option<Slot<C>>,
    prefilter: Option<Slot<P>>,
}

impl<C, P> Default for ProbeBake<C, P> {
    fn default() -> Self {
        Self {
            capture: None,
            prefilter: None,
        }
    }
}

impl<C, P> ProbeBake<C, P> {
    /// The capture in flight.
    pub fn capture(&self) -> Option<&C> {
        self.capture.as_ref().map(|s| &s.gpu)
    }

    /// The convolution in flight.
    pub fn prefilter(&self) -> Option<&P> {
        self.prefilter.as_ref().map(|s| &s.gpu)
    }

    /// Hand both slots' resources back, leaving nothing in flight.
    pub fn take(&mut self) -> (Option<C>, Option<P>) {
        (
            self.capture.take().map(|s| s.gpu),
            self.prefilter.take().map(|s| s.gpu),
        )
    }

    /// Replace the placements: abandon whatever is in flight, make room for
    /// every placement and queue them all. When the cube array cannot grow,
    /// no probe is placed and reflections read the sky.
    pub fn place<D>(&mut self, device: &mut D, placements: Vec<ProbePlacement>) -> RenderResult<()>
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        self.abandon(device);
        let reserved = device.reserve_cubes(placements.len());
        let placements = if reserved.is_ok() {
            placements
        } else {
            Vec::new()
        };
        device.book().reset(placements);
        reserved
    }

    /// Advance both slots one frame: the convolution first, so a cube that
    /// installs frees its slot for the capture that follows it.
    pub fn advance<D>(&mut self, device: &mut D, frame: &D::Frame<'_>) -> BakeReport
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        let prefilter = self.advance_prefilter(device);
        let capture = self.advance_capture(device, frame);
        BakeReport {
            installed: prefilter.installed.or(capture.installed),
            failed: prefilter.failed.or(capture.failed),
        }
    }

    /// Advance the convolution slot one frame: convolve the next mip, or
    /// install a cube whose convolution has retired.
    pub fn advance_prefilter<D>(&mut self, device: &mut D) -> BakeReport
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        if !self.admit(device) {
            return BakeReport::default();
        }
        let (phase, signals) = match &self.prefilter {
            Some(slot) => {
                let more_mips = slot.next < PrefilterPlan::RUNTIME.mips();
                let signals = BakeSignals {
                    more_mips,
                    mips_done: !more_mips && device.prefilter_retired(&slot.gpu),
                    ..BakeSignals::default()
                };
                (BakePhase::Prefiltering, signals)
            }
            None => (BakePhase::Idle, BakeSignals::default()),
        };
        match next_bake_action(phase, signals) {
            BakeAction::PrefilterMip => self.prefilter_next_mip(device),
            BakeAction::Install => self.install(device),
            _ => BakeReport::default(),
        }
    }

    /// Advance the capture slot one frame: render the next face, hand a
    /// retired capture to a free convolution slot, or start the next queued
    /// probe.
    pub fn advance_capture<D>(&mut self, device: &mut D, frame: &D::Frame<'_>) -> BakeReport
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        if !self.admit(device) {
            return BakeReport::default();
        }
        let prefilter_in_flight = self.prefilter.is_some();
        let (phase, signals) = if let Some(slot) = &self.capture {
            let more_faces = (slot.next as usize) < CAPTURE_FACES;
            let signals = BakeSignals {
                more_faces,
                faces_done: !more_faces
                    && !prefilter_in_flight
                    && device.capture_retired(&slot.gpu),
                ..BakeSignals::default()
            };
            (BakePhase::Rendering, signals)
        } else {
            let signals = BakeSignals {
                queue_pending: device.book().pending(),
                eligible: device.capture_ready(prefilter_in_flight),
                ..BakeSignals::default()
            };
            (BakePhase::Idle, signals)
        };
        match next_bake_action(phase, signals) {
            BakeAction::RenderFace => self.render_next_face(device, frame),
            BakeAction::StartPrefilter => self.begin_prefilter(device),
            BakeAction::StartNext => self.start_next(device, frame),
            _ => BakeReport::default(),
        }
    }

    /// Release both slots' resources through the device.
    pub fn abandon<D>(&mut self, device: &mut D)
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        if matches!((&self.capture, &self.prefilter), (None, None)) {
            return;
        }
        let (capture, prefilter) = self.take();
        device.abandon(capture, prefilter);
    }

    // Whether there is anything to advance. A renderer that cannot capture
    // abandons its queue here rather than re-checking every frame.
    fn admit<D>(&mut self, device: &mut D) -> bool
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        let idle = self.capture.is_none() && self.prefilter.is_none();
        if idle && !device.book().pending() {
            return false;
        }
        if !device.capture_supported() {
            self.abandon(device);
            device.book().abort();
            return false;
        }
        true
    }

    fn prefilter_next_mip<D>(&mut self, device: &mut D) -> BakeReport
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        let Some(slot) = self.prefilter.as_mut() else {
            return BakeReport::default();
        };
        match device.prefilter_mip(&mut slot.gpu, slot.next) {
            Ok(()) => {
                slot.next += 1;
                BakeReport::default()
            }
            Err(e) => self.fail(device, e),
        }
    }

    fn install<D>(&mut self, device: &mut D) -> BakeReport
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        let Some(slot) = self.prefilter.take() else {
            return BakeReport::default();
        };
        device.finish_prefilter(slot.gpu);
        match device.book().install(slot.index) {
            Ok(progress) => BakeReport {
                installed: Some(progress),
                failed: None,
            },
            Err(e) => self.fail(device, e),
        }
    }

    fn render_next_face<D>(&mut self, device: &mut D, frame: &D::Frame<'_>) -> BakeReport
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        let Some(slot) = self.capture.as_mut() else {
            return BakeReport::default();
        };
        match device.render_face(frame, &mut slot.gpu, slot.next as usize) {
            Ok(()) => {
                slot.next += 1;
                BakeReport::default()
            }
            Err(e) => self.fail(device, e),
        }
    }

    // Move a retired capture into the convolution slot and submit its first
    // step in the same frame.
    fn begin_prefilter<D>(&mut self, device: &mut D) -> BakeReport
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        let Some(slot) = self.capture.take() else {
            return BakeReport::default();
        };
        match device.begin_prefilter(slot.index, slot.gpu) {
            Ok(gpu) => {
                self.prefilter = Some(Slot {
                    index: slot.index,
                    next: 0,
                    gpu,
                });
                self.prefilter_next_mip(device)
            }
            Err(e) => self.fail(device, e),
        }
    }

    fn start_next<D>(&mut self, device: &mut D, frame: &D::Frame<'_>) -> BakeReport
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        let Some((index, placement)) = device.book().take_next() else {
            return BakeReport::default();
        };
        match device.start_capture(frame, index, placement) {
            Ok(gpu) => {
                self.capture = Some(Slot {
                    index,
                    next: 0,
                    gpu,
                });
                BakeReport::default()
            }
            Err(e) => self.fail(device, e),
        }
    }

    // Abandon both slots and the rest of the queue, keeping what installed.
    // The queue advanced when the failed probe started, so aborting it is what
    // keeps the book's records aligned with the placements.
    fn fail<D>(&mut self, device: &mut D, error: RenderError) -> BakeReport
    where
        D: ProbeBakeDevice<Capture = C, Prefilter = P>,
    {
        self.abandon(device);
        let book = device.book();
        book.abort();
        BakeReport {
            installed: None,
            failed: Some(BakeFailure {
                error,
                kept: book.count(),
            }),
        }
    }
}

/// Tracks how far a staggered probe bake has progressed. Baking every probe on
/// one frame stalls proportionally to the probe count; instead the renderer bakes
/// a bounded budget per frame and walks this cursor, so the load cost is spread
/// and unbaked probes fall back to the sky until their turn. Indices are handed
/// out in order so the baked cube array stays aligned with the placement list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeBakeQueue {
    total: usize,
    next: usize,
}

impl ProbeBakeQueue {
    /// A queue seeded with `total` probes waiting to bake.
    pub fn new(total: usize) -> Self {
        Self { total, next: 0 }
    }

    /// Whether any placement is still waiting to bake.
    pub fn pending(&self) -> bool {
        self.next < self.total
    }

    /// The next placement index to bake, advancing the cursor. `None` when done.
    pub fn take_next(&mut self) -> Option<usize> {
        (self.next < self.total).then(|| {
            let i = self.next;
            self.next += 1;
            i
        })
    }

    /// Abandon every remaining placement (e.g. a permanently ineligible world or
    /// an unrecoverable bake error). Leaves the queue not `pending`.
    pub fn abort(&mut self) {
        self.next = self.total;
    }
}

/// Phase of the single in-flight asynchronous bake. Exactly one probe is baked at
/// a time, walking three phases across frames so the render thread never blocks on
/// the capture: `Rendering` (six faces submitted, GPU running), `Prefiltering` (the
/// GGX convolution dispatched one destination mip per frame), and `Idle` (nothing
/// in flight). The renderer holds the GPU resources for the in-flight probe; this
/// enum only names which phase it is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BakePhase {
    /// Nothing to bake.
    Idle,
    /// Rendering the probe's cube faces.
    Rendering,
    /// Convolving the captured faces into the probe cube's mip chain.
    Prefiltering,
}

/// What one bake slot should do this frame.
/// [`ProbeBake`] maps each variant to a
/// step of the backend's device: `StartNext` builds the next
/// placement's capture buffers + targets (no face submitted yet), `RenderFace`
/// submits one cube face (the six are spread one-per-frame so no single frame
/// pays the whole capture), `StartPrefilter` releases the capture's draw
/// resources and builds the probe cube plus the capture pyramid the convolution
/// reads, `PrefilterMip` dispatches one destination mip's convolution, `Install`
/// publishes the finished cube and advances the queue, `Idle` does nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BakeAction {
    /// Nothing to do this frame.
    Idle,
    /// Begin the next queued probe.
    StartNext,
    /// Render one cube face.
    RenderFace,
    /// Move a finished capture into the convolution, building its pyramid.
    StartPrefilter,
    /// Convolve one destination mip of the probe cube.
    PrefilterMip,
    /// Install the convolved cube into the probe pool.
    Install,
}

/// What is known this frame about the bake in flight, as the pure
/// transition table below reads it. Defaulted so a call site naming one slot's
/// signals leaves the other slot's at "nothing happening" rather than spelling
/// out a row of `false`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BakeSignals {
    /// Every cube face has been submitted and the GPU has signaled completion.
    pub faces_done: bool,
    /// Every convolution dispatch has retired on the GPU. A backend whose queue
    /// ordering already puts those writes before any frame that samples the cube,
    /// and which has no per-dispatch recording resources to reclaim, has nothing
    /// to wait for and reports this the moment the last mip is dispatched.
    pub mips_done: bool,
    /// A placement is still waiting to bake.
    pub queue_pending: bool,
    /// A capture can start this frame.
    pub eligible: bool,
    /// Cube faces remain to submit.
    pub more_faces: bool,
    /// Destination mips remain to convolve.
    pub more_mips: bool,
}

/// Decide the next action for the single-in-flight asynchronous bake. Pure so the
/// transition table is locked by unit tests without a GPU:
///
/// - while `Rendering`, submit the next cube face while `more_faces` remain
///   (one per frame); once all six are submitted, do nothing until the GPU
///   completion flag `faces_done` is set, then `StartPrefilter`;
/// - while `Prefiltering`, convolve one destination mip per frame while
///   `more_mips` remain, then wait for `mips_done` and `Install`;
/// - while `Idle`, start the next placement only when one is `queue_pending`
///   and the world is `eligible` to bake this frame; otherwise stay `Idle`.
///
/// The invariants this guarantees: never convolve before all six faces are
/// submitted and the GPU signals completion, never install before every mip has
/// been dispatched and retired, and never begin a second bake while one is
/// already in flight.
pub fn next_bake_action(phase: BakePhase, signals: BakeSignals) -> BakeAction {
    match phase {
        BakePhase::Rendering => {
            if signals.more_faces {
                BakeAction::RenderFace
            } else if signals.faces_done {
                BakeAction::StartPrefilter
            } else {
                BakeAction::Idle
            }
        }
        BakePhase::Prefiltering => {
            if signals.more_mips {
                BakeAction::PrefilterMip
            } else if signals.mips_done {
                BakeAction::Install
            } else {
                BakeAction::Idle
            }
        }
        BakePhase::Idle => {
            if signals.queue_pending && signals.eligible {
                BakeAction::StartNext
            } else {
                BakeAction::Idle
            }
        }
    }
}

#[cfg(test)]
mod tests;
