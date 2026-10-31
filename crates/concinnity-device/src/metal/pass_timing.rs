//! Per-pass GPU timing on Metal via `MTLCounterSampleBuffer`. The whole-frame
//! timer in `MtlContext.diagnostics.gpu_time_us` only captures `GPUStartTime` /
//! `GPUEndTime`; this module supplements it with one start + end timestamp
//! per pass so the profiler overlay can attribute milliseconds to shadow /
//! main / SSAO / SSR / etc.
//!
//! Wiring. Each pass calls
//! [`PassTimingResources::attach_render`] (or `attach_compute`) on its
//! `MTLRenderPassDescriptor` / `MTLComputePassDescriptor` before creating
//! the encoder. The helper writes start- and end-of-encoder sample indices
//! into the descriptor's `sampleBufferAttachments[0]` and reserves a unique
//! pair of slots for that pass. Multi-encoder passes (the four shadow
//! cascades, the bloom mip chain) call `attach_render_first` on the first
//! encoder and `attach_render_last` on the last; intermediate encoders
//! don't write any timestamps and so don't contribute.
//!
//! Which stage boundary a render pass samples. Apple GPUs are tile-based
//! deferred: a render pass runs as a vertex/tiling phase and then a
//! fragment/rendering phase, and the tiling phase of one pass overlaps the
//! fragment phase of the passes before it. A pass's vertex phase therefore
//! starts near the top of the frame whenever nothing gates it, so
//! end-of-fragment minus start-of-vertex is time-to-end-of-pass, not the pass's
//! cost: the render passes then read monotonically through graph order and their
//! sum runs well past the frame's own GPU span. Render passes are bracketed
//! start-of-fragment to end-of-fragment instead, which is the pass's occupancy
//! of the fragment pipeline -- the phase that serializes across passes on one
//! queue, and where the work is on this hardware. Compute passes have one stage
//! and keep their encoder boundaries.
//!
//! The four stage boundaries are one capability, `AtStageBoundary`, which
//! [`PassTimingResources::new`] requires; a device without it reports no per-pass
//! timing rather than zeroes. Every boundary this module names reports a real
//! timestamp on Apple silicon (measured: vertex and fragment phases resolve to
//! distinct, ordered values), so there is no boundary to work around here.
//!
//! Summing. Fragment phases serialize on a queue in the common case, so the
//! graphics queue's per-pass sum is bounded by the frame's GPU span (which is
//! taken over the graphics queue alone, see [`frame_span`]) and falls
//! short of it by the vertex/tiling bubbles no fragment phase covers. Two things
//! break the bound rather than the measurement: the async-compute queue's passes
//! overlap the graphics ones, so the relation holds per queue and not overall,
//! and the GPU may run two independent render passes' fragment phases at once
//! (a depth-only shadow pass alongside a shading pass), which is a real overlap
//! both spans report honestly.
//!
//! Race avoidance. The sample buffer is per-frame: a ring of one buffer per
//! frame in flight is rotated each frame so the CPU-side resolve
//! of frame N-1's buffer never overlaps frame N's GPU writes. The completion
//! handler for frame N reads frame-N's buffer and publishes the results into
//! `MtlContext.pass_times_us_atomic[..]`; `render_stats()` then copies the
//! atomics into the `RenderStats.pass_times_us` array.
//!
//! Calibration. Apple Silicon's `MTLCommonCounterSetTimestamp` reports
//! the GPU clock in mach absolute time units, which on Apple Silicon
//! matches nanoseconds 1:1. We treat the raw u64 as nanoseconds and divide
//! by 1000 for microseconds. A proper `sampleTimestamps:gpuTimestamp:`
//! calibration would be needed for a future Intel Mac path.

use concinnity_core::math::SaturatingFrom;
pub(super) use concinnity_engine::render::render_graph::{
    PassId, TIMING_COUNT, TIMING_NAMES, TimingId,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLBlitPassDescriptor, MTLBuffer, MTLCommandBuffer,
    MTLCommandEncoder as _, MTLCommonCounterSetTimestamp, MTLComputePassDescriptor,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLOrigin, MTLRenderPassDescriptor,
    MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture,
};

// `NSUInteger::MAX` sentinel for an unused sample slot inside a render-pass
// or compute-pass `sampleBufferAttachments[0]`. Metal treats this as
// "don't sample at this stage".
const NO_SAMPLE: usize = usize::MAX;

// Every timing slot has a bit in the `attached` mask.
const _: () = assert!(TIMING_COUNT <= u64::BITS as usize);

// Sample slots per frame: one (start, end) pair per timing slot, in index order.
const SAMPLE_COUNT: usize = TIMING_COUNT * 2;

// A frame's sample buffer, moved into the completion work that resolves it.
//
// The resolve now runs from the frame's completion join rather than from the
// presenting command buffer's own handler, because the async-compute queue can
// still be writing timestamps when the composite retires. The join's completion
// work is a `Box<dyn FnOnce() + Send>`, so the buffer handle has to cross that
// bound.
pub(super) struct SendableSampleBuf(
    pub(super) Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
);

// SAFETY: the handle is created on the render thread, moved once into the
// frame's completion work, and read there by whichever thread makes the last
// arrival; no two threads hold it at once. `resolveCounterRange` is a read of a
// buffer the GPU has finished writing (every command buffer of the frame has
// retired by then), and Apple's Metal objects are safe for shared read.
unsafe impl Send for SendableSampleBuf {}

// One frame's worth of pass timestamps. The sample buffer lives on the GPU
// (private storage); `resolve` reads it back into CPU-visible bytes.
pub(super) struct PassTimingResources {
    // One per frame in flight, so frame N's resolve never overlaps a later
    // frame's GPU writes.
    buffers: Vec<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>>,
    // Rotates through `buffers` each frame. The active index picks which
    // buffer the next `attach_*` call binds to.
    frame_slot: usize,
    // Bitmask (1 << timing index) of the passes attached this frame, cleared by
    // `begin_frame` and set by each `attach_*`. The sample buffer is reused
    // across frames and never cleared, so a pass that ran a few frames ago but
    // is absent this frame leaves stale timestamps in its slot; the resolve
    // zeroes any pass whose bit is clear so the profiler reports 0 for a pass
    // that did not actually run (e.g. every world pass behind an opaque menu).
    attached: std::sync::atomic::AtomicU64,
    // Where a marker's one-pixel copy lands; never read.
    marker_sink: Retained<ProtocolObject<dyn MTLBuffer>>,
}

// Bytes a marker copies: one pixel of up to four 32-bit channels.
const MARKER_BYTES: usize = 16;

impl PassTimingResources {
    // Build per-frame timestamp sample buffers. Returns `None` if the
    // device does not expose the timestamp counter set (older Apple GPUs
    // or an Intel Mac without the right driver path).
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        frames_in_flight: usize,
    ) -> Option<Self> {
        // Every boundary the `attach_*` helpers name is a stage boundary, so a
        // device that cannot sample there reports no per-pass timing at all
        // rather than a table of zeroes.
        if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return None;
        }
        // Look up the timestamp counter set among the device's reported sets.
        let sets: Retained<NSArray<ProtocolObject<dyn MTLCounterSet>>> = device.counterSets()?;
        // SAFETY: `MTLCommonCounterSetTimestamp` is a framework-owned static NSString that outlives
        // this borrow.
        let target_name = unsafe { MTLCommonCounterSetTimestamp };
        let mut timestamp_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
        for set in &sets {
            let name: Retained<NSString> = set.name();
            if &*name == target_name {
                timestamp_set = Some(set);
                break;
            }
        }
        let timestamp_set = timestamp_set?;

        let make_buf = || -> Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>> {
            let desc = MTLCounterSampleBufferDescriptor::new();
            desc.setCounterSet(Some(&timestamp_set));
            // Shared storage so `resolveCounterRange` can copy the
            // CPU-visible result back without an explicit blit. Private
            // would be cheaper on the GPU side but is read-only from the
            // host on Apple Silicon and crashes the resolve.
            desc.setStorageMode(MTLStorageMode::Shared);
            // SAFETY: SAMPLE_COUNT is a small constant; the descriptor
            // accepts any non-zero sample count up to the device's max.
            unsafe { desc.setSampleCount(SAMPLE_COUNT) };
            device
                .newCounterSampleBufferWithDescriptor_error(&desc)
                .ok()
        };
        let buffers = (0..frames_in_flight.max(1))
            .map(|_| make_buf())
            .collect::<Option<Vec<_>>>()?;
        // Untracked, so frames in flight writing it never wait on each other.
        let marker_sink = device.newBufferWithLength_options(
            MARKER_BYTES,
            MTLResourceOptions::StorageModePrivate
                | MTLResourceOptions::HazardTrackingModeUntracked,
        )?;
        Some(Self {
            buffers,
            frame_slot: 0,
            attached: std::sync::atomic::AtomicU64::new(0),
            marker_sink,
        })
    }

    // Active sample buffer for the in-progress frame. The render/compute
    // pass descriptors attach to this; the completion handler resolves it
    // after the matching frame retires.
    fn active(&self) -> &ProtocolObject<dyn MTLCounterSampleBuffer> {
        &self.buffers[self.frame_slot]
    }

    // Move to the next sample buffer and return its slot: the buffer every
    // `attach_*` call binds to until the next `begin_frame`, and the one the
    // same frame's completion handler resolves.
    pub(super) fn begin_frame(&mut self) -> usize {
        self.frame_slot = (self.frame_slot + 1) % self.buffers.len();
        // Reset the per-frame attached mask; the passes that run this frame set
        // their bits via `attach_*`, and the resolve zeroes the rest.
        self.attached.store(0, std::sync::atomic::Ordering::Relaxed);
        self.frame_slot
    }

    // Bitmask of the passes attached since the last `begin_frame`. Captured
    // after the frame's passes are encoded and handed to the completion handler
    // so it can zero the stale slots of passes that did not run.
    pub(super) fn attached_mask(&self) -> u64 {
        self.attached.load(std::sync::atomic::Ordering::Relaxed)
    }

    // Record that `pass` was attached this frame (its slot holds fresh
    // timestamps).
    fn mark_attached(&self, pass: TimingId) {
        self.attached
            .fetch_or(1u64 << pass.index(), std::sync::atomic::Ordering::Relaxed);
    }

    // Buffer handle for an already-issued frame's completion handler.
    // `slot` is what [`begin_frame`] returned for that frame. Returns a
    // fresh `Retained` clone so the closure can outlive the borrow we
    // took on `self`.
    pub(super) fn buffer_for(
        &self,
        slot: usize,
    ) -> Retained<ProtocolObject<dyn MTLCounterSampleBuffer>> {
        self.buffers[slot].clone()
    }

    // Attach a single-encoder render pass to its start + end slot pair.
    pub(super) fn attach_render(&self, desc: &MTLRenderPassDescriptor, pass: impl Into<TimingId>) {
        let pass = pass.into();
        self.mark_attached(pass);
        let (start, end) = slot_pair(pass);
        self.sample_render(desc, start, end);
    }

    // Attach the FIRST encoder of a multi-encoder pass (e.g. shadow
    // cascade 0, bloom prefilter). Writes only the start-of-fragment sample;
    // the end is written by [`attach_render_last`], so the pair spans the
    // whole chain's fragment work including the intermediate encoders.
    pub(super) fn attach_render_first(
        &self,
        desc: &MTLRenderPassDescriptor,
        pass: impl Into<TimingId>,
    ) {
        let pass = pass.into();
        self.mark_attached(pass);
        let (start, _) = slot_pair(pass);
        self.sample_render(desc, start, NO_SAMPLE);
    }

    // Attach the LAST encoder of a multi-encoder pass. Writes only the
    // end-of-fragment sample; the start was written by
    // [`attach_render_first`].
    pub(super) fn attach_render_last(
        &self,
        desc: &MTLRenderPassDescriptor,
        pass: impl Into<TimingId>,
    ) {
        let pass = pass.into();
        let (_, end) = slot_pair(pass);
        self.sample_render(desc, NO_SAMPLE, end);
    }

    // Attach a single-encoder compute pass to its start + end slot pair.
    // Mirrors [`attach_render`] for `MTLComputePassDescriptor`.
    pub(super) fn attach_compute(
        &self,
        desc: &MTLComputePassDescriptor,
        pass: impl Into<TimingId>,
    ) {
        let pass = pass.into();
        self.mark_attached(pass);
        let (start, end) = slot_pair(pass);
        self.sample_compute(desc, start, end);
    }

    // Attach the FIRST encoder of a multi-encoder pass that opens with a
    // compute encoder. Mirrors [`attach_render_first`].
    pub(super) fn attach_compute_first(
        &self,
        desc: &MTLComputePassDescriptor,
        pass: impl Into<TimingId>,
    ) {
        let pass = pass.into();
        self.mark_attached(pass);
        let (start, _) = slot_pair(pass);
        self.sample_compute(desc, start, NO_SAMPLE);
    }

    // Attach the LAST encoder of a multi-encoder pass that closes with a
    // compute encoder. Mirrors [`attach_render_last`].
    pub(super) fn attach_compute_last(
        &self,
        desc: &MTLComputePassDescriptor,
        pass: impl Into<TimingId>,
    ) {
        let pass = pass.into();
        let (_, end) = slot_pair(pass);
        self.sample_compute(desc, NO_SAMPLE, end);
    }

    // Point `desc`'s first sample attachment at the active buffer, sampling the
    // fragment phase at `start` and `end`. The fragment phase is the pass's
    // occupancy; its vertex phase runs ahead of the passes before it and would
    // make the reading a time-to-end-of-pass. See the module header.
    fn sample_render(&self, desc: &MTLRenderPassDescriptor, start: usize, end: usize) {
        let arr = desc.sampleBufferAttachments();
        // SAFETY: index 0 is within the descriptor's fixed sample-attachment array.
        let entry = unsafe { arr.objectAtIndexedSubscript(0) };
        entry.setSampleBuffer(Some(self.active()));
        // SAFETY: `NO_SAMPLE` is Metal's don't-sample sentinel, valid for any buffer.
        unsafe { entry.setStartOfVertexSampleIndex(NO_SAMPLE) };
        // SAFETY: as above, the vertex phase is left unsampled.
        unsafe { entry.setEndOfVertexSampleIndex(NO_SAMPLE) };
        // SAFETY: `start` is a `slot_pair` index below `SAMPLE_COUNT`, the active
        // buffer's sample count, or `NO_SAMPLE`.
        unsafe { entry.setStartOfFragmentSampleIndex(start) };
        // SAFETY: `end` is likewise in the active buffer's range or `NO_SAMPLE`.
        unsafe { entry.setEndOfFragmentSampleIndex(end) };
    }

    fn sample_compute(&self, desc: &MTLComputePassDescriptor, start: usize, end: usize) {
        let arr = desc.sampleBufferAttachments();
        // SAFETY: index 0 is within the descriptor's fixed sample-attachment array.
        let entry = unsafe { arr.objectAtIndexedSubscript(0) };
        entry.setSampleBuffer(Some(self.active()));
        // SAFETY: `start` is a `slot_pair` index below `SAMPLE_COUNT`, the active
        // buffer's sample count, or `NO_SAMPLE`.
        unsafe { entry.setStartOfEncoderSampleIndex(start) };
        // SAFETY: `end` is likewise in the active buffer's range or `NO_SAMPLE`.
        unsafe { entry.setEndOfEncoderSampleIndex(end) };
    }

    // Attach a render encoder at `timer`'s place in its pass's span.
    pub(super) fn attach_render_timer(&self, desc: &MTLRenderPassDescriptor, timer: PassTimer) {
        match timer {
            PassTimer::None => {}
            PassTimer::Whole(id) => self.attach_render(desc, id),
            PassTimer::First(id) => self.attach_render_first(desc, id),
            PassTimer::Last(id) => self.attach_render_last(desc, id),
        }
    }

    // Attach a compute encoder at `timer`'s place in its pass's span.
    pub(super) fn attach_compute_timer(&self, desc: &MTLComputePassDescriptor, timer: PassTimer) {
        match timer {
            PassTimer::None => {}
            PassTimer::Whole(id) => self.attach_compute(desc, id),
            PassTimer::First(id) => self.attach_compute_first(desc, id),
            PassTimer::Last(id) => self.attach_compute_last(desc, id),
        }
    }

    // Record `timer`'s sample on a blit that copies one pixel of `after`, for
    // work encoded outside any encoder this module can attach to (MetalFX's
    // scaler). The read makes the hazard tracker hold the marker until every
    // earlier writer of `after` has finished, which places its sample against
    // that work; an encoder with nothing to do is dropped and samples nothing.
    pub(super) fn mark(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        timer: PassTimer,
        after: &ProtocolObject<dyn MTLTexture>,
    ) {
        let desc = MTLBlitPassDescriptor::new();
        let (start, end) = match timer {
            PassTimer::None => return,
            PassTimer::Whole(id) => {
                self.mark_attached(id);
                slot_pair(id)
            }
            PassTimer::First(id) => {
                self.mark_attached(id);
                (slot_pair(id).0, NO_SAMPLE)
            }
            PassTimer::Last(id) => (NO_SAMPLE, slot_pair(id).1),
        };
        let arr = desc.sampleBufferAttachments();
        // SAFETY: index 0 is within the descriptor's fixed sample-attachment array.
        let entry = unsafe { arr.objectAtIndexedSubscript(0) };
        entry.setSampleBuffer(Some(self.active()));
        // SAFETY: `start` is a `slot_pair` index below `SAMPLE_COUNT`, the active
        // buffer's sample count, or `NO_SAMPLE`.
        unsafe { entry.setStartOfEncoderSampleIndex(start) };
        // SAFETY: `end` is likewise in the active buffer's range or `NO_SAMPLE`.
        unsafe { entry.setEndOfEncoderSampleIndex(end) };
        let Some(enc) = cmd_buf.blitCommandEncoderWithDescriptor(&desc) else {
            return;
        };
        let origin = MTLOrigin { x: 0, y: 0, z: 0 };
        let size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        // SAFETY: a 1x1 region at the origin of slice 0, level 0 is inside any
        // texture, and one pixel of at most `MARKER_BYTES` fits the sink.
        unsafe {
            enc.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                after,
                0,
                0,
                origin,
                size,
                &self.marker_sink,
                0,
                MARKER_BYTES,
                MARKER_BYTES,
            );
        }
        enc.endEncoding();
    }
}

// Where an encoder sits within a pass's GPU-timing span. Most passes are a
// single encoder (`Whole`); a pass that spans several marks the start sample on
// its first encoder and the end sample on its last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassTimer {
    // Record no timing sample on this encoder.
    None,
    // The pass's only encoder: record both its start and end samples here.
    Whole(TimingId),
    // The first encoder of a multi-encoder pass: record the start sample.
    First(TimingId),
    // The last encoder of a multi-encoder pass: record the end sample.
    Last(TimingId),
}

impl PassTimer {
    // The timer for encoder `index` of `count` encoders timed as `pass`.
    pub(crate) fn span(pass: impl Into<TimingId>, index: usize, count: usize) -> Self {
        let pass = pass.into();
        match (index == 0, index + 1 == count) {
            (true, true) => Self::Whole(pass),
            (true, false) => Self::First(pass),
            (false, true) => Self::Last(pass),
            (false, false) => Self::None,
        }
    }
}

// (start_slot, end_slot) in the sample buffer for the given timing slot.
fn slot_pair(pass: impl Into<TimingId>) -> (usize, usize) {
    let base = pass.into().index() * 2;
    (base, base + 1)
}

// Read a frame's sample buffer back as raw timestamps, or `None` when the
// driver cannot resolve the range or hands back a short buffer. The two
// readers below share it so the buffer geometry is asserted in one place.
#[expect(
    clippy::cast_ptr_alignment,
    reason = "Metal resolves counter samples into 8-byte aligned storage"
)]
fn read_samples(
    buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>,
) -> Option<[u64; SAMPLE_COUNT]> {
    let range = NSRange::new(0, SAMPLE_COUNT);
    // SAFETY: `range` starts at 0 and the buffer was created with `SAMPLE_COUNT` sample slots, so
    // the range is in bounds; a driver that cannot resolve it returns None rather than faulting.
    let data = unsafe { buffer.resolveCounterRange(range) }?;
    let needed = size_of::<MTLCounterResultTimestamp>() * SAMPLE_COUNT;
    if data.len() < needed {
        return None;
    }
    // SAFETY: `data` is an immutable NSData this function owns, so its bytes
    // cannot be mutated while borrowed.
    let bytes = unsafe { data.as_bytes_unchecked() };
    // SAFETY: `bytes` is at least `needed` bytes long, and the underlying buffer is
    // `MTLCounterResultTimestamp` (one u64 per slot).
    let timestamps: &[MTLCounterResultTimestamp] = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr().cast::<MTLCounterResultTimestamp>(),
            SAMPLE_COUNT,
        )
    };
    let mut out = [0u64; SAMPLE_COUNT];
    for (slot, sample) in out.iter_mut().zip(timestamps) {
        *slot = sample.timestamp;
    }
    Some(out)
}

// Per-pass microseconds from a frame's raw samples: end minus start for every
// pass whose pair was written, zero for the rest. Pure arithmetic over the
// slot layout, so the sum-versus-span relation the module header states is
// testable without a device.
//
// Assumes the timestamp counter set reports nanoseconds, which is the
// case on Apple Silicon. A future calibration pass using
// `MTLDevice::sampleTimestamps` could lift this assumption if it ever
// proves wrong.
fn durations_us(samples: &[u64; SAMPLE_COUNT], attached: u64) -> [u32; TIMING_COUNT] {
    let mut out = [0u32; TIMING_COUNT];
    for (i, slot) in out.iter_mut().enumerate() {
        if !ran_this_frame(samples, attached, i) {
            continue;
        }
        let (start, end) = (samples[i * 2], samples[i * 2 + 1]);
        *slot = u32::saturating_from((end - start) / 1000);
    }
    out
}

// Did pass `i` write a usable pair into this frame's buffer? The buffer is
// reused across frames and never cleared, so a pass absent from this frame's
// graph still holds whatever it wrote on its last run: `attached` (the mask
// `attach_*` built while the frame was recorded) is what separates the two. A
// pass that is attached but left the GPU default is rejected on the timestamps
// themselves -- zero is never a real reading, and neither is a backwards pair.
fn ran_this_frame(samples: &[u64; SAMPLE_COUNT], attached: u64, i: usize) -> bool {
    if attached & (1u64 << i) == 0 {
        return false;
    }
    let (start, end) = (samples[i * 2], samples[i * 2 + 1]);
    start != 0 && end != 0 && end > start
}

// Earliest sampled start to latest sampled end, in microseconds, or `None`
// when no pass wrote a usable pair. Masked like [`durations_us`]: a pass that
// stopped running keeps its last timestamps in the buffer, and one of those
// anchoring `min_start` stretches the span across every frame since -- a
// setting toggled off mid-session read as a six-second frame.
#[cfg(test)]
fn span_us(samples: &[u64; SAMPLE_COUNT], attached: u64) -> Option<u32> {
    let (start, end) = span_bounds(samples, attached)?;
    Some(u32::saturating_from(
        ((end - start) / 1000).min(u64::from(u32::MAX)),
    ))
}

// The earliest sampled start and latest sampled end, in GPU nanoseconds, with
// the same masking as [`span_us`].
fn span_bounds(samples: &[u64; SAMPLE_COUNT], attached: u64) -> Option<(u64, u64)> {
    let mut min_start = u64::MAX;
    let mut max_end = 0u64;
    for i in 0..TIMING_COUNT {
        if !ran_this_frame(samples, attached, i) {
            continue;
        }
        min_start = min_start.min(samples[i * 2]);
        max_end = max_end.max(samples[i * 2 + 1]);
    }
    (max_end > min_start).then_some((min_start, max_end))
}

// The stretches between the graphics passes' sampled spans that an async-queue
// pass covers, in GPU nanoseconds: the graphics queue waiting on async work,
// or a render pass's unsampled vertex phase running beside it.
fn gap_beside_async(samples: &[u64; SAMPLE_COUNT], attached: u64, async_mask: u64) -> u64 {
    let mut graphics = [(0u64, 0u64); TIMING_COUNT];
    let mut asynchronous = [(0u64, 0u64); TIMING_COUNT];
    let (mut g, mut a) = (0, 0);
    for i in 0..TIMING_COUNT {
        if !ran_this_frame(samples, attached, i) {
            continue;
        }
        let span = (samples[i * 2], samples[i * 2 + 1]);
        if async_mask & (1u64 << i) == 0 {
            graphics[g] = span;
            g += 1;
        } else {
            asynchronous[a] = span;
            a += 1;
        }
    }
    let graphics = &mut graphics[..g];
    let asynchronous = &mut asynchronous[..a];
    graphics.sort_unstable();
    asynchronous.sort_unstable();
    let mut gap = 0;
    let mut covered = graphics.first().map_or(0, |s| s.1);
    for &(start, end) in graphics.iter().skip(1) {
        if start > covered {
            gap += covered_by(asynchronous, covered, start);
        }
        covered = covered.max(end);
    }
    gap
}

// How much of `from..to` the spans in `sorted` (ordered by start) cover, each
// instant counted once however many spans hold it.
fn covered_by(sorted: &[(u64, u64)], from: u64, to: u64) -> u64 {
    let mut total = 0;
    let mut reached = from;
    for &(start, end) in sorted {
        let (start, end) = (start.max(reached), end.min(to));
        if end > start {
            total += end - start;
            reached = end;
        }
    }
    total
}

// One frame's GPU timings, read from its sample buffer in one resolve.
pub(super) struct FrameTimings {
    // Microseconds per timing slot, zero for a slot that did not run.
    pub(super) per_slot: [u32; TIMING_COUNT],
    // The earliest start and latest end of the graphics-queue passes, in GPU
    // nanoseconds. A single command buffer's `GPUStartTime` / `GPUEndTime`
    // covers only its own slice of the frame, so the pass timestamps, which
    // share one clock, are what span it.
    pub(super) span: Option<(u64, u64)>,
    // The part of the graphics-queue gaps an async pass ran through, in nanoseconds.
    pub(super) gap_beside_async: u64,
}

// Resolve a frame's sample buffer. `attached` is
// [`PassTimingResources::attached_mask`] for the same frame, so a slot outside
// it reports zero rather than its last run's timestamps; the async-queue
// passes in `async_mask` overlap the graphics queue rather than extend it, so
// the span leaves them out.
pub(super) fn resolve_frame(
    buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>,
    attached: u64,
    async_mask: u64,
) -> Option<FrameTimings> {
    let samples = read_samples(buffer)?;
    Some(FrameTimings {
        per_slot: durations_us(&samples, attached),
        span: span_bounds(&samples, attached & !async_mask),
        gap_beside_async: gap_beside_async(&samples, attached, async_mask),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::render::render_graph::GpuSpan;

    #[test]
    fn a_span_opens_on_its_first_encoder_and_closes_on_its_last() {
        let p = PassId::PlanarReflection.timing();
        assert_eq!(PassTimer::span(p, 0, 1), PassTimer::Whole(p));
        let four: Vec<_> = (0..4).map(|i| PassTimer::span(p, i, 4)).collect();
        assert_eq!(
            four,
            [
                PassTimer::First(p),
                PassTimer::None,
                PassTimer::None,
                PassTimer::Last(p)
            ]
        );
    }

    // Write one pass's (start, end) nanosecond pair into a synthetic buffer.
    fn write(
        samples: &mut [u64; SAMPLE_COUNT],
        pass: impl Into<TimingId>,
        start_ns: u64,
        end_ns: u64,
    ) {
        let (s, e) = slot_pair(pass);
        samples[s] = start_ns;
        samples[e] = end_ns;
    }

    // The GPU clock the counter set reports, so a frame's timestamps sit far
    // from zero. Zero is the unwritten-slot sentinel and never a reading.
    const CLOCK: u64 = 1_700_000_000_000;

    // A frame whose graphics-queue fragment phases run back to back, with the
    // async-compute cull overlapping the front of it. Nanoseconds, matching the
    // shape a Bistro frame resolves to.
    fn serial_frame() -> [u64; SAMPLE_COUNT] {
        let mut s = [0u64; SAMPLE_COUNT];
        let mut at = |pass: TimingId, start: u64, end: u64| {
            write(&mut s, pass, CLOCK + start, CLOCK + end);
        };
        // Async-compute queue: overlaps the graphics passes below.
        at(PassId::Cull.timing(), 0, 30_000);
        // Graphics queue, one fragment phase after another.
        at(PassId::GBufferPrepass.timing(), 1_400_000, 1_800_000);
        at(GpuSpan::SsaoKernel.timing(), 1_800_000, 3_800_000);
        at(PassId::SsaoBlur.timing(), 3_800_000, 4_200_000);
        at(PassId::Shadow.timing(), 4_200_000, 5_600_000);
        // The gap to `Main` is its own vertex/tiling phase: real GPU time that
        // no fragment phase covers, so the sum falls short of the span.
        at(PassId::Main.timing(), 6_700_000, 11_600_000);
        at(PassId::Composite.timing(), 11_600_000, 13_800_000);
        s
    }

    // Every pass `serial_frame` writes, as an `attached_mask` would report it.
    fn serial_mask() -> u64 {
        [
            PassId::Cull.timing(),
            PassId::GBufferPrepass.timing(),
            GpuSpan::SsaoKernel.timing(),
            PassId::SsaoBlur.timing(),
            PassId::Shadow.timing(),
            PassId::Main.timing(),
            PassId::Composite.timing(),
        ]
        .iter()
        .fold(0u64, |m, p| m | 1u64 << p.index())
    }

    const GRAPHICS: [TimingId; 6] = [
        PassId::GBufferPrepass.timing(),
        GpuSpan::SsaoKernel.timing(),
        PassId::SsaoBlur.timing(),
        PassId::Shadow.timing(),
        PassId::Main.timing(),
        PassId::Composite.timing(),
    ];

    fn sum_us(durations: &[u32; TIMING_COUNT], passes: &[TimingId]) -> u64 {
        passes.iter().map(|p| u64::from(durations[p.index()])).sum()
    }

    #[test]
    fn a_pair_resolves_to_its_own_delta() {
        let d = durations_us(&serial_frame(), serial_mask());
        // Each pass reads its own fragment span, not the time to its end: the
        // defect this sampling replaced had `Main` reading 11.6 ms here.
        assert_eq!(d[PassId::Main as usize], 4_900);
        assert_eq!(d[PassId::Shadow as usize], 1_400);
        assert_eq!(d[PassId::Cull as usize], 30);
    }

    #[test]
    fn an_unwritten_pair_reports_zero() {
        let mut s = serial_frame();
        assert_eq!(durations_us(&s, serial_mask())[PassId::Fog as usize], 0);
        // A half-written pair (the end sample never landed) is zero too, even
        // with the pass in the attached set.
        let (start, _) = slot_pair(PassId::Fog);
        s[start] = CLOCK + 9_000_000;
        let mask = serial_mask() | 1u64 << (PassId::Fog as usize);
        assert_eq!(durations_us(&s, mask)[PassId::Fog as usize], 0);
    }

    #[test]
    fn the_graphics_queue_sum_is_bounded_by_the_span() {
        let samples = serial_frame();
        let d = durations_us(&samples, serial_mask());
        let graphics = serial_mask() & !(1u64 << (PassId::Cull as usize));
        let span = u64::from(span_us(&samples, graphics).expect("a written frame has a span"));
        let sum = sum_us(&d, &GRAPHICS);
        assert!(
            sum <= span,
            "graphics passes sum to {sum} us over a {span} us span"
        );
        // And close to it: the shortfall is the vertex/tiling bubble alone.
        assert!(
            sum * 10 >= span * 8,
            "sum {sum} us is not close to {span} us"
        );
    }

    #[test]
    fn async_work_is_left_out_of_the_graphics_span() {
        let mut samples = serial_frame();
        // An async pass committed with this frame runs while the previous one
        // still holds the graphics queue, a whole frame interval earlier.
        write(
            &mut samples,
            PassId::LightCull,
            CLOCK - 20_000_000,
            CLOCK - 19_000_000,
        );
        let mask = serial_mask() | 1u64 << (PassId::LightCull as usize);
        let async_mask = 1u64 << (PassId::Cull as usize) | 1u64 << (PassId::LightCull as usize);
        // Across both queues the frame reads more than twice its own span.
        assert_eq!(span_us(&samples, mask), Some(33_800));
        // The graphics queue alone starts at the prepass.
        assert_eq!(span_us(&samples, mask & !async_mask), Some(12_400));
    }

    #[test]
    fn a_frame_with_no_sample_has_no_span() {
        assert_eq!(span_us(&[0u64; SAMPLE_COUNT], u64::MAX), None);
        assert_eq!(
            durations_us(&[0u64; SAMPLE_COUNT], u64::MAX),
            [0; TIMING_COUNT]
        );
    }

    // The buffer is reused across frames and never cleared, so a pass switched
    // off mid-session keeps timestamps from whenever it last ran. Unmasked,
    // one of those anchors `min_start` and the frame reads as seconds long.
    #[test]
    fn a_pass_that_stopped_running_is_left_out_of_both_readings() {
        let mut s = serial_frame();
        // Nine seconds before this frame, the age a toggled-off pass reaches
        // after a few hundred frames.
        write(
            &mut s,
            PassId::Ssgi,
            CLOCK - 9_000_000_000,
            CLOCK - 8_998_000_000,
        );
        let mask = serial_mask();
        assert_eq!(durations_us(&s, mask)[PassId::Ssgi as usize], 0);
        assert_eq!(
            span_us(&s, mask),
            Some(13_800),
            "stale slot stretched the span"
        );
        // With the pass back in the attached set the same slots do count, so
        // the mask is what rejects it rather than the timestamps.
        let live = mask | 1u64 << (PassId::Ssgi as usize);
        assert_eq!(durations_us(&s, live)[PassId::Ssgi as usize], 2_000);
        assert!(span_us(&s, live).unwrap() > 9_000_000);
    }

    // A frame's passes have to write the buffer its completion handler reads.
    // When they did not, each resolve read the previous frame's passes beside
    // the next frame's first ones, and the span came out two frames long.
    #[test]
    fn a_frame_attaches_to_the_buffer_it_resolves() {
        let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
            return;
        };
        let Some(mut timing) = PassTimingResources::new(&device, 2) else {
            return;
        };
        for _ in 0..3 {
            let slot = timing.begin_frame();
            let resolved = timing.buffer_for(slot);
            assert!(
                std::ptr::eq(timing.active(), &raw const *resolved),
                "slot {slot}"
            );
        }
    }

    #[test]
    fn every_pass_owns_a_distinct_pair_inside_the_buffer() {
        let mut seen = std::collections::HashSet::new();
        for pass in TimingId::all() {
            let (s, e) = slot_pair(pass);
            assert!(seen.insert(s), "duplicate start slot for {pass:?}");
            assert!(seen.insert(e), "duplicate end slot for {pass:?}");
            assert!(e < SAMPLE_COUNT, "{pass:?} addresses past the buffer");
        }
    }

    #[test]
    fn graphics_gaps_count_only_where_async_work_covers_it() {
        let mut s = [0u64; SAMPLE_COUNT];
        let mut at = |pass: TimingId, start: u64, end: u64| {
            write(&mut s, pass, CLOCK + start, CLOCK + end);
        };
        // Graphics: 0..10, then 30..40 after a 20 us hole, then 45..50.
        at(PassId::Main.timing(), 0, 10_000);
        at(PassId::CloudComposite.timing(), 30_000, 40_000);
        at(PassId::Composite.timing(), 45_000, 50_000);
        // Async: covers 5..25 of the first hole, and the second hole not at all.
        at(PassId::Clouds.timing(), 5_000, 20_000);
        at(PassId::Atmosphere.timing(), 15_000, 25_000);
        let mask = [
            PassId::Main,
            PassId::CloudComposite,
            PassId::Composite,
            PassId::Clouds,
            PassId::Atmosphere,
        ]
        .iter()
        .fold(0u64, |m, p| m | 1u64 << p.timing().index());
        let async_mask =
            1u64 << PassId::Clouds.timing().index() | 1u64 << PassId::Atmosphere.timing().index();
        // 10..25 of the 10..30 hole is covered once, overlap counted once.
        assert_eq!(gap_beside_async(&s, mask, async_mask), 15_000);
        assert_eq!(gap_beside_async(&s, mask, 0), 0);
    }

    #[test]
    fn covered_by_merges_overlapping_spans() {
        let spans = [(0, 10), (5, 15), (20, 30)];
        assert_eq!(covered_by(&spans, 0, 40), 25);
        assert_eq!(covered_by(&spans, 12, 22), 5);
        assert_eq!(covered_by(&spans, 15, 20), 0);
        assert_eq!(covered_by(&[], 0, 10), 0);
    }
}
