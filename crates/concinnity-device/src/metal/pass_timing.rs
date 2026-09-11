// src/metal/pass_timing.rs
//
// Per-pass GPU timing on Metal via `MTLCounterSampleBuffer`. The whole-frame
// timer in `MtlContext.diagnostics.gpu_time_us` only captures `GPUStartTime` /
// `GPUEndTime`; this module supplements it with one start + end timestamp
// per pass so the profiler overlay can attribute milliseconds to shadow /
// main / SSAO / SSR / etc.
//
// Wiring. Each pass calls
// [`PassTimingResources::attach_render`] (or `attach_compute`) on its
// `MTLRenderPassDescriptor` / `MTLComputePassDescriptor` before creating
// the encoder. The helper writes start- and end-of-encoder sample indices
// into the descriptor's `sampleBufferAttachments[0]` and reserves a unique
// pair of slots for that pass. Multi-encoder passes (the four shadow
// cascades, the bloom mip chain) call `attach_render_first` on the first
// encoder and `attach_render_last` on the last; intermediate encoders
// don't write any timestamps and so don't contribute.
//
// Which stage boundary a render pass samples. Apple GPUs are tile-based
// deferred: a render pass runs as a vertex/tiling phase and then a
// fragment/rendering phase, and the tiling phase of one pass overlaps the
// fragment phase of the passes before it. A pass's vertex phase therefore
// starts near the top of the frame whenever nothing gates it, so
// end-of-fragment minus start-of-vertex is time-to-end-of-pass, not the pass's
// cost: the render passes then read monotonically through graph order and their
// sum runs well past the frame's own GPU span. Render passes are bracketed
// start-of-fragment to end-of-fragment instead, which is the pass's occupancy
// of the fragment pipeline -- the phase that serializes across passes on one
// queue, and where the work is on this hardware. Compute passes have one stage
// and keep their encoder boundaries.
//
// The four stage boundaries are one capability, `AtStageBoundary`, which
// [`PassTimingResources::new`] requires; a device without it reports no per-pass
// timing rather than zeroes. Every boundary this module names reports a real
// timestamp on Apple silicon (measured: vertex and fragment phases resolve to
// distinct, ordered values), so there is no boundary to work around here.
//
// Summing. Fragment phases serialize on a queue in the common case, so the
// graphics queue's per-pass sum is bounded by the frame's GPU span and falls
// short of it by the vertex/tiling bubbles no fragment phase covers. Two things
// break the bound rather than the measurement: the async-compute queue's passes
// overlap the graphics ones, so the relation holds per queue and not overall,
// and the GPU may run two independent render passes' fragment phases at once
// (a depth-only shadow pass alongside a shading pass), which is a real overlap
// both spans report honestly.
//
// Race avoidance. The sample buffer is per-frame: a ring of
// `FRAMES_IN_FLIGHT` buffers is rotated each frame so the CPU-side resolve
// of frame N-1's buffer never overlaps frame N's GPU writes. The completion
// handler for frame N reads frame-N's buffer and publishes the results into
// `MtlContext.pass_times_us_atomic[..]`; `render_stats()` then copies the
// atomics into the `RenderStats.pass_times_us` array.
//
// Calibration. Apple Silicon's `MTLCommonCounterSetTimestamp` reports
// the GPU clock in mach absolute time units, which on Apple Silicon
// matches nanoseconds 1:1. We treat the raw u64 as nanoseconds and divide
// by 1000 for microseconds. A proper `sampleTimestamps:gpuTimestamp:`
// calibration would be needed for a future Intel Mac path.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSRange, NSString};
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLComputePassDescriptor, MTLCounterResultTimestamp,
    MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint,
    MTLCounterSet, MTLDevice, MTLRenderPassDescriptor, MTLStorageMode,
};
// `PassId` and `PASS_COUNT` live in the shared render-graph module so the
// per-pass GPU timer and the future graph executor key off the same enum.
// Adding a new pass = adding a new variant there + a new entry in
// [`PASS_NAMES`] + a bumped `PASS_COUNT`; nothing else needs to change for
// timing to flow. The passes.rs `every_pass_id_round_trips_to_its_name` test
// forces those edits at compile time, and `slot_pair` debug_asserts the index
// at runtime, so a missed registration cannot silently report zero GPU time.
pub(super) use concinnity_core::render::render_graph::{PASS_COUNT, PASS_NAMES, PassId};

// Frames in flight on Apple Silicon. The sample buffer ring is sized to
// this so frame N's resolve never overlaps frame N+1's GPU writes.
pub(super) const FRAMES_IN_FLIGHT: usize = 3;

// `NSUInteger::MAX` sentinel for an unused sample slot inside a render-pass
// or compute-pass `sampleBufferAttachments[0]`. Metal treats this as
// "don't sample at this stage".
const NO_SAMPLE: usize = usize::MAX;

// Sample slots per frame: one (start, end) pair per pass, in `PassId` order.
const SAMPLE_COUNT: usize = PASS_COUNT * 2;

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
    buffers: [Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>; FRAMES_IN_FLIGHT],
    // Rotates 0..FRAMES_IN_FLIGHT each frame. The active index picks which
    // buffer the next `attach_*` call binds to.
    frame_slot: usize,
    // Bitmask (1 << PassId index) of the passes attached this frame, cleared by
    // `begin_frame` and set by each `attach_*`. The sample buffer is reused
    // across frames and never cleared, so a pass that ran a few frames ago but
    // is absent this frame leaves stale timestamps in its slot; the resolve
    // zeroes any pass whose bit is clear so the profiler reports 0 for a pass
    // that did not actually run (e.g. every world pass behind an opaque menu).
    attached: std::sync::atomic::AtomicU64,
}

impl PassTimingResources {
    // Build per-frame timestamp sample buffers. Returns `None` if the
    // device does not expose the timestamp counter set (older Apple GPUs
    // or an Intel Mac without the right driver path).
    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
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
        for set in sets.iter() {
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
        let b0 = make_buf()?;
        let b1 = make_buf()?;
        let b2 = make_buf()?;
        Some(Self {
            buffers: [b0, b1, b2],
            frame_slot: 0,
            attached: std::sync::atomic::AtomicU64::new(0),
        })
    }

    // Active sample buffer for the in-progress frame. The render/compute
    // pass descriptors attach to this; the completion handler resolves it
    // after the matching frame retires.
    fn active(&self) -> &ProtocolObject<dyn MTLCounterSampleBuffer> {
        &self.buffers[self.frame_slot]
    }

    // Pick which sample buffer the next set of `attach_*` calls binds to.
    // Call at the top of each frame; the completion handler for the same
    // frame resolves the same buffer.
    pub(super) fn begin_frame(&mut self) -> usize {
        let slot = self.frame_slot;
        self.frame_slot = (self.frame_slot + 1) % FRAMES_IN_FLIGHT;
        // Reset the per-frame attached mask; the passes that run this frame set
        // their bits via `attach_*`, and the resolve zeroes the rest.
        self.attached.store(0, std::sync::atomic::Ordering::Relaxed);
        slot
    }

    // Bitmask of the passes attached since the last `begin_frame`. Captured
    // after the frame's passes are encoded and handed to the completion handler
    // so it can zero the stale slots of passes that did not run.
    pub(super) fn attached_mask(&self) -> u64 {
        self.attached.load(std::sync::atomic::Ordering::Relaxed)
    }

    // Record that `pass` was attached this frame (its slot holds fresh
    // timestamps). `pass as usize < PASS_COUNT <= 64`, so the shift is in range.
    fn mark_attached(&self, pass: PassId) {
        self.attached.fetch_or(
            1u64 << (pass as usize),
            std::sync::atomic::Ordering::Relaxed,
        );
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
    pub(super) fn attach_render(&self, desc: &MTLRenderPassDescriptor, pass: PassId) {
        self.mark_attached(pass);
        let (start, end) = slot_pair(pass);
        // SAFETY: All Metal sample-buffer accessors are marked unsafe by
        // objc2-metal because they take `*mut self`-equivalent ObjC arrays.
        // We hold a unique borrow of `desc` through the call, so no
        // aliasing exists.
        unsafe {
            let arr = desc.sampleBufferAttachments();
            let entry = arr.objectAtIndexedSubscript(0);
            entry.setSampleBuffer(Some(self.active()));
            // The fragment phase is the pass's occupancy; its vertex phase runs
            // ahead of the passes before it and would make the reading a
            // time-to-end-of-pass. See the module header.
            entry.setStartOfVertexSampleIndex(NO_SAMPLE);
            entry.setEndOfVertexSampleIndex(NO_SAMPLE);
            entry.setStartOfFragmentSampleIndex(start);
            entry.setEndOfFragmentSampleIndex(end);
        }
    }

    // Attach the FIRST encoder of a multi-encoder pass (e.g. shadow
    // cascade 0, bloom prefilter). Writes only the start-of-fragment sample;
    // the end is written by [`attach_render_last`], so the pair spans the
    // whole chain's fragment work including the intermediate encoders.
    pub(super) fn attach_render_first(&self, desc: &MTLRenderPassDescriptor, pass: PassId) {
        self.mark_attached(pass);
        let (start, _) = slot_pair(pass);
        // SAFETY: see `attach_render`.
        unsafe {
            let arr = desc.sampleBufferAttachments();
            let entry = arr.objectAtIndexedSubscript(0);
            entry.setSampleBuffer(Some(self.active()));
            entry.setStartOfVertexSampleIndex(NO_SAMPLE);
            entry.setEndOfVertexSampleIndex(NO_SAMPLE);
            entry.setStartOfFragmentSampleIndex(start);
            entry.setEndOfFragmentSampleIndex(NO_SAMPLE);
        }
    }

    // Attach the LAST encoder of a multi-encoder pass. Writes only the
    // end-of-fragment sample; the start was written by
    // [`attach_render_first`].
    pub(super) fn attach_render_last(&self, desc: &MTLRenderPassDescriptor, pass: PassId) {
        let (_, end) = slot_pair(pass);
        // SAFETY: see `attach_render`.
        unsafe {
            let arr = desc.sampleBufferAttachments();
            let entry = arr.objectAtIndexedSubscript(0);
            entry.setSampleBuffer(Some(self.active()));
            entry.setStartOfVertexSampleIndex(NO_SAMPLE);
            entry.setEndOfVertexSampleIndex(NO_SAMPLE);
            entry.setStartOfFragmentSampleIndex(NO_SAMPLE);
            entry.setEndOfFragmentSampleIndex(end);
        }
    }

    // Attach a single-encoder compute pass to its start + end slot pair.
    // Mirrors [`attach_render`] for `MTLComputePassDescriptor`.
    pub(super) fn attach_compute(&self, desc: &MTLComputePassDescriptor, pass: PassId) {
        self.mark_attached(pass);
        let (start, end) = slot_pair(pass);
        // SAFETY: see `attach_render`.
        unsafe {
            let arr = desc.sampleBufferAttachments();
            let entry = arr.objectAtIndexedSubscript(0);
            entry.setSampleBuffer(Some(self.active()));
            entry.setStartOfEncoderSampleIndex(start);
            entry.setEndOfEncoderSampleIndex(end);
        }
    }
}

// (start_slot, end_slot) in the sample buffer for the given pass. The buffer
// is sized to `PASS_COUNT * 2` slots, so a `PassId` whose index reaches
// PASS_COUNT would address past the end and silently report zero GPU time.
// That only happens if a new pass variant was added without bumping
// PASS_COUNT + registering its PASS_NAMES entry; the debug_assert names the
// miss in dev builds (the passes.rs `every_pass_id_round_trips_to_its_name`
// test is the compile-time guard).
fn slot_pair(pass: PassId) -> (usize, usize) {
    debug_assert!(
        (pass as usize) < PASS_COUNT,
        "PassId {pass:?} (index {}) >= PASS_COUNT {PASS_COUNT}: register it in PASS_NAMES \
         and bump PASS_COUNT",
        pass as usize,
    );
    let base = pass as usize * 2;
    (base, base + 1)
}

// Read a frame's sample buffer back as raw timestamps, or `None` when the
// driver cannot resolve the range or hands back a short buffer. The two
// readers below share it so the buffer geometry is asserted in one place.
fn read_samples(
    buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>,
) -> Option<[u64; SAMPLE_COUNT]> {
    let range = NSRange::new(0, SAMPLE_COUNT);
    // SAFETY: `range` starts at 0 and the buffer was created with `SAMPLE_COUNT` sample slots, so
    // the range is in bounds; a driver that cannot resolve it returns None rather than faulting.
    let data = unsafe { buffer.resolveCounterRange(range) }?;
    let needed = std::mem::size_of::<MTLCounterResultTimestamp>() * SAMPLE_COUNT;
    if data.len() < needed {
        return None;
    }
    // SAFETY: `data` is at least `needed` bytes long, and the underlying buffer is
    // `MTLCounterResultTimestamp` (one u64 per slot).
    let timestamps: &[MTLCounterResultTimestamp] = unsafe {
        std::slice::from_raw_parts(
            data.as_bytes_unchecked().as_ptr() as *const MTLCounterResultTimestamp,
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
fn durations_us(samples: &[u64; SAMPLE_COUNT], attached: u64) -> [u32; PASS_COUNT] {
    let mut out = [0u32; PASS_COUNT];
    for (i, slot) in out.iter_mut().enumerate() {
        if !ran_this_frame(samples, attached, i) {
            continue;
        }
        let (start, end) = (samples[i * 2], samples[i * 2 + 1]);
        *slot = ((end - start) / 1000).min(u32::MAX as u64) as u32;
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
fn span_us(samples: &[u64; SAMPLE_COUNT], attached: u64) -> Option<u32> {
    let mut min_start = u64::MAX;
    let mut max_end = 0u64;
    for i in 0..PASS_COUNT {
        if !ran_this_frame(samples, attached, i) {
            continue;
        }
        min_start = min_start.min(samples[i * 2]);
        max_end = max_end.max(samples[i * 2 + 1]);
    }
    if max_end <= min_start {
        return None;
    }
    Some(((max_end - min_start) / 1000).min(u32::MAX as u64) as u32)
}

// Resolve a frame's sample buffer into per-pass microsecond deltas, one entry
// per pass in `PassId` order. `attached` is [`PassTimingResources::attached_mask`]
// for the same frame; a pass outside it reports zero rather than its last run's
// timestamps.
pub(super) fn resolve(
    buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>,
    attached: u64,
) -> [u32; PASS_COUNT] {
    match read_samples(buffer) {
        Some(samples) => durations_us(&samples, attached),
        None => [0; PASS_COUNT],
    }
}

// Whole-frame GPU span in microseconds: the earliest sampled pass start to the
// latest sampled pass end across this frame's buffer. A single command buffer's
// `GPUStartTime` / `GPUEndTime` covers only its own slice of the frame (the
// frame is split across many command buffers), so summing or reading one buffer
// under-reports the true frame time. The pass timestamps all share one GPU
// clock, so min-start to max-end across them is the real GPU-busy span.
//
// The earliest start is a render pass's fragment phase or a compute encoder's
// start, so a frame whose leading work is a render pass's vertex/tiling phase
// would begin its span slightly late. Every graph leads with the compute cull,
// which anchors the span at the frame's first encoder.
//
// Returns `None` when no pass wrote a valid timestamp pair.
pub(super) fn frame_span_us(
    buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>,
    attached: u64,
) -> Option<u32> {
    span_us(&read_samples(buffer)?, attached)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Write one pass's (start, end) nanosecond pair into a synthetic buffer.
    fn write(samples: &mut [u64; SAMPLE_COUNT], pass: PassId, start_ns: u64, end_ns: u64) {
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
        let mut at = |pass, start: u64, end: u64| write(&mut s, pass, CLOCK + start, CLOCK + end);
        // Async-compute queue: overlaps the graphics passes below.
        at(PassId::Cull, 0, 30_000);
        // Graphics queue, one fragment phase after another.
        at(PassId::GBufferPrepass, 1_400_000, 1_800_000);
        at(PassId::SsaoKernel, 1_800_000, 3_800_000);
        at(PassId::SsaoBlur, 3_800_000, 4_200_000);
        at(PassId::Shadow, 4_200_000, 5_600_000);
        // The gap to `Main` is its own vertex/tiling phase: real GPU time that
        // no fragment phase covers, so the sum falls short of the span.
        at(PassId::Main, 6_700_000, 11_600_000);
        at(PassId::Composite, 11_600_000, 13_800_000);
        s
    }

    // Every pass `serial_frame` writes, as an `attached_mask` would report it.
    fn serial_mask() -> u64 {
        [
            PassId::Cull,
            PassId::GBufferPrepass,
            PassId::SsaoKernel,
            PassId::SsaoBlur,
            PassId::Shadow,
            PassId::Main,
            PassId::Composite,
        ]
        .iter()
        .fold(0u64, |m, p| m | 1u64 << (*p as usize))
    }

    const GRAPHICS: [PassId; 6] = [
        PassId::GBufferPrepass,
        PassId::SsaoKernel,
        PassId::SsaoBlur,
        PassId::Shadow,
        PassId::Main,
        PassId::Composite,
    ];

    fn sum_us(durations: &[u32; PASS_COUNT], passes: &[PassId]) -> u64 {
        passes.iter().map(|p| durations[*p as usize] as u64).sum()
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
        let span = span_us(&samples, serial_mask()).expect("a written frame has a span") as u64;
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
    fn the_span_covers_both_queues() {
        let samples = serial_frame();
        // Cull runs on the async queue from 0, before any graphics pass, so the
        // span starts there -- which is why the sum relation is per queue.
        assert_eq!(span_us(&samples, serial_mask()), Some(13_800));
    }

    #[test]
    fn a_frame_with_no_sample_has_no_span() {
        assert_eq!(span_us(&[0u64; SAMPLE_COUNT], u64::MAX), None);
        assert_eq!(
            durations_us(&[0u64; SAMPLE_COUNT], u64::MAX),
            [0; PASS_COUNT]
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

    #[test]
    fn every_pass_owns_a_distinct_pair_inside_the_buffer() {
        let mut seen = std::collections::HashSet::new();
        for pass in PassId::ALL {
            let (s, e) = slot_pair(pass);
            assert!(seen.insert(s), "duplicate start slot for {pass:?}");
            assert!(seen.insert(e), "duplicate end slot for {pass:?}");
            assert!(e < SAMPLE_COUNT, "{pass:?} addresses past the buffer");
        }
    }
}
