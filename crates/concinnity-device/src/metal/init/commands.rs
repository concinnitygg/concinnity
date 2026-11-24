//! Per-frame submission bookkeeping: the transient buffer rings and the pass
//! timing diagnostics.

use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex};

use concinnity_engine::profile::RenderStats;

use super::InitGpu;
use crate::metal::context::{Diagnostics, FrameRings};
use crate::metal::frame_rings::{JointRing, TransientRing};
use crate::metal::material_params::MaterialParamRing;
use crate::metal::pass_timing::PassTimingResources;
use crate::record_pack::RecordPacker;
use concinnity_engine::render::material_params::MaterialParamTable;
use concinnity_engine::render::record_pack::StaticSlots;

// Ring slots for the buffers a probe bake reads across frames: one past
// `frames_in_flight`, reserved for the bake since the frame never writes it.
pub(super) fn ring_slots(frames_in_flight: usize) -> usize {
    frames_in_flight.max(1) + 1
}

pub(super) fn build_rings(gpu: &InitGpu<'_>, material_params: &MaterialParamTable) -> FrameRings {
    let frames_in_flight = gpu.frames_in_flight;
    let slots = ring_slots(frames_in_flight);
    FrameRings {
        object: TransientRing::new(slots),
        draw_args: TransientRing::new(slots),
        instance_tail: StaticSlots::new(slots),
        packer: RecordPacker::default(),
        model_history: TransientRing::new(frames_in_flight),
        bindless_tex: TransientRing::new(frames_in_flight.max(1)),
        probe_records: TransientRing::new(slots),
        probe_tail: StaticSlots::new(slots),
        joint: JointRing::new(slots),
        material_params: MaterialParamRing::new(material_params),
    }
}

pub(super) fn build_diagnostics(gpu: &InitGpu<'_>) -> Diagnostics {
    // `None` when the device does not expose the timestamp counter set; the
    // per-pass GPU timer then stays at zero for every pass.
    let pass_timing = PassTimingResources::new(&gpu.hw.device, gpu.frames_in_flight);
    tracing::info!(
        "pass-timing: per-pass GPU sample buffers {}",
        if pass_timing.is_some() {
            "ready"
        } else {
            "unavailable (no MTLCommonCounterSetTimestamp)"
        }
    );
    Diagnostics {
        frame_stats: RenderStats::default(),
        gpu_time_us: Arc::new(AtomicU32::new(0)),
        device_error: Arc::new(Mutex::new(None)),
        pass_timing,
        pass_times_us: Arc::new(std::array::from_fn(|_| AtomicU32::new(0))),
        pass_async_mask: Arc::new(AtomicU64::new(0)),
        gpu_overlap_us: Arc::new(AtomicU32::new(0)),
        gpu_gap_beside_async_us: Arc::new(AtomicU32::new(0)),
        gpu_done: Arc::new(Mutex::new(None)),
        gpu_span_end_ns: Arc::new(AtomicU64::new(0)),
        draw_calls_accum: AtomicU32::new(0),
    }
}
