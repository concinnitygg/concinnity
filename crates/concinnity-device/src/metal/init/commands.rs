//! Per-frame submission bookkeeping: the transient buffer rings and the pass
//! timing diagnostics.

use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, Mutex};

use concinnity_core::gfx::profile::RenderStats;

use super::InitGpu;
use crate::metal::context::{Diagnostics, FrameRings};
use crate::metal::pass_timing::PassTimingResources;
use crate::metal::transient::{JointRing, TransientRing};

// The bindless buffers an async reflection-probe bake reads (object,
// draw-args, bindless-texture-args, and the skinned joint palettes) get
// one EXTRA ring slot. The frame only ever uses slots
// `frame_ring_index % frames_in_flight` -- i.e. `[0, frames_in_flight)`
// -- so slot `frames_in_flight` is reserved for the bake: a slot the
// frame never overwrites, keeping the bake's CPU-written buffers valid
// across its asynchronous (no `waitUntilCompleted`) GPU capture. See
// metal/probe.rs `bake_ring_slot`.
pub(super) fn build_rings(gpu: &InitGpu<'_>) -> FrameRings {
    let frames_in_flight = gpu.frames_in_flight;
    FrameRings {
        object: TransientRing::new(frames_in_flight.max(1) + 1),
        draw_args: TransientRing::new(frames_in_flight.max(1) + 1),
        model_history: TransientRing::new(frames_in_flight),
        bindless_tex: TransientRing::new(frames_in_flight.max(1) + 1),
        probe_cube: TransientRing::new(frames_in_flight.max(1) + 1),
        joint: JointRing::new(frames_in_flight.max(1) + 1),
        object_scratch: Vec::new(),
        draw_args_scratch: Vec::new(),
    }
}

pub(super) fn build_diagnostics(gpu: &InitGpu<'_>) -> Diagnostics {
    // `None` when the device does not expose the timestamp counter set; the
    // per-pass GPU timer then stays at zero for every pass.
    let pass_timing = PassTimingResources::new(&gpu.hw.device);
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
        render_fault_logged: Arc::new(AtomicBool::new(false)),
        device_error: Arc::new(Mutex::new(None)),
        pass_fault_count: Arc::new(AtomicU32::new(0)),
        pass_timing,
        pass_times_us: Arc::new(std::array::from_fn(|_| AtomicU32::new(0))),
        draw_calls_accum: AtomicU32::new(0),
    }
}
