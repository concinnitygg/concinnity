//! The render half's per-frame report back to the simulation: the sampled
//! window input, the frame's render stats, what the snapshot's op replay
//! produced, and the consumed snapshot returned for buffer reuse. The
//! counterpart of `RenderSnapshot` on the pipelined driver's return channel.

use crate::gfx::profile::RenderStats;
use crate::render::input::InputPacket;
use crate::render::ops::ReplayOutcome;
use crate::render::snapshot::RenderSnapshot;

/// One frame's results crossing from the render half to the simulation.
pub struct FrameFeedback {
    /// Window input sampled right after the draw (whose event pump produced
    /// it), destined for the simulation's input mailbox.
    pub input: InputPacket,
    /// The drawn frame's backend stats, for the profiler surfaces.
    pub render_stats: RenderStats,
    /// What the frame's op replay produced (failures to roll back, memory
    /// pressure from an upload).
    pub replay: ReplayOutcome,
    /// The consumed snapshot, returned so its buffers keep their capacity.
    pub recycled: RenderSnapshot,
    /// The render half stopped (window closed, frame policy shutdown, device
    /// lost); the simulation must stop too.
    pub stop: bool,
}

// The feedback must stay owned data so it can cross the thread boundary.
const _: () = {
    const fn require_send<T: Send + 'static>() {}
    require_send::<FrameFeedback>()
};
