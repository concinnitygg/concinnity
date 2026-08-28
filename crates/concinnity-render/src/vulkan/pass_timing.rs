//! Per-pass GPU timing on Vulkan via TIMESTAMP queries. The slot layout is the
//! shared one ([`crate::pass_timing`]); this module only re-exports it so the
//! backend keeps its `crate::vulkan::pass_timing` path.
//!
//! The start buffer resets the whole block and writes the whole-frame start;
//! each per-pass command buffer writes its own pair around its encode; the end
//! buffer writes the whole-frame end.
//!
//! Unlike D3D12 (which can pre-write every slot so a pass that did not run still
//! reads a value), Vulkan forbids writing a timestamp to a query that is already
//! written without an intervening reset. A pass absent from this frame's graph
//! therefore leaves its reset-but-unwritten slots `unavailable`; the readback
//! uses `WITH_AVAILABILITY` and reports 0 for any pair that is not both
//! available.

pub use crate::pass_timing::*;
