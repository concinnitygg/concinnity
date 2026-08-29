//! Per-pass GPU timing on D3D12 via TIMESTAMP queries. The slot layout is the
//! shared one ([`crate::render::pass_timing`]); this module only re-exports it so the
//! backend keeps its `crate::render::directx::pass_timing` path.
//!
//! `execute_graph` issues an `EndQuery` before and after each pass's `encode_*`,
//! and the resolve at the end of the command list copies the whole block into
//! the persistently-mapped readback buffer. The CPU reads the previous frame's
//! block at the top of `draw_frame` (after the matching fence wait gates the GPU
//! writes) and publishes the per-pass microseconds into `RenderStats`.
//!
//! SsaoPrepass / SsaoKernel / ParticlesSim are bundled inside their parent
//! encoders, and the FogFroxel / Upscale / Transparent / Raymarch arms are
//! no-ops here, so those slots stay zero. `StatHud.passes_text` picks the top
//! six non-zero entries, so zero slots drop out of the on-screen chip.

pub use crate::render::pass_timing::*;
