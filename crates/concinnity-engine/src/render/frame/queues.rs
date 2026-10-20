//! The frame's recording surfaces and the device facts recording systems read
//! without reaching the backend.

use crate::ecs::Parked;
use crate::render::backend::DeviceCapabilities;
use crate::render::frame::ops::RenderOps;
use crate::render::slots::render_slots::RenderSlots;

/// The frame's recording surfaces, taken and re-parked together by each
/// recording system: the op queue the tick's backend effects accumulate into
/// (drained into the frame snapshot by the render extract, replayed in record
/// order before the draw) and the slot-allocation authority ops name
/// destinations from.
pub struct RenderQueues {
    /// The tick's recorded backend effects.
    pub ops: RenderOps,
    /// Draw-slot and skinned-instance allocation.
    pub slots: RenderSlots,
}

/// The world's parked [`RenderQueues`]. Published by graphics init; absent in
/// a world with no graphics, so recording systems no-op.
pub type ActiveRenderQueues = Parked<RenderQueues>;

/// The active backend's capability flags, published by graphics init. The
/// backend itself is parked (and on a pipelined frame, owned by the render
/// thread), so a system that only needs to know what it supports reads this
/// instead. Absent in a world with no graphics.
#[derive(Debug, Clone, Copy)]
pub struct ActiveDeviceCaps(pub DeviceCapabilities);
