//! Input for the engine: the system that consumes a frame's raw input, and the
//! gilrs adapter at its edge. The pad fold, stick shaping, navigation pulses
//! and text editing are core's `input`.

pub(crate) mod gamepad;

// Per-frame input sampling + FrameInput publish. Internal system, constructed
// alongside GraphicsSystem (same gate) and scheduled immediately after it.
pub(crate) mod system;
