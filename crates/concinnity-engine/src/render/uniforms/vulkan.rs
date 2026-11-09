//! Push-constant sizes only the Vulkan frame encoders declare. Blocks declared
//! once for every backend live in the parent module.

/// Byte size of the auto-exposure push-constant range. Pins the struct size to
/// what the pipeline layout declares.
pub const AUTO_EXPOSURE_PUSH_BYTES: u32 = 16;
