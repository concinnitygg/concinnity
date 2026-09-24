//! What the Vulkan backend compiles to SPIR-V: the shared engine programs and
//! the single-pass Hi-Z downsampler.
//!
//! Each program compiles a file under `src/render/shaders/`, at build time where
//! the host has dxc and at renderer init otherwise, cached in the
//! content-addressed shader cache.
//!
//! The `[[vk::binding]]` annotations in the sources are the engine's
//! descriptor-set layouts. dxc names every SPIR-V entry point `main`, which is
//! what pipeline stage creation asks for.
//!
//! The ray-traced programs compile to `SPV_KHR_ray_query` (SPIR-V 1.5), so the
//! host builds them only on a device with `VK_KHR_ray_query`.

use super::{Table, shared, spd};

/// Everything the Vulkan backend compiles.
pub static TABLE: Table = Table {
    programs: &[shared::ALL, spd::ALL],
    msaa: &[false, true],
};
