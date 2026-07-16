// src/gfx/streaming/mod.rs
//
// The asset-streaming home. The `no_std` policy core (`StreamPlanner` /
// `StreamState` and its LRU scoring) lives in `concinnity-render` so a future
// `no_std` client runtime can share it; it is re-exported here so the `std`
// drivers below and the rest of `gfx` name it as `crate::gfx::streaming::*`.
#[cfg_attr(not(target_os = "macos"), allow(unused_imports))]
pub(crate) use concinnity_render::streaming::*;

// Async asset-streaming drivers. Currently driven only by the Metal
// backend (Vulkan and DirectX catch-up is a separate follow-up), so on
// non-macOS builds these modules are compiled but unreferenced.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod chunk;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod mesh;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod texture;
