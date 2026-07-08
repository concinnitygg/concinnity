// src/vulkan/mod.rs
//
// GPU-free, CPU-side pieces of the Vulkan backend: the repr(C) uniform structs
// mirrored in the GLSL shaders (std140/std430 layouts), the reflection-probe
// uniforms, and the per-pass GPU-timing slot arithmetic. None of these touch a
// Vulkan device or command buffer, and they are unit-tested without a GPU. They
// live in concinnity-render (not the excluded concinnity-device crate) and are
// compiled unconditionally, so their layout tests run on every platform's CI and
// count toward coverage. The Vulkan backend re-exports them under `vulkan` so it
// keeps its existing `pass_timing` / `probe_uniforms` / `uniforms` paths.

pub mod pass_timing;
pub mod probe_uniforms;
pub mod uniforms;
