// src/metal/mod.rs
//
// GPU-free, CPU-side pieces of the Metal backend: the repr(C) uniform structs
// mirrored in the MSL shaders and the shader-layout binding contract. None of these touch a Metal device or encoder, and they are
// unit-tested without a GPU. They live in concinnity-render (not the excluded
// concinnity-device crate) and are compiled unconditionally, so their layout
// tests run on every platform's CI and count toward coverage. The Metal backend
// re-exports them under `metal` so it keeps its existing `uniforms` /
// `shader_layout` paths.

pub mod shader_layout;
pub mod uniforms;
