// src/metal/mod.rs
//
// GPU-free, CPU-side pieces of the Metal backend: the repr(C) uniform structs
// mirrored in the MSL shaders. None of these touch a Metal device or encoder,
// and they are unit-tested without a GPU. They live in `core::render` (not the
// excluded concinnity-device crate) and are compiled unconditionally, so their
// layout tests run on every platform's CI and count toward coverage. The Metal
// backend re-exports them under `metal` so it keeps its existing `uniforms`
// path.

pub mod uniforms;
