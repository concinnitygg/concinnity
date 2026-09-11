//! GPU-free, CPU-side pieces of the DirectX backend: the repr(C) uniform structs
//! mirrored in the HLSL shaders (cbuffer / root-constant layouts) and the
//! reflection-probe uniforms.
//!
//! None of these touch a D3D12 device or command list, and they are unit-tested
//! without a GPU. They live in `core::render` (not the excluded
//! concinnity-device crate) and are compiled unconditionally, so their layout
//! tests run on every platform's CI and count toward coverage.

pub mod uniforms;
