// src/directx/mod.rs
//
// GPU-free, CPU-side pieces of the DirectX backend: the repr(C) uniform structs
// mirrored in the HLSL shaders (cbuffer / root-constant layouts), the
// reflection-probe uniforms, and the per-pass GPU-timing slot arithmetic. None
// of these touch a D3D12 device or command list, and they are unit-tested
// without a GPU. They live in concinnity-render (not the excluded
// concinnity-device crate) and are compiled unconditionally, so their layout
// tests run on every platform's CI and count toward coverage. The DirectX
// backend re-exports them under `directx` so it keeps its existing
// `pass_timing` / `probe_uniforms` / `uniforms` paths.

pub mod pass_timing;
pub mod probe_uniforms;
pub mod uniforms;
