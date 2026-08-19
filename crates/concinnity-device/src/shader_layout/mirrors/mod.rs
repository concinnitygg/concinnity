// src/shader_layout/mirrors/
//
// The `#[repr(C)]` mirrors, grouped by the shader family whose reflection
// checks them. A mirror pairs runs of Rust fields with the runs of shader
// fields covering the same bytes, and carries no expected offset of its own:
// every number in the comparison comes from slangc.

pub(super) mod forward;
pub(super) mod geometry;
pub(super) mod post;
pub(super) mod transparent;
