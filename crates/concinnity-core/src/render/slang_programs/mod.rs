//! What each backend compiles from the single-source `.slang` shaders.
//!
//! Declarations only: the file, the entry point, the target profile, and the
//! variant defines, as plain `&'static` data. They live here rather than in the
//! device crate because both halves of the toolchain iterate them -- the
//! renderer to compile a program at init, and the device build script to
//! compile the same programs ahead of time -- and a build script cannot read
//! its own crate's data. One table is what keeps the two from disagreeing about
//! what a program's source is, which the content-addressed shader cache would
//! otherwise paper over by serving one path's bytes to the other.
//!
//! Everything that needs a compiler, a cache, or a filesystem stays in the
//! device crate and reaches these through a trait.

/// What the DirectX backend compiles to DXIL.
pub mod dx;

/// What the Vulkan backend compiles to SPIR-V.
pub mod vk;
