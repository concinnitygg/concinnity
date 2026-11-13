//! Where each program's resources bind on the backends whose hosts place
//! them by hand: the DirectX root signatures and the Metal encoders. Each pass
//! also names its compute kernels' thread-group sizes here (`group`), which
//! every backend dispatches at.
//!
//! The device build script asserts every row against the compiled shaders, so
//! a declaration that moves fails the build instead of binding garbage at draw
//! time. Each pass names its bindings once, in its module here, and both its
//! table rows and its host bind through those constants, so the two cannot
//! disagree. The fullscreen post passes place their sources by position
//! instead (`post`), so their rows spell each source at its position. Kept
//! beside the program table so the build script reads it as data.

pub mod atmosphere;
pub mod clouds;
pub mod composite;
pub mod cull;
pub mod decal;
mod dx;
pub mod exposure;
pub mod fog;
pub mod gbuffer_sky;
pub mod grass;
pub mod group;
pub mod history;
pub mod hiz;
pub mod light_cull;
pub mod line;
pub mod main;
mod metal;
pub mod ocean;
pub mod particle;
pub mod post;
pub mod probe;
pub mod raymarch;
pub mod receivers;
pub mod register;
pub mod rt_reflections;
pub mod shadow;
pub mod skin;
pub mod sky;
pub mod slot;
pub mod text;
pub mod transparent;
pub mod wake;

pub use dx::DXIL_ENTRY_ABI;
pub use metal::METAL_ENTRY_ABI;

use super::ShaderProgram;

/// One DirectX program and the registers its root signature binds.
pub struct DxilAbi {
    /// The program.
    pub program: &'static ShaderProgram,
    /// The registers its root signature binds.
    pub registers: &'static [register::DxBinding],
}

/// One Metal program and the slots its host encoder writes.
pub struct MetalAbi {
    /// The program.
    pub program: &'static ShaderProgram,
    /// The slots its encoder writes, grouped as the encoder binds them.
    pub slots: &'static [&'static [slot::MetalBinding]],
    /// Argument-buffer members and the `[[id(n)]]` the host encodes each at.
    pub argument_ids: &'static [(&'static str, usize)],
}

impl MetalAbi {
    /// Every slot of the row, across its groups.
    pub fn slots(&self) -> impl Iterator<Item = &'static slot::MetalBinding> {
        self.slots.iter().flat_map(|group| group.iter())
    }
}

#[cfg(test)]
mod tests;
