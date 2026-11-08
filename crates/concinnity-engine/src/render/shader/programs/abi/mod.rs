//! Where each program's resources bind on the backends whose hosts place
//! them by hand: the DirectX root signatures and the Metal encoders.
//!
//! The device build script asserts every row against the compiled shaders, so
//! a declaration that moves fails the build instead of binding garbage at draw
//! time. A pass whose bindings have named constants here (`cull`, `main`)
//! binds through them too, so its table row and its host cannot disagree. Kept
//! beside the program table so the build script reads it as data.

pub mod cull;
mod dx;
pub mod main;
mod metal;
pub mod register;
pub mod slot;

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
