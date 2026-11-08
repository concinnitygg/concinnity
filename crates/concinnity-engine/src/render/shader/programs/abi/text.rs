//! The text overlay's bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

    /// The overlay's block.
    pub const UNIFORMS: Register<Cbv> = Register::new("uni", 0);
    /// The glyph atlas.
    pub const ATLAS: Register<Srv> = Register::new("atlas", 0);
    /// The atlas's sampler.
    pub const ATLAS_SAMPLER: Register<Sampler> = Register::new("atlas_sampler", 0);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

    /// The overlay's block.
    pub const UNIFORMS: Slot<Buffer> = Slot::new("uni", 0);
    /// The glyph atlas.
    pub const ATLAS: Slot<Texture> = Slot::new("atlas", 0);
    /// The atlas's sampler.
    pub const ATLAS_SAMPLER: Slot<Sampler> = Slot::new("atlas_sampler", 0);
}
