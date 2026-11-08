//! The decal pass's bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

    /// The view block.
    pub const VIEW: Register<Cbv> = Register::new("view", 0);
    /// The decal's block.
    pub const PARAMS: Register<Cbv> = Register::new("params", 1);
    /// The scene depth the decal projects onto.
    pub const SCENE_DEPTH: Register<Srv> = Register::new("scene_depth", 0);
    /// The decal's texture.
    pub const DECAL_TEX: Register<Srv> = Register::new("decal_tex", 1);
    /// The texture's sampler.
    pub const DECAL_TEX_SAMPLER: Register<Sampler> = Register::new("decal_tex_sampler", 0);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

    /// The view block.
    pub const VIEW: Slot<Buffer> = Slot::new("view", 0);
    /// The decal's block.
    pub const PARAMS: Slot<Buffer> = Slot::new("params", 1);
    /// The scene depth the decal projects onto.
    pub const SCENE_DEPTH: Slot<Texture> = Slot::new("scene_depth", 0);
    /// The decal's texture.
    pub const DECAL_TEX: Slot<Texture> = Slot::new("decal_tex", 1);
    /// The texture's sampler.
    pub const DECAL_TEX_SAMPLER: Slot<Sampler> = Slot::new("decal_tex_sampler", 0);
}
