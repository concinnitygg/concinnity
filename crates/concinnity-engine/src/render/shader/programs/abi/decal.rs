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
