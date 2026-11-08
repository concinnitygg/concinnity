//! The sky pass's bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

    /// The pass's view block.
    pub const VIEW: Register<Cbv> = Register::new("view_cb", 0);
    /// The sky prefilter cube.
    pub const PREFILTER_CUBE: Register<Srv> = Register::new("prefilter_cube", 0);
    /// The cube's sampler.
    pub const CUBE_SAMPLER: Register<Sampler> = Register::new("cube_sampler", 0);
}
