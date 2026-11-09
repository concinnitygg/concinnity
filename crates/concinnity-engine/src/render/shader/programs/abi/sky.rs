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

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

    /// The pass's view block.
    pub const VIEW: Slot<Buffer> = Slot::new("view_cb", 0);
    /// The sky prefilter cube.
    pub const PREFILTER_CUBE: Slot<Texture> = Slot::new("prefilter_cube", 0);
    /// The cube's sampler.
    pub const CUBE_SAMPLER: Slot<Sampler> = Slot::new("cube_sampler", 0);
}
