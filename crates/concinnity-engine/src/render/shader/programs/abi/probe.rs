//! The reflection-probe prefilter's bindings.

/// The DirectX registers.
pub mod dx {

    /// The mip-0 copy and the downsample, which share one root signature.
    pub mod mip {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Uav};

        /// The dispatch's parameters, as root constants.
        pub const PARAMS: Register<Cbv> = Register::new("params", 0);
        /// The mip read, with the mip written in the register after.
        pub const SRC_MIP: Register<Uav> = Register::new("src_mip", 0);
        /// The mip written.
        pub const DST_MIP: Register<Uav> = Register::new("dst_mip", 1);
    }

    /// The GGX convolution.
    pub mod ggx {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv, Uav};

        /// The dispatch's parameters, as root constants.
        pub const PARAMS: Register<Cbv> = Register::new("params", 0);
        /// The cube it convolves.
        pub const SRC_CUBE: Register<Srv> = Register::new("src_cube", 0);
        /// The cube's sampler.
        pub const SRC_SAMPLER: Register<Sampler> = Register::new("src_sampler", 0);
        /// The mip written.
        pub const DST_MIP: Register<Uav> = Register::new("dst_mip", 0);
    }
}
