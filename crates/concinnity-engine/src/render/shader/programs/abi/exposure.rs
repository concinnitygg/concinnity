//! The auto-exposure's bindings.

/// The DirectX registers.
pub mod dx {

    /// The histogram build.
    pub mod build {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

        /// The pass's parameters, as root constants.
        pub const PARAMS: Register<Cbv> = Register::new("params", 0);
        /// The HDR scene the histogram reads.
        pub const HDR_TEXTURE: Register<Srv> = Register::new("hdr_texture", 0);
        /// The histogram.
        pub const HISTOGRAM: Register<Uav> = Register::new("histogram", 0);
    }

    /// The histogram average.
    pub mod average {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Uav};

        /// The pass's parameters, as root constants.
        pub const PARAMS: Register<Cbv> = Register::new("params", 0);
        /// The histogram.
        pub const HISTOGRAM: Register<Uav> = Register::new("histogram", 0);
        /// The measured exposure.
        pub const MEASUREMENT: Register<Uav> = Register::new("measurement", 1);
    }
}
