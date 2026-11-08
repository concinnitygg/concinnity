//! The atmosphere's bindings: the table kernels, the sky drawn from them and the haze.

/// The DirectX registers.
pub mod dx {

    /// The table kernels, which share one root signature.
    pub mod kernel {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv, Uav};

        /// The atmosphere block.
        pub const ATMOSPHERE: Register<Cbv> = Register::new("atmo_cb", 0);
        /// The transmittance table, with the multiple-scattering and sky-view
        /// tables in the registers after.
        pub const TRANSMITTANCE: Register<Srv> = Register::new("transmittance_lut", 0);
        /// The multiple-scattering table.
        pub const MULTISCATTER: Register<Srv> = Register::new("multiscatter_lut", 1);
        /// The sky-view table.
        pub const SKY_VIEW: Register<Srv> = Register::new("sky_view_lut", 2);
        /// The clouds' panorama the cubes are drawn behind.
        pub const CLOUD_PANORAMA: Register<Srv> = Register::new("cloud_panorama", 4);
        /// The table the kernel writes.
        pub const OUTPUT: Register<Uav> = Register::new("atmo_out", 0);
        /// The tables' sampler.
        pub const LUT_SAMPLER: Register<Sampler> = Register::new("lut_sampler", 0);
    }

    /// The sky drawn from the sky-view table.
    pub mod sky {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

        /// The view block of the pass it completes.
        pub const VIEW: Register<Cbv> = Register::new("view_cb", 0);
        /// The atmosphere block.
        pub const ATMOSPHERE: Register<Cbv> = Register::new("atmo_cb", 1);
        /// The sky-view table.
        pub const SKY_VIEW: Register<Srv> = Register::new("sky_view_lut", 0);
        /// The table's sampler.
        pub const LUT_SAMPLER: Register<Sampler> = Register::new("lut_sampler", 0);
    }

    /// The haze over the scene.
    pub mod aerial {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

        /// The atmosphere block.
        pub const ATMOSPHERE: Register<Cbv> = Register::new("atmo_cb", 0);
        /// The scene depth.
        pub const SCENE_DEPTH: Register<Srv> = Register::new("scene_depth", 0);
        /// The aerial-perspective volume.
        pub const AERIAL_VOLUME: Register<Srv> = Register::new("aerial_volume", 1);
        /// The transmittance table, with the multiple-scattering table in the register after.
        pub const TRANSMITTANCE: Register<Srv> = Register::new("transmittance_lut", 2);
        /// The multiple-scattering table.
        pub const MULTISCATTER: Register<Srv> = Register::new("multiscatter_lut", 3);
        /// The tables' sampler.
        pub const LUT_SAMPLER: Register<Sampler> = Register::new("lut_sampler", 0);
    }
}
