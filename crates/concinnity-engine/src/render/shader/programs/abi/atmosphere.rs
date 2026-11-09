//! The atmosphere's bindings: the table kernels, the sky drawn from them and the haze.

use super::group::ThreadGroup;

/// The transmittance table's group.
pub const TRANSMITTANCE_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);
/// The multiple-scattering table's group.
pub const MULTISCATTER_GROUP: ThreadGroup = ThreadGroup::new(64, 1, 1);
/// The sky-view table's group.
pub const SKY_VIEW_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);
/// The aerial-perspective volume's group.
pub const AERIAL_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);
/// The sky irradiance's group.
pub const IRRADIANCE_GROUP: ThreadGroup = ThreadGroup::new(64, 1, 1);
/// The sky cube's group.
pub const SKY_CUBE_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);

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

/// The Metal slots.
pub mod metal {

    /// The table kernels, which share one encoder.
    pub mod kernel {
        use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

        /// The atmosphere block.
        pub const ATMOSPHERE: Slot<Buffer> = Slot::new("atmo_cb", 0);
        /// The transmittance table.
        pub const TRANSMITTANCE: Slot<Texture> = Slot::new("transmittance_lut", 0);
        /// The multiple-scattering table.
        pub const MULTISCATTER: Slot<Texture> = Slot::new("multiscatter_lut", 1);
        /// The sky-view table.
        pub const SKY_VIEW: Slot<Texture> = Slot::new("sky_view_lut", 2);
        /// The table the kernel writes.
        pub const OUTPUT: Slot<Texture> = Slot::new("atmo_out", 3);
        /// The clouds' panorama the cubes are drawn behind.
        pub const CLOUD_PANORAMA: Slot<Texture> = Slot::new("cloud_panorama", 4);
        /// The tables' sampler.
        pub const LUT_SAMPLER: Slot<Sampler> = Slot::new("lut_sampler", 0);
    }

    /// The sky drawn from the sky-view table.
    pub mod sky {
        use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

        /// The view block of the pass it completes.
        pub const VIEW: Slot<Buffer> = Slot::new("view_cb", 0);
        /// The atmosphere block.
        pub const ATMOSPHERE: Slot<Buffer> = Slot::new("atmo_cb", 1);
        /// The sky-view table.
        pub const SKY_VIEW: Slot<Texture> = Slot::new("sky_view_lut", 0);
        /// The table's sampler.
        pub const LUT_SAMPLER: Slot<Sampler> = Slot::new("lut_sampler", 0);
    }

    /// The haze over the scene.
    pub mod aerial {
        use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

        /// The atmosphere block.
        pub const ATMOSPHERE: Slot<Buffer> = Slot::new("atmo_cb", 0);
        /// The scene depth.
        pub const SCENE_DEPTH: Slot<Texture> = Slot::new("scene_depth", 0);
        /// The aerial-perspective volume.
        pub const AERIAL_VOLUME: Slot<Texture> = Slot::new("aerial_volume", 1);
        /// The transmittance table.
        pub const TRANSMITTANCE: Slot<Texture> = Slot::new("transmittance_lut", 2);
        /// The multiple-scattering table.
        pub const MULTISCATTER: Slot<Texture> = Slot::new("multiscatter_lut", 3);
        /// The tables' sampler.
        pub const LUT_SAMPLER: Slot<Sampler> = Slot::new("lut_sampler", 0);
    }
}
