//! The volumetric clouds' bindings: the noise, the march kernels and the composite.

use super::group::ThreadGroup;

/// The noise volume's group.
pub const NOISE_GROUP: ThreadGroup = ThreadGroup::new(4, 4, 4);
/// The shadow map's group.
pub const SHADOW_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);
/// The panorama's group.
pub const PANORAMA_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);
/// The march's group.
pub const MARCH_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);
/// The reconstruction's group.
pub const RECONSTRUCT_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);
/// The reflected sky's group, one thread per listed ray.
pub const REFLECT_GROUP: ThreadGroup =
    ThreadGroup::new(crate::render::post::rt_reflections::MISS_GROUP, 1, 1);

/// The DirectX registers.
pub mod dx {

    /// The noise kernel.
    pub mod noise {
        use crate::render::shader::programs::abi::register::{Register, Uav};

        /// The noise volume it writes.
        pub const NOISE_OUT: Register<Uav> = Register::new("cloud_noise_out", 0);
    }

    /// The march kernels and the reconstruction, which share one root signature.
    pub mod march {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv, Uav};

        /// The atmosphere block.
        pub const ATMOSPHERE: Register<Cbv> = Register::new("atmo_cb", 0);
        /// The clouds block.
        pub const CLOUDS: Register<Cbv> = Register::new("cloud_cb", 1);
        /// The scene depth.
        pub const SCENE_DEPTH: Register<Srv> = Register::new("scene_depth", 0);
        /// The aerial-perspective volume.
        pub const AERIAL_VOLUME: Register<Srv> = Register::new("aerial_volume", 1);
        /// The transmittance table, with the multiple-scattering table in the register after.
        pub const TRANSMITTANCE: Register<Srv> = Register::new("transmittance_lut", 2);
        /// The multiple-scattering table.
        pub const MULTISCATTER: Register<Srv> = Register::new("multiscatter_lut", 3);
        /// The noise volume.
        pub const NOISE: Register<Srv> = Register::new("cloud_noise", 4);
        /// The march, which the reconstruction reads.
        pub const MARCHED: Register<Srv> = Register::new("cloud_marched", 6);
        /// The previous frame's target.
        pub const PREVIOUS: Register<Srv> = Register::new("cloud_previous", 7);
        /// The sky-view table, which the reflected sky reads.
        pub const SKY_VIEW: Register<Srv> = Register::new("sky_view_lut", 8);
        /// The reflection rays listed for the sky.
        pub const MISSES: Register<Srv> = Register::new("reflection_misses", 9);
        /// The listed rays' indirect arguments.
        pub const MISS_ARGS: Register<Srv> = Register::new("reflection_miss_args", 10);
        /// The panorama, which the reflected ground's water reads.
        pub const PANORAMA: Register<Srv> = Register::new("cloud_panorama", 11);
        /// The target written.
        pub const OUTPUT: Register<Uav> = Register::new("cloud_out", 0);
        /// The tables' sampler.
        pub const LUT_SAMPLER: Register<Sampler> = Register::new("lut_sampler", 0);
        /// The noise's sampler.
        pub const NOISE_SAMPLER: Register<Sampler> = Register::new("cloud_noise_sampler", 1);
    }

    /// The composite over the scene.
    pub mod composite {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Srv};

        /// The clouds block.
        pub const CLOUDS: Register<Cbv> = Register::new("cloud_cb", 0);
        /// The scene depth.
        pub const SCENE_DEPTH: Register<Srv> = Register::new("scene_depth", 0);
        /// The reconstructed clouds.
        pub const TARGET: Register<Srv> = Register::new("cloud_target", 1);
    }
}

/// The Metal slots.
pub mod metal {

    /// The noise kernel.
    pub mod noise {
        use crate::render::shader::programs::abi::slot::{Slot, Texture};

        /// The noise volume it writes.
        pub const NOISE_OUT: Slot<Texture> = Slot::new("cloud_noise_out", 0);
    }

    /// The march kernels and the reconstruction.
    pub mod march {
        use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

        /// The atmosphere block.
        pub const ATMOSPHERE: Slot<Buffer> = Slot::new("atmo_cb", 0);
        /// The clouds block.
        pub const CLOUDS: Slot<Buffer> = Slot::new("cloud_cb", 1);
        /// The scene depth.
        pub const SCENE_DEPTH: Slot<Texture> = Slot::new("scene_depth", 0);
        /// The aerial-perspective volume.
        pub const AERIAL_VOLUME: Slot<Texture> = Slot::new("aerial_volume", 1);
        /// The transmittance table.
        pub const TRANSMITTANCE: Slot<Texture> = Slot::new("transmittance_lut", 2);
        /// The multiple-scattering table.
        pub const MULTISCATTER: Slot<Texture> = Slot::new("multiscatter_lut", 3);
        /// The noise volume.
        pub const NOISE: Slot<Texture> = Slot::new("cloud_noise", 4);
        /// The target written.
        pub const OUTPUT: Slot<Texture> = Slot::new("cloud_out", 5);
        /// The march, which the reconstruction reads.
        pub const MARCHED: Slot<Texture> = Slot::new("cloud_marched", 6);
        /// The previous frame's target.
        pub const PREVIOUS: Slot<Texture> = Slot::new("cloud_previous", 7);
        /// The sky-view table, which the reflected sky reads.
        pub const SKY_VIEW: Slot<Texture> = Slot::new("sky_view_lut", 8);
        /// The reflection rays listed for the sky.
        pub const MISSES: Slot<Buffer> = Slot::new("reflection_misses", 2);
        /// The listed rays' indirect arguments.
        pub const MISS_ARGS: Slot<Buffer> = Slot::new("reflection_miss_args", 3);
        /// The panorama, which the reflected ground's water reads.
        pub const PANORAMA: Slot<Texture> = Slot::new("cloud_panorama", 9);
        /// The tables' sampler.
        pub const LUT_SAMPLER: Slot<Sampler> = Slot::new("lut_sampler", 0);
        /// The noise's sampler.
        pub const NOISE_SAMPLER: Slot<Sampler> = Slot::new("cloud_noise_sampler", 1);
    }

    /// The composite over the scene.
    pub mod composite {
        use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

        /// The clouds block.
        pub const CLOUDS: Slot<Buffer> = Slot::new("cloud_cb", 0);
        /// The scene depth.
        pub const SCENE_DEPTH: Slot<Texture> = Slot::new("scene_depth", 0);
        /// The reconstructed clouds.
        pub const TARGET: Slot<Texture> = Slot::new("cloud_target", 1);
    }
}
