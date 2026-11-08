//! The volumetric fog's bindings.

/// The DirectX registers.
pub mod dx {

    /// The froxel kernel.
    pub mod froxel {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv, Uav};

        /// The fog block.
        pub const FOG: Register<Cbv> = Register::new("fog", 0);
        /// The froxel grid block.
        pub const FROXEL: Register<Cbv> = Register::new("froxel", 1);
        /// The cascaded shadow block.
        pub const SHADOW: Register<Cbv> = Register::new("shadow_uni", 2);
        /// The cascaded shadow map array.
        pub const SHADOW_MAP: Register<Srv> = Register::new("shadow_map", 0);
        /// The shadow comparison sampler.
        pub const SHADOW_SAMPLER: Register<Sampler> = Register::new("shadow_samp", 0);
        /// The froxel volume it writes.
        pub const VOLUME: Register<Uav> = Register::new("fog_volume", 0);
    }

    /// The fullscreen fog pass.
    pub mod draw {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

        /// The fog block.
        pub const FOG: Register<Cbv> = Register::new("fog", 0);
        /// The froxel grid block.
        pub const FROXEL: Register<Cbv> = Register::new("froxel", 1);
        /// The scene depth.
        pub const SCENE_DEPTH: Register<Srv> = Register::new("scene_depth", 0);
        /// The froxel volume.
        pub const VOLUME: Register<Srv> = Register::new("fog_volume", 1);
        /// The volume's sampler.
        pub const VOLUME_SAMPLER: Register<Sampler> = Register::new("fog_volume_samp", 0);
    }
}

/// The Metal slots.
pub mod metal {

    /// The froxel kernel.
    pub mod froxel {
        use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

        /// The fog block.
        pub const FOG: Slot<Buffer> = Slot::new("fog", 0);
        /// The froxel grid block.
        pub const FROXEL: Slot<Buffer> = Slot::new("froxel", 1);
        /// The cascaded shadow block.
        pub const SHADOW: Slot<Buffer> = Slot::new("shadow_uni", 2);
        /// The cascaded shadow map array.
        pub const SHADOW_MAP: Slot<Texture> = Slot::new("shadow_map", 0);
        /// The shadow comparison sampler.
        pub const SHADOW_SAMPLER: Slot<Sampler> = Slot::new("shadow_samp", 0);
        /// The froxel volume it writes.
        pub const VOLUME: Slot<Texture> = Slot::new("fog_volume", 1);
    }

    /// The fullscreen fog pass.
    pub mod draw {
        use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

        /// The fog block.
        pub const FOG: Slot<Buffer> = Slot::new("fog", 0);
        /// The froxel grid block.
        pub const FROXEL: Slot<Buffer> = Slot::new("froxel", 1);
        /// The scene depth.
        pub const SCENE_DEPTH: Slot<Texture> = Slot::new("scene_depth", 0);
        /// The froxel volume.
        pub const VOLUME: Slot<Texture> = Slot::new("fog_volume", 1);
        /// The volume's sampler.
        pub const VOLUME_SAMPLER: Slot<Sampler> = Slot::new("fog_volume_samp", 0);
    }
}
