//! The particle passes' bindings.

/// The DirectX registers.
pub mod dx {

    /// The simulation.
    pub mod simulate {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

        /// The simulation's block.
        pub const PARAMS: Register<Cbv> = Register::new("params", 0);
        /// The particle pool.
        pub const POOL: Register<Uav> = Register::new("pool", 0);
        /// This frame's spawn rows.
        pub const FRAME_ROWS: Register<Srv> = Register::new("frame_rows", 1);
    }

    /// The draw, lit or not.
    pub mod draw {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

        /// The view block.
        pub const VIEW: Register<Cbv> = Register::new("view", 0);
        /// The emitter's block.
        pub const PARAMS: Register<Cbv> = Register::new("params", 1);
        /// The particle pool.
        pub const POOL: Register<Srv> = Register::new("pool", 0);
        /// The sprite texture.
        pub const ALBEDO: Register<Srv> = Register::new("albedo", 1);
        /// The scene depth the sprites fade against.
        pub const SCENE_DEPTH: Register<Srv> = Register::new("scene_depth", 2);
        /// The sprite's sampler.
        pub const ALBEDO_SAMPLER: Register<Sampler> = Register::new("albedo_sampler", 0);
    }
}
