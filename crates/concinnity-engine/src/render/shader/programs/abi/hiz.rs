//! The Hi-Z pyramid's SPD dispatches' bindings.

use super::group::ThreadGroup;

/// The single-pass downsampler's group.
pub const SPD_GROUP: ThreadGroup = ThreadGroup::new(16, 16, 1);
/// The pyramid's first level's group.
pub const INIT_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);
/// Each further level's group.
pub const DOWNSAMPLE_GROUP: ThreadGroup = ThreadGroup::new(8, 8, 1);

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

    /// The dispatch's parameters, as root constants.
    pub const PARAMS: Register<Cbv> = Register::new("params", 0);
    /// The depth the first dispatch reduces.
    pub const SRC_DEPTH: Register<Srv> = Register::new("src_depth", 0);
    /// The pyramid's mips, one table.
    pub const SPD_MIPS: Register<Uav> = Register::new("spd_mips", 0);
}

/// The Metal slots.
pub mod metal {

    /// The first level, from the scene depth.
    pub mod init {
        use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

        /// The dispatch's parameters.
        pub const PARAMS: Slot<Buffer> = Slot::new("params", 0);
        /// The scene depth.
        pub const SRC_DEPTH: Slot<Texture> = Slot::new("src_depth", 0);
        /// The first level.
        pub const DST_MIP: Slot<Texture> = Slot::new("dst_mip", 1);
    }

    /// Each further level.
    pub mod downsample {
        use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

        /// The dispatch's parameters.
        pub const PARAMS: Slot<Buffer> = Slot::new("params", 0);
        /// The level above.
        pub const SRC_HIZ: Slot<Texture> = Slot::new("src_hiz", 0);
        /// The level written.
        pub const DST_MIP: Slot<Texture> = Slot::new("dst_mip", 1);
    }
}
