//! The shadow receiver bounds' bindings: the reduction of the depth into
//! per-tile bounds, and their merge into one block the CPU reads back.

use super::group::ThreadGroup;

/// The per-tile reduction's group. Each group covers a
/// [`RECEIVER_TILE`](crate::gfx::render_types::RECEIVER_TILE) square.
pub const REDUCE_GROUP: ThreadGroup = ThreadGroup::new(16, 16, 1);
/// The merge's single group.
pub const MERGE_GROUP: ThreadGroup = ThreadGroup::new(256, 1, 1);

/// The DirectX registers.
pub mod dx {

    /// The per-tile reduction.
    pub mod reduce {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

        /// The pass's parameters, as root constants.
        pub const PARAMS: Register<Cbv> = Register::new("params", 0);
        /// The Hi-Z pyramid's first level.
        pub const DEPTH: Register<Srv> = Register::new("depth", 0);
        /// Each tile's bounds.
        pub const PARTIALS: Register<Uav> = Register::new("partials", 0);
    }

    /// The merge.
    pub mod merge {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Uav};

        /// The pass's parameters, as root constants.
        pub const PARAMS: Register<Cbv> = Register::new("params", 0);
        /// Each tile's bounds.
        pub const PARTIALS: Register<Uav> = Register::new("partials", 0);
        /// The merged bounds.
        pub const BOUNDS: Register<Uav> = Register::new("bounds", 1);
    }
}

/// The Metal slots.
pub mod metal {

    /// The per-tile reduction.
    pub mod reduce {
        use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

        /// The Hi-Z pyramid's first level.
        pub const DEPTH: Slot<Texture> = Slot::new("depth", 0);
        /// Each tile's bounds.
        pub const PARTIALS: Slot<Buffer> = Slot::new("partials", 0);
        /// The pass's parameters.
        pub const PARAMS: Slot<Buffer> = Slot::new("params", 1);
    }

    /// The merge.
    pub mod merge {
        use crate::render::shader::programs::abi::slot::{Buffer, Slot};

        /// Each tile's bounds.
        pub const PARTIALS: Slot<Buffer> = Slot::new("partials", 0);
        /// The merged bounds.
        pub const BOUNDS: Slot<Buffer> = Slot::new("bounds", 1);
        /// The pass's parameters.
        pub const PARAMS: Slot<Buffer> = Slot::new("params", 2);
    }
}
