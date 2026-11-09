//! The grass passes' bindings.

use super::group::ThreadGroup;
use crate::render::environment::grass::tiles::GRASS_GROUP_SIZE;

/// The blade generation's group.
pub const GENERATE_GROUP: ThreadGroup = ThreadGroup::new(GRASS_GROUP_SIZE, 1, 1);
/// The bend field's group.
pub const BEND_GROUP: ThreadGroup = ThreadGroup::new(GRASS_GROUP_SIZE, 1, 1);

/// The DirectX registers.
pub mod dx {

    /// The kernel that places the visible blades.
    pub mod generate {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

        /// The grass block.
        pub const GRASS: Register<Cbv> = Register::new("grass", 0);
        /// The blades the kernel appends.
        pub const BLADES_OUT: Register<Uav> = Register::new("blades_out", 1);
        /// The draw arguments it fills.
        pub const DRAW_ARGS: Register<Uav> = Register::new("draw_args", 2);
        /// The terrain heights.
        pub const TERRAIN_HEIGHTS: Register<Srv> = Register::new("terrain_heights", 3);
        /// The grass mask texels.
        pub const GRASS_MASKS: Register<Srv> = Register::new("grass_masks", 4);
        /// Last frame's depth pyramid.
        pub const HIZ: Register<Srv> = Register::new("grass_hiz", 5);
        /// The bend field.
        pub const BEND_FIELD: Register<Srv> = Register::new("grass_bend_field", 6);
        /// The planet tiles.
        pub const PLANET_TILES: Register<Srv> = Register::new("planet_tiles", 7);
        /// The shared vertex stream.
        pub const MESH_VERTICES: Register<Srv> = Register::new("mesh_vertices", 8);
        /// The tile records.
        pub const TILES: Register<Srv> = Register::new("grass_tiles", 9);
    }

    /// The bend pass.
    pub mod bend {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Uav};

        /// The bend block.
        pub const BEND: Register<Cbv> = Register::new("bend", 0);
        /// The bend field it writes.
        pub const BEND_FIELD: Register<Uav> = Register::new("bend_field", 1);
    }

    /// The cascade draw.
    pub mod shadow {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Srv};

        /// The cascade's block.
        pub const GRASS: Register<Cbv> = Register::new("grass", 0);
        /// The cascade's blades.
        pub const BLADES: Register<Srv> = Register::new("grass_blades", 1);
    }
}

/// The Metal slots.
pub mod metal {

    /// The kernel that places the visible blades.
    pub mod generate {
        use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

        /// The grass block.
        pub const GRASS: Slot<Buffer> = Slot::new("grass", 0);
        /// The blades the kernel appends.
        pub const BLADES_OUT: Slot<Buffer> = Slot::new("blades_out", 1);
        /// The draw arguments it fills.
        pub const DRAW_ARGS: Slot<Buffer> = Slot::new("draw_args", 2);
        /// The terrain heights.
        pub const TERRAIN_HEIGHTS: Slot<Buffer> = Slot::new("terrain_heights", 3);
        /// The grass mask texels.
        pub const GRASS_MASKS: Slot<Buffer> = Slot::new("grass_masks", 4);
        /// Last frame's depth pyramid.
        pub const HIZ: Slot<Texture> = Slot::new("grass_hiz", 5);
        /// The bend field.
        pub const BEND_FIELD: Slot<Buffer> = Slot::new("grass_bend_field", 6);
        /// The planet tiles.
        pub const PLANET_TILES: Slot<Buffer> = Slot::new("planet_tiles", 7);
        /// The shared vertex stream.
        pub const MESH_VERTICES: Slot<Buffer> = Slot::new("mesh_vertices", 8);
        /// The tile records.
        pub const TILES: Slot<Buffer> = Slot::new("grass_tiles", 9);
    }

    /// The bend pass.
    pub mod bend {
        use crate::render::shader::programs::abi::slot::{Buffer, Slot};

        /// The bend block.
        pub const BEND: Slot<Buffer> = Slot::new("bend", 0);
        /// The bend field it writes.
        pub const BEND_FIELD: Slot<Buffer> = Slot::new("bend_field", 1);
    }

    /// The cascade draw.
    pub mod shadow {
        use crate::render::shader::programs::abi::slot::{Buffer, Slot};

        /// The cascade's block.
        pub const GRASS: Slot<Buffer> = Slot::new("grass", 0);
        /// The cascade's blades.
        pub const BLADES: Slot<Buffer> = Slot::new("grass_blades", 1);
    }

    /// The main and pre-pass draws, beside the main pass's own slots.
    pub mod draw {
        use crate::render::shader::programs::abi::slot::{Buffer, Slot};

        /// The grass block.
        pub const GRASS: Slot<Buffer> = Slot::new("grass", 19);
        /// The visible blades.
        pub const BLADES: Slot<Buffer> = Slot::new("grass_blades", 20);
    }
}
