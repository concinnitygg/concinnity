//! The ray-tracing skin dispatch's bindings.

use super::group::ThreadGroup;

/// The skin kernel's group.
pub const SKIN_GROUP: ThreadGroup = ThreadGroup::new(64, 1, 1);

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

    /// The dispatch's parameters, as root constants.
    pub const PARAMS: Register<Cbv> = Register::new("params", 0);
    /// The skinned vertices.
    pub const SRC: Register<Srv> = Register::new("src", 0);
    /// The joint palette.
    pub const PALETTE: Register<Srv> = Register::new("palette", 1);
    /// The morph deltas.
    pub const MORPH_DATA: Register<Srv> = Register::new("morph_data", 2);
    /// The morph weights.
    pub const MORPH_WEIGHTS: Register<Srv> = Register::new("morph_weights", 3);
    /// The deformed vertices.
    pub const DST: Register<Uav> = Register::new("dst", 0);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot};

    /// The skinned vertices.
    pub const SRC: Slot<Buffer> = Slot::new("src", 0);
    /// The deformed vertices.
    pub const DST: Slot<Buffer> = Slot::new("dst", 1);
    /// The joint palette.
    pub const PALETTE: Slot<Buffer> = Slot::new("palette", 2);
    /// The dispatch's parameters.
    pub const PARAMS: Slot<Buffer> = Slot::new("params", 3);
    /// The morph deltas.
    pub const MORPH_DATA: Slot<Buffer> = Slot::new("morph_data", 4);
    /// The morph weights.
    pub const MORPH_WEIGHTS: Slot<Buffer> = Slot::new("morph_weights", 5);
}
