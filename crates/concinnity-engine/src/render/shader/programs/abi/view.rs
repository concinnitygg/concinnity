//! The view samples' bindings: the depth a coarse grid of cells is sampled
//! from, and the cells the CPU reads back.

use super::group::ThreadGroup;

/// The group that samples one cell.
pub const SAMPLE_GROUP: ThreadGroup = ThreadGroup::new(16, 16, 1);

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

    /// The pass's parameters, as root constants.
    pub const PARAMS: Register<Cbv> = Register::new("params", 0);
    /// The Hi-Z pyramid's first level.
    pub const DEPTH: Register<Srv> = Register::new("depth", 0);
    /// Each cell's samples.
    pub const CELLS: Register<Uav> = Register::new("cells", 0);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

    /// The Hi-Z pyramid's first level.
    pub const DEPTH: Slot<Texture> = Slot::new("depth", 0);
    /// Each cell's samples.
    pub const CELLS: Slot<Buffer> = Slot::new("cells", 0);
    /// The pass's parameters.
    pub const PARAMS: Slot<Buffer> = Slot::new("params", 1);
}
