//! A planet sea's wake simulation's bindings, one root signature.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

    /// The kernel's parameters.
    pub const PARAMS: Register<Cbv> = Register::new("params", 0);
    /// The wave state.
    pub const STATE: Register<Uav> = Register::new("state", 1);
    /// The transform's scratch.
    pub const WORK: Register<Uav> = Register::new("work", 2);
    /// The wake field written, with the mip read in the register after.
    pub const DST_FIELD: Register<Uav> = Register::new("dst_field", 3);
    /// The mip a downsample reads.
    pub const SRC_FIELD: Register<Srv> = Register::new("src_field", 4);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

    /// The kernel's parameters.
    pub const PARAMS: Slot<Buffer> = Slot::new("params", 0);
    /// The wave state.
    pub const STATE: Slot<Buffer> = Slot::new("state", 1);
    /// The transform's scratch.
    pub const WORK: Slot<Buffer> = Slot::new("work", 2);
    /// The wake field written.
    pub const DST_FIELD: Slot<Texture> = Slot::new("dst_field", 3);
    /// The mip a downsample reads.
    pub const SRC_FIELD: Slot<Texture> = Slot::new("src_field", 4);
}
