//! The model-history snapshot's bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

    /// The snapshot's parameters, as root constants.
    pub const PARAMS: Register<Cbv> = Register::new("params", 0);
    /// The frame's records.
    pub const OBJECTS: Register<Srv> = Register::new("objects", 0);
    /// The ring slot the snapshot fills.
    pub const HISTORY: Register<Uav> = Register::new("history", 0);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot};

    /// The snapshot's parameters.
    pub const PARAMS: Slot<Buffer> = Slot::new("params", 0);
    /// The frame's records.
    pub const OBJECTS: Slot<Buffer> = Slot::new("objects", 1);
    /// The ring slot the snapshot fills.
    pub const HISTORY: Slot<Buffer> = Slot::new("history", 2);
}
