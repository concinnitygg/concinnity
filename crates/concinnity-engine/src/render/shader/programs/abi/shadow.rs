//! The GPU-driven shadow pass's bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv};

    /// The per-draw object id, a root constant the cull's commands set.
    pub const OBJECT_ID: Register<Cbv> = Register::new("objid_cb", 0);
    /// The cascaded shadow block.
    pub const SHADOW: Register<Cbv> = Register::new("shadow_cb", 1);
    /// The view being drawn, as root constants.
    pub const PUSH: Register<Cbv> = Register::new("push", 2);
    /// The frame's records.
    pub const OBJECTS: Register<Srv> = Register::new("objects", 0);
    /// The shared vertex stream.
    pub const MESH_VERTICES: Register<Srv> = Register::new("mesh_vertices", 1);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot};

    /// The cascaded shadow block.
    pub const SHADOW: Slot<Buffer> = Slot::new("shadow_cb", 0);
    /// The shared vertex stream.
    pub const MESH_VERTICES: Slot<Buffer> = Slot::new("mesh_vertices", 2);
    /// The cascade being drawn.
    pub const CASCADE: Slot<Buffer> = Slot::new("cascade_cb", 7);
    /// The frame's records.
    pub const OBJECTS: Slot<Buffer> = Slot::new("objects", 9);
}
