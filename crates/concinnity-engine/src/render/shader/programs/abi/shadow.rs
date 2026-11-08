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
