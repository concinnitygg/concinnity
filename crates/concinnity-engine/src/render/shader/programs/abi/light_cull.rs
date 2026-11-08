//! The clustered light cull's bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

    /// The cluster grid's parameters.
    pub const CLUSTER: Register<Cbv> = Register::new("cluster", 0);
    /// The scene's local lights.
    pub const LIGHTS: Register<Srv> = Register::new("lights", 0);
    /// The per-cluster lists the cull writes.
    pub const CLUSTER_LIST: Register<Uav> = Register::new("cluster_list", 0);
    /// The reflection-probe records it bins.
    pub const PROBE_RECORDS: Register<Srv> = Register::new("probe_records", 1);
}
