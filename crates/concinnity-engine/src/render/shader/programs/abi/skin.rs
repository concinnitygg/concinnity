//! The ray-tracing skin dispatch's bindings.

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
