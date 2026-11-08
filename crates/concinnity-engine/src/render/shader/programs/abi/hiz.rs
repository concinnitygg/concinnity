//! The Hi-Z pyramid's SPD dispatches' bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

    /// The dispatch's parameters, as root constants.
    pub const PARAMS: Register<Cbv> = Register::new("params", 0);
    /// The depth the first dispatch reduces.
    pub const SRC_DEPTH: Register<Srv> = Register::new("src_depth", 0);
    /// The pyramid's mips, one table.
    pub const SPD_MIPS: Register<Uav> = Register::new("spd_mips", 0);
}
