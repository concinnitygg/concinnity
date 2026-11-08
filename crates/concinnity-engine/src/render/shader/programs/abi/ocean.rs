//! The simulated ocean's FFT and mip kernels' bindings, one root signature.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

    /// The kernel's parameters, as root constants.
    pub const PARAMS: Register<Cbv> = Register::new("params", 0);
    /// The displacement a mip reads, with the slopes in the register after.
    pub const SRC_DISPLACEMENT: Register<Srv> = Register::new("src_displacement", 0);
    /// The slopes a mip reads.
    pub const SRC_SLOPES: Register<Srv> = Register::new("src_slopes", 1);
    /// The wave spectrum.
    pub const SPECTRUM: Register<Srv> = Register::new("spectrum", 5);
    /// The displacement written, with the slopes in the register after.
    pub const DST_DISPLACEMENT: Register<Uav> = Register::new("dst_displacement", 2);
    /// The slopes written.
    pub const DST_SLOPES: Register<Uav> = Register::new("dst_slopes", 3);
    /// The row transform's scratch.
    pub const ROWS: Register<Uav> = Register::new("rows", 6);
    /// The foam the columns accumulate.
    pub const FOAM: Register<Uav> = Register::new("foam", 7);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

    /// The kernel's parameters.
    pub const PARAMS: Slot<Buffer> = Slot::new("params", 0);
    /// The wave spectrum.
    pub const SPECTRUM: Slot<Buffer> = Slot::new("spectrum", 5);
    /// The row transform's scratch.
    pub const ROWS: Slot<Buffer> = Slot::new("rows", 6);
    /// The foam the columns accumulate.
    pub const FOAM: Slot<Buffer> = Slot::new("foam", 7);
    /// The displacement a mip reads.
    pub const SRC_DISPLACEMENT: Slot<Texture> = Slot::new("src_displacement", 0);
    /// The slopes a mip reads.
    pub const SRC_SLOPES: Slot<Texture> = Slot::new("src_slopes", 1);
    /// The displacement written.
    pub const DST_DISPLACEMENT: Slot<Texture> = Slot::new("dst_displacement", 2);
    /// The slopes written.
    pub const DST_SLOPES: Slot<Texture> = Slot::new("dst_slopes", 3);
}
