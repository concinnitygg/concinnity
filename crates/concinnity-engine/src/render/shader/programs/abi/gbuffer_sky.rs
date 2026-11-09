//! The G-buffer sky fill's bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register};

    /// The pre-pass's view block.
    pub const GBUFFER_VIEW: Register<Cbv> = Register::new("gb_view", 1);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot};

    /// The pre-pass's view block.
    pub const GBUFFER_VIEW: Slot<Buffer> = Slot::new("gb_view", 0);
}
