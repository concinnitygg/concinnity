//! The line pass's bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv};

    /// The view block.
    pub const VIEW: Register<Cbv> = Register::new("view", 0);
    /// The scene depth the lines test against.
    pub const SCENE_DEPTH: Register<Srv> = Register::new("scene_depth", 0);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

    /// The view block.
    pub const VIEW: Slot<Buffer> = Slot::new("view", 0);
    /// The scene depth the lines test against.
    pub const SCENE_DEPTH: Slot<Texture> = Slot::new("scene_depth", 0);
}
