//! The line pass's bindings.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv};

    /// The view block.
    pub const VIEW: Register<Cbv> = Register::new("view", 0);
    /// The scene depth the lines test against.
    pub const SCENE_DEPTH: Register<Srv> = Register::new("scene_depth", 0);
}
