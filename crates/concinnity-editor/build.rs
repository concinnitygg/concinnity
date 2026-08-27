//! Resolve the rendering backend once and expose it as a single cfg the crate
//! gates on (`backend_metal` / `backend_dx` / `backend_vk`), via the shared
//! `concinnity-toolchain` helper. The `#[cfg(backend_metal)] shader_reflect`
//! module gates on it, as do the editor / debug modules.
//!
//! This crate is a library, not a final binary, so it does NOT bundle the
//! graphics-SDK runtime DLLs next to an artifact or emit the Agility linker
//! exports (`BinaryTargets::None`); those belong to whichever package owns the
//! final executable: the root package, which builds both bins and the
//! examples.

use concinnity_toolchain::{BinaryTargets, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, &[BinaryTargets::None]);
}
