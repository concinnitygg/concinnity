//! Resolve the rendering backend cfg this crate gates on (src/cli/export.rs
//! gates on backend_dx / backend_vk).
//!
//! This crate is a library, not a final binary: the `concinnity` binary is a
//! target of the workspace's root package, whose build script bundles the
//! graphics-SDK runtime DLLs and emits the Agility linker exports. The setup
//! here still links the NGX import lib, which this crate's own test binaries
//! need.

use concinnity_toolchain::{emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    setup_graphics_sdks(emit_backend_cfg());
}
