//! concinnity-runtime is a final binary that links the runtime crate, so it runs
//! the same graphics-SDK setup the CLI and example binaries do: emit the backend
//! cfg (main.rs gates its backend stamp on it), link the NGX import lib, emit the
//! Agility linker exports, and bundle the runtime DLLs next to the .exe.

use concinnity_toolchain::{BinaryTargets, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, BinaryTargets::Bins);
}
