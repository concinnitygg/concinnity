// build.rs
//
// concinnity-runtime is a final binary that links the runtime crate, so it runs
// the same graphics-SDK setup the editor and example binaries do: emit the
// backend cfg (main.rs gates the Agility SDK export statics on backend_dx),
// link the NGX import lib, and bundle the runtime DLLs next to the .exe.

use concinnity_toolchain::{SdkOptions, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, SdkOptions { bundle_dlls: true });
}
