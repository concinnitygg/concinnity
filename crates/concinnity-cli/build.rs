// build.rs
//
// This crate owns the final `concinnity` binary, so it resolves the rendering
// backend cfg (src/main.rs gates the Agility SDK export statics on backend_dx)
// and bundles the graphics-SDK runtime DLLs next to the artifact plus emits the
// Agility linker exports (`SdkOptions { bundle_dlls: true }`). The command logic
// it wraps lives in concinnity-editor

use concinnity_toolchain::{SdkOptions, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, SdkOptions { bundle_dlls: true });
}
