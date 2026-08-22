// build.rs
//
// This crate owns the final `concinnity` binary, so it resolves the rendering
// backend cfg (src/cli/export.rs gates on backend_dx / backend_vk) and bundles
// the graphics-SDK runtime DLLs next to the artifact plus emits the Agility
// linker exports that pull the export statics out of concinnity-device
// (`BinaryTargets::Bins`). The command logic it wraps lives in
// concinnity-editor

use concinnity_toolchain::{BinaryTargets, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, BinaryTargets::Bins);
}
