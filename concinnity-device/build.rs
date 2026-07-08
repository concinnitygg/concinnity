// build.rs
//
// The device-backend crate (rlib). Two jobs, both delegated to the shared
// concinnity-toolchain build helper:
//
// 1. Resolve the rendering backend once and expose it as a single cfg the crate
//    gates on (backend_metal / backend_dx / backend_vk).
//
// 2. Detect the optional upscaler SDKs and emit the cfgs the backends gate on.
//    This crate produces only an rlib (consumed by the client) plus its own test
//    binaries, so it does NOT bundle runtime DLLs next to a binary (that belongs
//    to whichever package owns the final artifact): SdkOptions { bundle_dlls: false }.

use concinnity_toolchain::{SdkOptions, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, SdkOptions { bundle_dlls: false });
}
