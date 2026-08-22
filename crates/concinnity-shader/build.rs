//! Resolve the rendering backend once and expose it as a single cfg the crate
//! gates on (backend_metal / backend_dx / backend_vk), so the shader toolchain
//! compiled in matches the backend the client links. No graphics-SDK setup: this
//! crate drives the shader compilers, not the upscalers.

use concinnity_toolchain::{emit_backend_cfg, emit_check_cfgs};

fn main() {
    emit_check_cfgs();
    emit_backend_cfg();
}
