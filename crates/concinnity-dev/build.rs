//! Resolve the rendering backend once and expose it as a single cfg the crate
//! gates on (`backend_metal` / `backend_dx` / `backend_vk`), via the shared
//! `concinnity-toolchain` helper. The editor / debug modules gate on it.
//!
//! Also stamps the version this build reports: `cn version` and `cn --version`
//! read the commit and date baked in here.
//!
//! This crate is a library, not a final binary, so it does NOT bundle the
//! graphics-SDK runtime DLLs next to an artifact or emit the Agility linker
//! exports; those belong to whichever package owns the final executable: the
//! root package, which builds both bins and the examples.

use concinnity_toolchain::{
    emit_backend_cfg, emit_check_cfgs, emit_version_stamp, setup_graphics_sdks,
};

fn main() {
    emit_check_cfgs();
    setup_graphics_sdks(emit_backend_cfg());
    emit_version_stamp();
}
