//! Resolve the rendering backends and expose one cfg per backend the crate
//! gates on (`backend_metal` / `backend_dx` / `backend_vk`), via the shared
//! `concinnity-toolchain` helper's `setup_graphics_backends`. The editor / debug
//! modules gate on it.
//!
//! Also stamps the version this build reports: `cn version` and `cn --version`
//! read the commit and date baked in here.

use concinnity_toolchain::{emit_version_stamp, setup_graphics_backends};

fn main() {
    setup_graphics_backends();
    emit_version_stamp();
}
