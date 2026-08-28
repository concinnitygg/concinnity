//! The bench targets are final link targets that pull in the full engine (and
//! with it the renderer), so this package needs the same backend cfg and SDK
//! setup as the other packages with link targets: `cargo:rustc-link-arg` is
//! per-package and does not propagate from concinnity-device, so without the
//! NGX import-lib link here the bench executables fail to resolve the DLSS
//! symbols whenever the SDK cfgs are on. This package builds no bin and no
//! example, so the setup bundles no runtime DLLs beside anything, matching the
//! concinnity-engine rationale.

use concinnity_toolchain::{emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    setup_graphics_sdks(emit_backend_cfg());
}
