//! Runs `concinnity_toolchain::setup_graphics_backend` under `std`: the NGX link
//! directives, the SDK runtime DLLs per binary kind, and the backend cfg that
//! `concinnity-run` gates its platform stamp on. Without `std` `main` is empty. The
//! rationale lives on `setup_graphics_backend` and `setup_graphics_sdks_for_consumer`.

#[cfg(feature = "std")]
fn main() {
    concinnity_toolchain::setup_graphics_backend();
}

#[cfg(not(feature = "std"))]
fn main() {}
