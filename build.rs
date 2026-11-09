//! Runs `concinnity_toolchain::setup_graphics_backends` under `std`: the NGX link
//! directives, the SDK runtime DLLs per binary kind, and the backend cfgs. Also
//! hands the bins `CN_RUNTIME_PLATFORMS`, the comma-separated platform keys of
//! every compiled backend, which `concinnity-run` stamps into its binary.
//! Without `std` `main` is empty. The rationale lives on
//! `setup_graphics_backends`.

#[cfg(feature = "std")]
fn main() {
    let platforms: Vec<&str> = concinnity_toolchain::setup_graphics_backends()
        .into_iter()
        .map(|backend| backend.platform().key())
        .collect();
    println!(
        "cargo::rustc-env=CN_RUNTIME_PLATFORMS={}",
        platforms.join(",")
    );
}

#[cfg(not(feature = "std"))]
fn main() {}
