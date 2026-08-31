//! This package's binaries, its examples and its own test and bench
//! executables are final links that pull in the runtime's DLSS modules. Those
//! are `#[cfg(ngx_sdk_bundled)]`, so when that cfg is on their NGX symbols have
//! to resolve here too -- and the NGX link directive is scoped to the package
//! that emits it (see `dlss_directives` in concinnity-toolchain), so every
//! package linking the DLSS code emits its own.
//!
//! The binaries and the examples are what a user runs, so the graphics-SDK
//! runtime DLLs are bundled beside them. Cargo places each target kind in its
//! own directory and scopes a linker argument by kind, with no key covering
//! both, so the setup runs once per kind: bins land in `target/<profile>/` and
//! examples in `target/<profile>/examples/`. Which kinds this package builds
//! the setup reads off the manifest, which is what keeps it right in the
//! published crate.
//!
//! The backend cfg is emitted because `src/bin/concinnity-run` gates its
//! platform stamp on it, and the check-cfg list with it.
//!
//! All of that exists only once the engine is in the graph, so it rides on
//! `std` -- the feature that pulls concinnity-engine in, and with it the device
//! code the NGX symbols come from. Every feature that selects a backend enables
//! `std`, and a `std` build that selects none resolves no backend, so the setup
//! is inert there. Below `std` there is nothing to link and the script compiles
//! to an empty `main`. concinnity-engine already carries concinnity-toolchain
//! as a build dependency, so gating here on `std` adds no build graph a `std`
//! consumer did not already have.

#[cfg(feature = "std")]
fn main() {
    use concinnity_toolchain::{emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

    emit_check_cfgs();
    setup_graphics_sdks(emit_backend_cfg());
}

#[cfg(not(feature = "std"))]
fn main() {}
