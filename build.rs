//! This package's binaries, its lib test target and the examples under
//! examples/ are final links that pull in the runtime's DLSS modules. Those are
//! `#[cfg(ngx_sdk_bundled)]`, so when that cfg is on their NGX symbols have to
//! resolve here too -- and the NGX link directive is scoped to the package that
//! emits it (see `dlss_directives` in concinnity-toolchain), so every package
//! linking the DLSS code emits its own.
//!
//! The binaries and the examples are what a user runs, so the graphics-SDK
//! runtime DLLs are bundled beside them. Cargo places each target kind in its
//! own directory and scopes a linker argument by kind, with no key covering
//! both, which is why the setup takes them as a pair: bins land in
//! `target/<profile>/` and examples in `target/<profile>/examples/`.
//!
//! The backend cfg is emitted because `src/bin/concinnity-run` gates its
//! platform stamp on it, and the check-cfg list with it.
//!
//! All of that serves targets only this package builds, so it rides on
//! `player` -- which `dev` enables. Without it there is nothing here to link and
//! the script compiles to an empty `main`, so a consumer of the library builds
//! no build dependency at all. On Windows with the Streamline SDK installed,
//! that also means the lib's own test target and the `cube` example only resolve
//! their NGX symbols under that feature: use the pair the CI gates use
//! (`--features concinnity/cook,concinnity/dev`).

#[cfg(feature = "player")]
fn main() {
    use concinnity_toolchain::{
        BinaryTargets, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks,
    };

    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, &[BinaryTargets::Bins, BinaryTargets::Examples]);
}

#[cfg(not(feature = "player"))]
fn main() {}
