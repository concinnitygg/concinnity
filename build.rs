//! This package's lib test target and the examples under examples/ are final
//! links that pull in the runtime's DLSS modules. Those are
//! `#[cfg(ngx_sdk_bundled)]`, so when that cfg is on their NGX symbols have to
//! resolve here too -- and the NGX link directive is scoped to the package that
//! emits it (see `dlss_directives` in concinnity-toolchain), so every package
//! linking the DLSS code emits its own.
//!
//! The examples are real binaries a user runs, so the graphics-SDK runtime DLLs
//! are bundled beside them. `BinaryTargets::Examples` is what puts those in
//! `target/<profile>/examples/` rather than the profile directory, and scopes
//! the Agility linker exports to the example targets: Cargo rejects the `-bins`
//! form outright from a package that, like this one, builds no bin.
//!
//! No source in this package gates on a backend or SDK cfg -- the export statics
//! those Agility arguments resolve against live in concinnity-device -- so the
//! backend is resolved without emitting a cfg, and no check-cfg list is needed.

use concinnity_toolchain::{BinaryTargets, backend_from_cargo, setup_graphics_sdks};

fn main() {
    setup_graphics_sdks(backend_from_cargo(), BinaryTargets::Examples);
}
