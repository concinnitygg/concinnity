# concinnity-toolchain

Build-script support for the
[Concinnity](https://crates.io/crates/concinnity) workspace: backend cfg
resolution and graphics SDK setup.

Consumed only under `[build-dependencies]`, never linked into a shipped
binary. It has two responsibilities, shared by every build script that
produces a Concinnity binary:

1. Resolve the rendering backend once and emit it as a single cfg
   (`backend_metal` / `backend_dx` / `backend_vk`) the source gates on.
2. Detect the optional graphics SDKs (Agility, FidelityFX, XeSS, NGX, DXC)
   and emit the cfgs the renderer gates on; for packages that produce final
   binaries it also bundles the runtime DLLs next to the executable and
   links the NGX import library.

The entry points emit `cargo::` directives on stdout, which Cargo
attributes to whichever package's build script called in; that is what lets
an example binary pick up the same SDK setup as the CLI without duplicating
logic. Metal shader precompilation and source hashing live here too.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
