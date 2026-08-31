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

## Depending on Concinnity

A package that links `concinnity` needs this crate too, because the NGX link
directive is scoped to the package that emits it: when the runtime is built
with DLSS available its upscaler compiles into the rlib, and the binary
linking that rlib has to resolve the `NVSDK_NGX_*` symbols itself. Add the
build dependency and one build script:

```toml
[build-dependencies]
concinnity-toolchain = "0.19"
```

```rust
// build.rs
fn main() {
    concinnity_toolchain::setup_graphics_sdks_for_consumer();
}
```

That links the NGX import library and stages the SDK runtime DLLs next to
the executable, where `LoadLibrary` finds them. A missing SDK is reported as
a `cargo::warning` naming the root it was looked for under, and costs that
upscaler rather than the build.

The backend is resolved as if `native` were on, matching what `concinnity`'s
own defaults give for the target. A build script cannot see the features its
dependencies were built with, so a package that took `concinnity`'s `vulkan`
feature says so by carrying a `vulkan` feature of its own.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
