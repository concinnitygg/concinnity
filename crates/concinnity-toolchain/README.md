# concinnity-toolchain

Build-script support for the
[Concinnity](https://crates.io/crates/concinnity) workspace: backend cfg
resolution, source hashing and the version stamp.

Consumed only under `[build-dependencies]`, never linked into a shipped
binary. Every build script that compiles the renderer resolves the rendering
backends the build compiles in and emits one cfg per backend
(`backend_metal` / `backend_dx` / `backend_vk`) the source gates on:

```rust
// build.rs
fn main() {
    concinnity_toolchain::setup_graphics_backends();
}
```

The native backend is the one the target renders with, and the `vulkan`
feature adds Vulkan beside it, matching what `concinnity` resolves. A build
script cannot see the features its
dependencies were built with, so a package that took `concinnity`'s `vulkan`
feature says so by carrying a `vulkan` feature of its own.

No graphics SDK is read here: the upscalers and the Agility D3D12 runtime are
loaded when an app runs, from the libraries `concinnity vendor` stages.
