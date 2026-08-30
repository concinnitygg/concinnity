# concinnity-slang

The `slangc` invocation shared by
[Concinnity](https://crates.io/crates/concinnity) build scripts and the
renderer.

The engine's single-source shaders compile through the `slangc` binary,
mostly at build time: the device build script emits Metal metallibs, DXIL,
and SPIR-V for the backend it targets. The renderer compiles the rest, such
as a hot-reload edit or a permutation no build-time artifact covers. Every
call site assembles the full source text first, so a compile is a pure
function of that text, the entry list, and the target, which is what lets
the renderer's content-addressed shader cache key it.

## Constraints

- The single invocation is the point: build script and runtime must produce
  byte-identical artifacts, so the flag list exists exactly once, here.
- Sits below `concinnity-toolchain` (which is build-script-only and never
  linked into a shipped binary), holds no policy, and depends on nothing
  but std; a compile is a subprocess, not a linked compiler.
- `slangc` resolves from `PATH`, then `$VULKAN_SDK/bin`. A host without it
  degrades gracefully: the build emits a stub and the renderer compiles at
  startup, reporting a clear error if `slangc` is absent or too old.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
