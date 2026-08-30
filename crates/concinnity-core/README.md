# concinnity-core

Runtime vocabulary for the [Concinnity](https://crates.io/crates/concinnity)
engine: GPU layouts, ECS components, registry, CPU kernels.

This is the crate every other Concinnity crate agrees on and none of them
owns: the backend-agnostic GPU data layouts the CPU and the shaders both
name, the transform and skeleton math those layouts are expressed in, the
ECS storage plus the component definitions and the registry built from them,
the behavior virtual machine that evaluates declarative logic, and the
`.cnb` blob container format a cooked world travels in. Above that
vocabulary sit the kernels that belong to no single consumer: skinning and
pose blending, IK, LOD decimation, rasterization, IBL convolution,
procedural geometry, and the payload codecs.

## Constraints

- `no_std`: the crate builds without the standard library, so the headless
  core runs anywhere. Math goes through `libm`, not std's inherent float
  methods.
- No build-side edge: the asset compile pipeline lives in
  `concinnity-cook`, which depends on this crate; this crate never depends
  on it. A shipped game links core without any importers or compilers.

## Terminology

- **Component**: one typed entry in a world; both the authored schema (what
  a `world.jsonl` declares) and the runtime data it bakes into.
- **Bake**: turning authored data into the runtime form; the decoders and
  shared payload types live here under `bake`/`decode`.
- **Blob (`.cnb`)**: the compiled binary container the runtime loads.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
