# concinnity-engine

The `no_std` engine of [Concinnity](https://crates.io/crates/concinnity):
everything that runs a world short of the machine it runs on.

The ECS storage and scheduler, the component types and the registry built
from them, every system (physics, animation, behavior, audio mixing, UI,
spawning, settings), the render preparation and render graph the device
backends draw from, the bake kernels and payload codecs, and the `.cnb` blob
container format a cooked world travels in. It builds on `concinnity-core`,
the shared foundation; `concinnity-runtime` is the std host that runs it.

## Constraints

- `no_std`: the crate builds without the standard library, so the engine
  runs anywhere a host can lend it threads and I/O. Math goes through
  `libm`, not std's inherent float methods.
- No build-side edge: the asset compile pipeline lives in
  `concinnity-cook`, which depends on this crate; this crate never depends
  on it. A shipped game links the engine without any importers or compilers.

## Terminology

- **Component**: one typed entry in a world; both the authored schema (what
  a `world.jsonl` declares) and the runtime data it bakes into.
- **Bake**: turning authored data into the runtime form; the decoders and
  shared payload types live here under `bake`.
- **Blob (`.cnb`)**: the compiled binary container the runtime loads.
