# concinnity-core

The `no_std` foundation of the [Concinnity](https://crates.io/crates/concinnity)
engine: math, the allocation layer, thread fan-out, and small primitives.

Every other Concinnity crate depends on this one. It knows nothing about
worlds, components or rendering; those live in `concinnity-engine`, which
builds on it.

## Constraints

- `no_std`: the crate builds without the standard library. Math goes through
  `libm`, not std's inherent float methods.
- No upward edges: nothing here depends on another Concinnity crate.
