# concinnity-derive

Derive macros for the authored asset schemas of the
[Concinnity](https://crates.io/crates/concinnity) engine.

`#[derive(AssetFields)]` reads a schema struct's field types and generates the
authoring tables the build and the editor consult: which fields reference other
assets (and which asset types each may name), and which accept a closed
vocabulary. Deriving them from the types is what keeps the tables complete.

## Constraints

- Internal to the engine: `concinnity-core` and `concinnity-cook` use it, and
  the facade does not re-export it. Components a library declares at runtime
  never reach an authored world, so they have no schema to describe.
- The generated code names `::concinnity_core::` and `::core` paths only, so it
  compiles inside the `no_std` core itself.
