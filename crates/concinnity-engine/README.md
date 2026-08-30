# concinnity-engine

Runtime engine for [Concinnity](https://crates.io/crates/concinnity): the
world loop, the ECS schedule, the renderer driver, audio, and physics.

This crate is what runs a compiled world. It loads `.cnb` blobs, steps
behavior, physics, and animation on a fixed timestep, and drives rendering
through a `Box<dyn RenderBackend>` from `concinnity-device`, never naming a
concrete backend. The GPU-free render-prep lives in `concinnity-core`'s
`render` module.

## Constraints

- Runtime only: depends on `concinnity-core`, `concinnity-host`, and
  `concinnity-device`. No `concinnity-cook`, no importers, no image
  decoders; a shipped game plays compiled blobs and never sees authored
  input.
- The dev tooling (`concinnity-dev`) drives this crate's `App` and renderer
  through the public API widened here; internals stay `pub(crate)` unless
  the editor specifically needs them.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
