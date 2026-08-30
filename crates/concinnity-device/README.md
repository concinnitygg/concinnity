# concinnity-device

GPU backends for the [Concinnity](https://crates.io/crates/concinnity)
engine: the hardware-facing renderers, Metal (macOS), DirectX 12 (Windows),
and Vulkan (Windows/Linux), plus the shared native window and input layers.

Exactly one backend compiles per build; the build script resolves the
target (and the optional `vulkan` feature) into a single `backend_*` cfg.
The engine drives a backend through the `RenderBackend`/`SceneControl`
trait seam defined in `concinnity-core`'s `render` module, obtained from
`init_backend` as a `Box<dyn RenderBackend>`; no consumer names a concrete
context type.

## Constraints

- Owns GPU submission and windowing only: no gameplay, ECS runtime, audio,
  or physics.
- The GPU-free render-prep (culling, draw building) lives in
  `concinnity-core`; this crate consumes its output.
- macOS Vulkan runs over the MoltenVK ICD and is a testing path, not a
  shipping one.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
