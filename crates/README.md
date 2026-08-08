# Crates

| Crate                  | Kind      |      no_std?       | Role                                                                 |
| ---------------------- | --------- | :----------------: | -------------------------------------------------------------------- |
| `concinnity-cli`       | bin       |                    | `concinnity` executable: build, run, add, export, debug.             |
| `concinnity-runtime`   | bin       |                    | Standalone runtime player for a cooked world.                        |
| `concinnity-editor`    | lib       |                    | In-engine world editor: live preview, draggable panels, hot-reload.  |
| `concinnity-cook`      | lib       |                    | Asset cook pipeline that bakes an authored world into a blob.        |
| `concinnity-world`     | lib       |                    | Authored world source, args schema, validation, and spec builders.   |
| `concinnity-engine`    | lib       |                    | Runtime engine: ECS schedule, graphics/spawn/streaming, allocator.   |
| `concinnity-device`    | lib       |                    | GPU backends (Metal/Vulkan/DX) behind a device facade.               |
| `concinnity-shader`    | lib       |                    | Build-time shader compilers for the backend being built.             |
| `concinnity-render`    | lib       |                    | GPU-free render preparation.                                         |
| `concinnity-physics`   | lib       |                    | Physics system (wraps [rapier3d]).                                   |
| `concinnity-audio`     | lib       |                    | Audio system (wraps [kira]).                                         |
| `concinnity-store`     | lib       |                    | State tree on disk: paths, source lookup, blob reads.                |
| `concinnity-core`      | lib       |                    | CPU compute over the runtime vocabulary: skinning, LOD, raster, IBL. |
| `concinnity-types`     | lib       | :white_check_mark: | Runtime vocabulary: GPU layouts, ECS components, registry, settings. |
| `concinnity-blob`      | lib       | :white_check_mark: | Packed asset blob format; `write` feature gated to cook.             |
| `concinnity-asset`     | lib       | :white_check_mark: | User-facing asset schema (the single home for asset types).          |
| `concinnity-eas`       | lib       | :white_check_mark: | Entity/archetype storage backing the ECS.                            |
| `concinnity-memory`    | lib       | :white_check_mark: | Allocation layer: tracking allocator, tagged budgets, arenas, pools. |
| `concinnity-docs`      | lib       | :white_check_mark: | Asset reference, extracted at build time and embedded.               |
| `concinnity-toolchain` | build-dep |                    | Build-time codegen for the binary and graphics crates.               |

[kira]: https://docs.rs/kira/latest/kira
[rapier3d]: https://docs.rs/rapier3d/latest/rapier3d

## Linkage

```mermaid
block
columns 7

space:2
cli
space:1
runtime
space:2

space:7

space:1
editor
space:5

space:7

space:5
cook
space:1

space:7

space:1
engine
space:1
docs
space:3

space:7

device
space:5
world

space:7

space:3
store
space:2
render

space:7

audio
space:5
physics

space:7

space:3
core
space:3

space:7

space:3
types
space:3

space:7

space:4
blob
space:2

space:7

space:1
memory
eas
asset
space:3

runtime --> engine

cli --> cook
cli --> editor
cli --> engine
cli --> world
cli --> docs

core --> types

types --> eas
types --> asset
types --> blob
types --> memory

store --> core
store --> blob

engine --> audio
engine --> core
engine --> store
engine --> device
engine --> physics
engine --> render

editor --> engine
editor --> device
editor --> store
editor --> cook
editor --> core
editor --> world

cook --> blob
cook --> core
cook --> store
cook --> world
cook --> docs

docs --> world

device --> core
device --> render
device --> store

audio --> core

physics --> core

render --> core

blob --> asset

world --> core
world --> store
```
