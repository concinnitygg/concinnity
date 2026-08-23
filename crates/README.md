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
| `concinnity-font`      | lib       |                    | Build-time glyph atlas rasteriser for cook and the engine binary.    |
| `concinnity-render`    | lib       |                    | GPU-free render preparation.                                         |
| `concinnity-physics`   | lib       |                    | Physics system (wraps [rapier3d]).                                   |
| `concinnity-audio`     | lib       |                    | Audio system (wraps [kira]).                                         |
| `concinnity-store`     | lib       |                    | State tree on disk: paths, source lookup, blob reads.                |
| `concinnity-cpu`       | lib       |                    | CPU compute over the vocabulary: payload codecs, geometry, kernels.  |
| `concinnity-core`      | lib       | :white_check_mark: | Runtime vocabulary: GPU layouts, ECS components, world data, settings. |
| `concinnity-blob`      | lib       | :white_check_mark: | Packed asset blob format; `write` feature gated to cook.             |
| `concinnity-asset`     | lib       | :white_check_mark: | User-facing asset schema (the single home for asset types).          |
| `concinnity-eas`       | lib       | :white_check_mark: | Entity/archetype storage backing the ECS.                            |
| `concinnity-memory`    | lib       | :white_check_mark: | Allocation layer: tracking allocator, tagged budgets, arenas, pools. |
| `concinnity-docs`      | lib       | :white_check_mark: | Asset reference, extracted at build time and embedded.               |
| `concinnity-toolchain` | build-dep |                    | Build-script support: cfgs, SDKs, source hashing, doc extraction.    |

[kira]: https://docs.rs/kira/latest/kira
[rapier3d]: https://docs.rs/rapier3d/latest/rapier3d

## Linkage

```mermaid
block
columns 9

runtime
space:3
cli
space:4

space:9

space:7
editor
space:1

space:9

space:3
engine
space:1
cook
space:2
font

space:9

docs
space:6
device
space:1

space:9

render
space:1
physics
space:1
world
space:4

space:9

audio
space:5
cpu
space:1
store

space:9

space:4
core
space:4

space:9

space:8
blob

space:9

space:3
eas
memory
asset
space:3

runtime --> engine

cli --> cook
cli --> editor
cli --> engine
cli --> world
cli --> docs

cpu --> core

core --> eas
core --> asset
core --> blob
core --> memory

store --> core
store --> blob

engine --> audio
engine --> core
engine --> cpu
engine --> store
engine --> device
engine --> physics
engine --> render

editor --> engine
editor --> device
editor --> store
editor --> cook
editor --> core
editor --> cpu
editor --> world

cook --> blob
cook --> core
cook --> cpu
cook --> font
cook --> store
cook --> world
cook --> docs

docs --> world

device --> core
device --> cpu
device --> render
device --> store

audio --> core

physics --> core
physics --> cpu

render --> core
render --> cpu

blob --> asset

world --> core
world --> cpu
world --> store
```
