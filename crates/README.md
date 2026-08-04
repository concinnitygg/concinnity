# Crates

| Crate                  | Kind      |      no_std?       | Role                                                                |
| ---------------------- | --------- | :----------------: | ------------------------------------------------------------------- |
| `concinnity-cli`       | bin       |                    | `concinnity` executable: build, run, add, export, debug.            |
| `concinnity-runtime`   | bin       |                    | Standalone runtime player for a cooked world.                       |
| `concinnity-editor`    | lib       |                    | In-engine world editor: live preview, draggable panels, hot-reload. |
| `concinnity-cook`      | lib       |                    | Asset cook pipeline that bakes an authored world into a blob.       |
| `concinnity-world`     | lib       |                    | Authored world source, args schema, validation, and spec builders.  |
| `concinnity-engine`    | lib       |                    | Runtime engine: ECS schedule and graphics/spawn/streaming systems.  |
| `concinnity-device`    | lib       |                    | GPU backends (Metal/Vulkan/DX) behind a device facade.              |
| `concinnity-shader`    | lib       |                    | Build-time shader compilers for the backend being built.            |
| `concinnity-render`    | lib       |                    | GPU-free render preparation.                                        |
| `concinnity-physics`   | lib       |                    | Physics system (rapier3d).                                          |
| `concinnity-audio`     | lib       |                    | Audio system (kira).                                                |
| `concinnity-core`      | lib       |                    | Shared ECS, assets, resources, and math foundation.                 |
| `concinnity-blob`      | lib       | :white_check_mark: | Packed asset blob format; `write` feature gated to cook.            |
| `concinnity-asset`     | lib       | :white_check_mark: | User-facing asset schema (the single home for asset types).         |
| `concinnity-eas`       | lib       | :white_check_mark: | Entity/archetype storage backing the ECS.                           |
| `concinnity-memory`    | lib       | :white_check_mark: | Tracking global allocator and heap stats.                           |
| `concinnity-docs`      | lib       | :white_check_mark: | Asset reference, extracted at build time and embedded.              |
| `concinnity-toolchain` | build-dep |                    | Build-time codegen for the binary and graphics crates.              |

## Linkage

```mermaid
block
columns 6

runtime
space:4
cli

space:6

space:1
engine
space:1
editor
space:2

space:6

render
space:1
device
space:3

space:6

audio
space:3
cook
space:1

space:6

space:5
docs

space:1
physics
core
world
space:2

space:6

eas
asset
space:2
blob
space:1

runtime --> engine

cli --> cook
cli --> editor
cli --> engine
cli --> world
cli --> docs

core --> eas
core --> asset
core --> blob

engine --> audio
engine --> core
engine --> device
engine --> physics
engine --> render

editor --> engine
editor --> device
editor --> cook
editor --> core
editor --> world

cook --> blob
cook --> core
cook --> world
cook --> docs

docs --> world

device --> core
device --> render

audio --> core

physics --> core

render --> core

blob --> asset

world --> core
```
