# Concinnity

![DirectX 12][dx-img]
![Metal][ml-img]
![Vulkan][vk-img]
[![Concinnity GitHub Actions][gh-img]][gh-checks]
[![codecov.io][codecov-img]][codecov-link]

[dx-img]: https://img.shields.io/badge/DirectX%2012-107C10
[ml-img]: https://img.shields.io/badge/Metal-1A1A1A
[vk-img]: https://img.shields.io/badge/Vulkan-A41E22?logo=vulkan&logoColor=white
[gh-img]: https://github.com/concinnitygg/concinnity/actions/workflows/ci.yml/badge.svg?branch=main
[gh-checks]: https://github.com/concinnitygg/concinnity/actions/workflows/ci.yml?query=branch%3Amain
[codecov-img]: https://img.shields.io/codecov/c/github/concinnitygg/concinnity?logo=codecov
[codecov-link]: https://codecov.io/gh/concinnitygg/concinnity

Application software for [`concinnity.gg`](https://concinnity.gg).

## Overview

Concinnity is an asset-driven 3D rendering engine. Runtime behavior is defined
declaratively through a static set of assets compiled into binary blobs. There
are no scripting languages (yet); behavior emerges entirely from asset
composition.

## Installation

This project is in **early development** and no releases are available yet.
For now, you'll need to [build it manually](concinnity-docs/public/development/building.md).

After a successful build, you may optionally install the `concinnity` executable:

```sh
cargo install --path crates/concinnity-cli
concinnity help
```

## Getting Started

Since this project is in early development and CLI commands are bound to change,
there currently isn't any CLI documentation. Run `concinnity help` for a list of
supported commands.

Check out the [asset documentation](concinnity-docs/public/assets/index.md) for all
supported asset types and fields.

This project currently has one example, which uses the
[Amazon Lumberyard Bistro](examples/bistro/README.md) assets and can be run
with `cargo`:

```sh
cargo run -p bistro --release
```

## Crates

| Crate                                                |      no_std?       | Description                                                          |
| ---------------------------------------------------- | :----------------: | -------------------------------------------------------------------- |
| [concinnity-cli](crates/concinnity-cli/)             |                    | The `concinnity` dev CLI binary: command tree + subcommands          |
| [concinnity-editor](crates/concinnity-editor/)       |                    | Dev tooling library: world authoring, in-engine editor, debug server |
| [concinnity-ffi](crates/concinnity-ffi/)             |                    | General-purpose C-ABI cdylib for embedding in a host app             |
| [concinnity-runtime](crates/concinnity-runtime/)     |                    | Shipped player binary for a world's compiled blobs                   |
| [concinnity-engine](crates/concinnity-engine/)       |                    | Runtime engine: world loop, ECS, renderer, audio                     |
| [concinnity-physics](crates/concinnity-physics/)     |                    | Rapier rigid-body simulation: props, joints, character rigs          |
| [concinnity-cook](crates/concinnity-cook/)           |                    | Asset compile pipeline: world.jsonl + sources -> blobs               |
| [concinnity-world](crates/concinnity-world/)         |                    | Build-side world model: authoring vocabulary, parsing, validation    |
| [concinnity-render](crates/concinnity-render/)       |                    | Backend-agnostic, GPU-free render preparation                        |
| [concinnity-device](crates/concinnity-device/)       |                    | GPU backends: Metal / DirectX 12 / Vulkan (+ Win32)                  |
| [concinnity-core](crates/concinnity-core/)           |                    | Renderer-free foundation: assets, GPU layouts, math                  |
| [concinnity-blob](crates/concinnity-blob/)           |                    | The .cnb blob container format (read-only; write is cook-gated)      |
| [concinnity-asset](crates/concinnity-asset/)         | :white_check_mark: | Authored-data schema: asset structs, identity + handle primitives    |
| [concinnity-eas](crates/concinnity-eas/)             |                    | Entity-Asset-System: the engine's generic ECS                        |
| [concinnity-templates](crates/concinnity-templates/) | :white_check_mark: | Engine-owned world/asset templates (static data)                     |
| [concinnity-docs](crates/concinnity-docs/)           |                    | Asset + API reference documentation generator                        |
| [concinnity-toolchain](crates/concinnity-toolchain/) |                    | Shared build-script support (backend + SDK cfgs)                     |

## License

See [LICENSE](LICENSE).
