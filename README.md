# Concinnity

[![Concinnity GitHub Actions][gh-image]][gh-checks]
[![codecov.io][codecov-img]][codecov-link]

[gh-image]: https://github.com/concinnitygg/concinnity/actions/workflows/ci.yml/badge.svg?branch=main
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
cargo install --path concinnity-editor
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

| Crate                                         |      no_std?       | Description                                               |
| --------------------------------------------- | :----------------: | --------------------------------------------------------- |
| [concinnity-app](concinnity-app/)             |                    | C-ABI cdylib for the Swift app + world authoring API      |
| [concinnity-client](concinnity-client/)       |                    | Runtime engine: world loop, ECS, renderer, audio, physics |
| [concinnity-cook](concinnity-cook/)           |                    | Asset compile pipeline: world.jsonl + sources -> blobs    |
| [concinnity-core](concinnity-core/)           |                    | Renderer-free foundation: assets, GPU layouts, math       |
| [concinnity-device](concinnity-device/)       |                    | GPU backends: Metal / DirectX 12 / Vulkan (+ Win32)       |
| [concinnity-docs](concinnity-docs/)           |                    | Asset + API reference documentation generator             |
| [concinnity-eas](concinnity-eas/)             |                    | Entity-Asset-System: the engine's generic ECS             |
| [concinnity-editor](concinnity-editor/)       |                    | The `concinnity` dev CLI + in-engine world editor         |
| [concinnity-render](concinnity-render/)       |                    | Backend-agnostic, GPU-free render preparation             |
| [concinnity-runtime](concinnity-runtime/)     |                    | Shipped player binary for a world's compiled blobs        |
| [concinnity-templates](concinnity-templates/) | :white_check_mark: | Engine-owned world/asset templates (static data)          |
| [concinnity-toolchain](concinnity-toolchain/) |                    | Shared build-script support (backend + SDK cfgs)          |

## License

See [LICENSE](LICENSE).
