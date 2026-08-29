# Concinnity

[![crates.io][crates-img]][crates-link]
[![docs.rs][docs-img]][docs-link]
[![GitHub License][license-img]][license-link]
[![Build Status][gh-img]][gh-checks]
[![codecov.io][codecov-img]][codecov-link]

Concinnity is an asset-driven world engine. A world is a set of components
describing what exists -- a camera, lights, geometry, UI -- and the engine runs
it. Behaviour is declared as data rather than assembled from calls, so building
an application is mostly a matter of saying what is in it.

This project is in **early development** and may not build reliably depending
on the platform requirements installed. See the [Build Guide][build-guide]
for support options.

It's available as a command-line tool:

```sh
cargo install concinnity --features editor
```

And as a Rust crate:

```rust
use concinnity::components::{GraphicsConfig, TextLabel};
use concinnity::{App, World};

fn main() {
    let mut world = World::new();
    world.add_component(GraphicsConfig::default());
    world.add_component(TextLabel {
        content: "Hello, world!".to_string(),
        centered: true,
        ..Default::default()
    });

    App::from_world(world).run().expect("the app runs");
}
```

Check out the [Rust Documentation][docs-link] to explore the crate.

## Crates

| Crate                  | Kind      |      no_std?       | Role                                                                                                                                                                              |
| ---------------------- | --------- | :----------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `concinnity-dev`       | lib       |                    | Dev tooling: world authoring, the `cn` subcommands, the asset reference generator, bundle packaging, the in-engine editor, the debug server.                                      |
| `concinnity-cook`      | lib       |                    | Asset cook pipeline that bakes an authored world into a blob.                                                                                                                     |
| `concinnity-world`     | lib       |                    | Authored world source, build-only asset schemas, validation, and spec builders.                                                                                                   |
| `concinnity-engine`    | lib       |                    | Runtime engine: ECS schedule, graphics/audio ([kira])/spawn, allocator.                                                                                                           |
| `concinnity-device`    | lib       |                    | GPU backends (Metal/Vulkan/DX) behind a device facade.                                                                                                                            |
| `concinnity-shader`    | lib       |                    | Build-time shader compilers for the backend being built.                                                                                                                          |
| `concinnity-host`      | lib       |                    | Host services: the on-disk state tree, the worker pool, the build-time name interner.                                                                                             |
| `concinnity-core`      | lib       | :white_check_mark: | Asset schemas beside their runtime components, the compute over them, the GPU-free render preparation, the rigid-body simulation, the allocation layer, the headless sim drivers. |
| `concinnity-toolchain` | build-dep |                    | Build-script support: cfgs, SDKs, source hashing, doc extraction.                                                                                                                 |

[kira]: https://docs.rs/kira/latest/kira
[crates-img]: https://img.shields.io/crates/v/concinnity.svg?logo=rust
[crates-link]: https://crates.io/crates/concinnity
[docs-img]: https://img.shields.io/docsrs/concinnity?logo=docsdotrs
[docs-link]: https://docs.rs/concinnity
[gh-img]: https://img.shields.io/github/actions/workflow/status/concinnitygg/concinnity/ci.yml?branch=main
[gh-checks]: https://github.com/concinnitygg/concinnity/actions/workflows/ci.yml?query=branch%3Amain
[codecov-img]: https://img.shields.io/codecov/c/github/concinnitygg/concinnity?logo=codecov
[codecov-link]: https://codecov.io/gh/concinnitygg/concinnity
[license-img]: https://img.shields.io/github/license/concinnitygg/concinnity
[license-link]: https://github.com/concinnitygg/concinnity/blob/main/LICENSE
[build-guide]: https://github.com/concinnitygg/concinnity/blob/main/docs/development/building.md
