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

It's available as a command-line tool:

```sh
cargo install concinnity --features dev
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

Check out the [Rust Documentation][docs-link] to explore the crate, and the
[Asset Reference][asset-ref] for the components a world can hold.

This project is in **early development** and may not build reliably depending
on the platform requirements installed. See the [Build Guide][build-guide]
for all available support options.

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
[asset-ref]: https://github.com/concinnitygg/concinnity/blob/main/docs/assets/index.md
[build-guide]: https://github.com/concinnitygg/concinnity/blob/main/docs/development/building.md
