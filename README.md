# Concinnity

[![crates.io][crates-img]][crates-link]
[![docs.rs][docs-img]][docs-link]
[![Build Status][gh-img]][gh-checks]
[![codecov.io][codecov-img]][codecov-link]
[![License][license-img]][license-link]

Concinnity is an asset-driven world application builder and runner.
Runtime behavior is defined declaratively through a static set of assets
compiled into runnable worlds.

```rust
use concinnity::assets::{GraphicsConfig, TextLabel};
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

Check out the [Rust Documentation][docs-link] to explore the features.

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
[license-img]: https://img.shields.io/crates/l/concinnity.svg
[license-link]: https://github.com/concinnitygg/concinnity/blob/main/LICENSE
[build-guide]: https://github.com/concinnitygg/concinnity/blob/main/docs/development/building.md
