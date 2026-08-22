# Concinnity

[![crates.io][crates-img]][crates-link]
[![docs.rs][docs-img]][docs-link]
[![Build Status][gh-img]][gh-checks]
[![codecov.io][codecov-img]][codecov-link]

[crates-img]: https://img.shields.io/crates/v/concinnity.svg?logo=rust
[crates-link]: https://crates.io/crates/concinnity
[docs-img]: https://img.shields.io/docsrs/concinnity?logo=docsdotrs
[docs-link]: https://docs.rs/concinnity
[gh-img]: https://img.shields.io/github/actions/workflow/status/concinnitygg/concinnity/ci.yml?branch=main
[gh-checks]: https://github.com/concinnitygg/concinnity/actions/workflows/ci.yml?query=branch%3Amain
[codecov-img]: https://img.shields.io/codecov/c/github/concinnitygg/concinnity?logo=codecov
[codecov-link]: https://codecov.io/gh/concinnitygg/concinnity

## Overview

Concinnity is an asset-driven 3D rendering engine. Runtime behavior is defined
declaratively through a static set of assets compiled into binary blobs. There
are no scripting languages (yet); behavior emerges entirely from asset
composition.

```rust
use concinnity::assets::{GraphicsConfig, TextLabel};
use concinnity::{App, World};

fn main() -> std::io::Result<()> {
    let mut world = World::new();
    world.add_component(GraphicsConfig::default());
    world.add_component(TextLabel {
        content: "Hello, world!".to_string(),
        ..Default::default()
    });

    App::from_world(world).run()
}
```

Check out the [Asset Documentation](https://github.com/concinnitygg/concinnity/blob/main/docs/assets/index.md) for all
supported asset types and fields.

## Installation

This project is in **early development** and no releases are available yet.

The [Build Guide](https://github.com/concinnitygg/concinnity/blob/main/docs/development/building.md) covers how to manually
build the project.

## License

See [LICENSE](https://github.com/concinnitygg/concinnity/blob/main/LICENSE).
