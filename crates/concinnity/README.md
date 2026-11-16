# concinnity

[![crates.io][crates-img]][crates-link]
[![docs.rs][docs-img]][docs-link]
[![License][license-img]][license-link]

The [Concinnity](https://github.com/concinnitygg/concinnity) SDK: an
asset-driven world engine, the cook that compiles authored assets into a
world, the `concinnity` CLI and editor, and the `concinnity-player` binary that
plays a compiled world.

```bash
cargo install --locked concinnity
```

installs both binaries. An application links the same crate:

```toml
[dependencies]
concinnity = "0.20"
```

```rust,no_run
use concinnity::components::TextLabel;
use concinnity::{App, World};

let mut world = World::new();
world.add_component(TextLabel {
    content: "Hello, world!".to_string(),
    centered: true,
    ..Default::default()
});

App::from_world(world).run().expect("the app runs");
```

This opens a window displaying "Hello, world!" in large, white, centered text.

## Features

| Feature  | Default | Description                                       |
| -------- | ------- | ------------------------------------------------- |
| `std`    | ✅      | The engine and the target's render backend        |
| `player` | ✅      | The `concinnity-player` binary                    |
| `cook`   | ✅      | Build worlds into blobs in process                |
| `editor` | ✅      | The dev CLI and editor (implies `player`, `cook`) |
| `vulkan` |         | Vulkan beside Metal or DirectX 12                 |

A build that wants a smaller graph turns the defaults off and names what it
keeps; see the [Build Guide][build-guide].

[crates-img]: https://img.shields.io/crates/v/concinnity.svg?logo=rust
[crates-link]: https://crates.io/crates/concinnity
[docs-img]: https://img.shields.io/docsrs/concinnity?logo=docsdotrs
[docs-link]: https://docs.rs/concinnity
[license-img]: https://img.shields.io/crates/l/concinnity.svg
[license-link]: https://github.com/concinnitygg/concinnity/blob/main/LICENSE
[build-guide]: https://github.com/concinnitygg/concinnity/blob/main/docs/building.md
