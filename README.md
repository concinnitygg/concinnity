# Concinnity

[![Build Status][gh-img]][gh-checks]
[![codecov.io][codecov-img]][codecov-link]

[gh-img]: https://img.shields.io/github/actions/workflow/status/concinnitygg/concinnity/ci.yml?branch=main
[gh-checks]: https://github.com/concinnitygg/concinnity/actions/workflows/ci.yml?query=branch%3Amain
[codecov-img]: https://img.shields.io/codecov/c/github/concinnitygg/concinnity?logo=codecov
[codecov-link]: https://codecov.io/gh/concinnitygg/concinnity

## Overview

Concinnity is an asset-driven 3D rendering engine. Runtime behavior is defined
declaratively through a static set of assets compiled into binary blobs. There
are no scripting languages (yet); behavior emerges entirely from asset
composition.

## Installation

This project is in **early development** and no releases are available yet.
You can install it manually from this repo:

```sh
cargo install --path crates/concinnity-cli
```

See [building.md](crates/concinnity-docs/public/development/building.md) for
more build options.

## Quick Start

Since this project is in early development and CLI commands are bound to change,
there currently isn't any CLI documentation. Run `concinnity help` for a list of
supported commands.

Check out the [asset documentation](crates/concinnity-docs/public/assets/index.md) for all
supported asset types and fields.

This project currently has one example, which uses the
[Amazon Lumberyard Bistro](examples/bistro/README.md) assets and can be run
with `cargo`:

```sh
cargo run -p bistro --release
```

## Crates

See [crates/README.md](crates/README.md) for details.

## License

See [LICENSE](LICENSE).
