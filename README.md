# Concinnity

[![crates.io][crates-img]][crates-link]
[![docs.rs][docs-img]][docs-link]
[![License][license-img]][license-link]
[![deps.rs][deps-img]][deps-link]
[![Build Status][gh-img]][gh-checks]
[![codecov.io][codecov-img]][codecov-link]

An agent-friendly graphics application SDK.

## Features

#### Client SDK

- Editor with graphical, command-line, and MCP interfaces
- DirectX 12, Metal, and Vulkan rendering
- DLSS, FSR, XeSS, and MetalFX upscaling

## Installation

This project is in **early development** and may not build reliably
depending on which platform requirements are installed. See the
[Build Guide][build-guide] for support options.

#### From crates.io

```bash
cargo install --locked concinnity
```

#### From Source

```bash
git clone https://github.com/concinnitygg/concinnity.git
cd concinnity
cargo install --locked --path crates/concinnity
```

## Quick Start

```bash
concinnity new my-app
cd my-app
concinnity edit
```

Where to go next:

- [Tutorials][tutorials]
- [CLI Reference][cli-reference]
- [Technical Specification][spec]

[crates-img]: https://img.shields.io/crates/v/concinnity.svg?logo=rust
[crates-link]: https://crates.io/crates/concinnity
[docs-img]: https://img.shields.io/docsrs/concinnity?logo=docsdotrs
[docs-link]: https://docs.rs/concinnity
[gh-img]: https://img.shields.io/github/actions/workflow/status/concinnitygg/concinnity/ci.yml?branch=main
[gh-checks]: https://github.com/concinnitygg/concinnity/actions/workflows/ci.yml?query=branch%3Amain
[deps-img]: https://img.shields.io/deps-rs/concinnity/latest
[deps-link]: https://deps.rs/crate/concinnity
[codecov-img]: https://img.shields.io/codecov/c/github/concinnitygg/concinnity?logo=codecov
[codecov-link]: https://codecov.io/gh/concinnitygg/concinnity
[license-img]: https://img.shields.io/crates/l/concinnity.svg
[license-link]: https://github.com/concinnitygg/concinnity/blob/main/LICENSE
[build-guide]: docs/building.md
[cli-reference]: docs/cli.md
[tutorials]: docs/tutorials/index.md
[spec]: docs/spec.md
