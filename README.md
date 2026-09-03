# Concinnity

[![crates.io][crates-img]][crates-link]
[![docs.rs][docs-img]][docs-link]
[![License][license-img]][license-link]
[![deps.rs][deps-img]][deps-link]
[![Build Status][gh-img]][gh-checks]
[![codecov.io][codecov-img]][codecov-link]

Concinnity is an asset-driven world engine. A world is a set of components
describing what exists (a camera, lights, geometry, UI, etc.) and the engine
runs it. Behavior is declared as data rather than assembled from code, so
building an application is mostly a matter of saying what is in it.

## Features

- **Declarative worlds.** Describe what exists and the engine runs it.
  Logic itself is data, executed by an embedded behavior VM.
- **Three native render backends.** DirectX 12, Metal, and Vulkan, driven by
  single-source shaders and a shared render graph.
- **Modern [rendering](#rendering).** Bindless drawing, GPU-driven culling,
  PBR + IBL, clustered lighting, hardware ray tracing, HDR display output,
  and temporal upscaling.
- **Built-in [physics](#physics).** Native rigid-body simulation with
  continuous collision detection, joints, a character controller, and a
  fixed timestep.
- **[Skeletal animation](#skeletal-animation).** State machines, blend
  trees, root motion, and IK on top of blended, cross-faded clips.
- **[3D audio](#audio).** Positional emitters, buses, one-shot cues, and
  raycast occlusion.
- **Data-driven [UI](#ui--input).** Screens, layout containers, text input,
  and keyboard, mouse, and gamepad navigation.
- **[Streaming](#streaming--lod) at scale.** Textures, meshes, and voxel
  chunks stream in and out under a VRAM budget, with automatic LOD.
- **Editor and hot reload.** Edit a running world in-engine and save it
  back; shaders, textures, and geometry reload live.
- **Ship it.** Export a built world as a distributable app; the headless
  `no_std` core runs anywhere.

## Installation

This project is in **early development** and may not build reliably
depending on which platform requirements are installed. See the
[Build Guide][build-guide] for support options.

#### From crates.io

```bash
cargo install concinnity --features editor
```

#### From Source

```bash
git clone https://github.com/concinnitygg/concinnity.git
cd concinnity
cargo build --release
```

#### As a Library

```toml
[dependencies]
concinnity = "0.19"
```

## Quick Start

#### CLI Usage

```bash
concinnity editor
```

This launches an editor UI where you can create your first world.

#### Library Usage

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

This opens a window displaying "Hello, world!" in large, white, centered
text. Resize the window and the text scales with it.

Check out the [Rust Documentation][docs-link] to explore the crate.

## Feature Flags

| Feature   | Default | Description                                       |
| --------- | ------- | ------------------------------------------------- |
| `std`     | ✅      | Standard engine features                          |
| `native`  | ✅      | The render backend the target builds with         |
| `metal`   |         | Metal rendering backend (Apple platforms)         |
| `directx` |         | DirectX 12 rendering backend (Windows)            |
| `vulkan`  |         | Vulkan rendering backend (cross-platform)         |
| `cook`    |         | Build worlds into blobs in process                |
| `player`  |         | The `concinnity-run` player binary                |
| `editor`  |         | The dev CLI and editor (implies `player`, `cook`) |

## CLI Reference

```
Usage: concinnity <COMMAND>

Commands:
  init     Create a new app in the current directory
  new      Create a new app in a new directory
  build    Build a world from worlds/ into binary blobs
  run      Run a compiled world
  debug    Run interpreted directly from a world jsonl file
  editor   Edit a compiled world in-engine with a save-back HUD
  add      Add an asset to the active world
  rm       Remove an asset from the active world by its unique name
  list     List all declared assets
  explain  Print an asset's effective entry from the expanded world
  docs     Regenerate the asset reference pages under docs/assets
  test     Validate a world without building
  export   Package a built world into a distributable app
  mcp      Serve the debug protocol to an MCP client over stdio
  version  Print the version
  help     Print this message or the help of the given subcommand(s)

Options:
  -V, --version  Print the version
  -h, --help     Print help
```

## Engine Capabilities

All capabilities below are supported on every render backend (DirectX 12,
Metal, and Vulkan) unless noted.

#### Rendering

- Physically based shading + IBL (Cook-Torrance)
- Clustered (Forward+) light culling
- Directional, point, spot, and rect area lights
- Cascaded shadow maps with soft PCF
- Hardware ray-traced reflections
- Screen-space global illumination (SSGI)
- Reflection probes + planar reflections
- Bindless main pass
- GPU-driven culling (compute + indirect draw)
- Two-pass Hi-Z occlusion culling
- Parallel command-buffer recording
- Single-source shaders (DXIL / MSL / SPIR-V)
- Projected decals
- Volumetric fog (froxel volume or ray-march)
- Raymarched SDF volumes + SDF shadow casters
- Transparency + ray-traced glass
- GPU particles (compute simulation)
- Water surfaces (Gerstner waves + refraction)
- Voxel worlds (chunked meshing + impostors)

#### Post-processing & display

- HDR off-screen pipeline + MSAA resolve
- ACES tonemapping
- Auto-exposure (histogram + EMA)
- Bloom (prefilter + Karis + tent)
- TAA (jitter + velocity + variance clamp)
- FXAA
- SSAO (GTAO + depth-aware blur)
- SSR (depth / normal / roughness aware)
- 3D-LUT color grading
- Exposure + vignette
- HDR display output (scRGB and PQ)
- Temporal upscaling (FSR 3, MetalFX)
- DLSS / XeSS

#### Skeletal animation

- 4-influence linear blend skinning
- Multi-clip weighted blending + cross-fades
- Animation state machines
- Blend trees
- Root motion
- 2-bone IK (foot pinning)
- Skinned shadows + TAA velocity pre-pass
- glTF mesh, skeleton, and animation import

#### Physics

- Rigid bodies (dynamic, kinematic, static)
- Continuous collision detection (CCD)
- Joints with velocity motors
- Third-person character controller
- Collision layers + filtering
- Sensors, trigger volumes, and contact events
- Heightfield colliders
- Ray + shape queries
- Pickup / carry / throw interactions
- Fixed timestep with render interpolation
- Parallel simulation + sim/render pipelining

#### Audio

- 3D positional emitters + listener
- Audio buses
- One-shot cues + runtime start/stop
- Raycast occlusion
- Physics-driven impact sounds

#### UI & input

- Screen stack + navigation
- Layout containers
- Text rendering (wrapping, bundled fallback)
- Text input fields
- Scroll panels
- Sprites with 9-slice borders
- Built-in main menu + settings screens
- Keyboard, mouse, and gamepad bindings
- Hit testing + interaction events

#### Worlds & behaviors

- Behaviors (logic as data on an embedded VM)
- Runtime variables
- Templates with per-instance overrides
- Runtime spawning + despawning
- Entity hierarchy + transform propagation
- World validation (`concinnity test`)

#### Streaming & LOD

- Texture + normal-map streaming
- Mesh streaming (GPU sub-allocation)
- Voxel chunk streaming
- VRAM budget caps + eviction
- Automatic mesh decimation (QEM)
- Per-draw LOD for static + skinned meshes
- Instanced cluster LOD bucketing
- Distant chunk impostors

#### Tooling

- In-engine editor with save-back
- Hot reload (shaders, textures, geometry, LUTs)
- Per-pass GPU timing
- Frame profiler + stats HUD
- Pipeline (PSO) disk caches
- Headless screenshots
- Crash reporting

#### Import formats

| Category            | Formats                                         |
| ------------------- | ----------------------------------------------- |
| Scenes & models     | `.glb`, `.gltf`, `.fbx`, `.obj`                 |
| Images              | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tga`, `.gif` |
| Compressed textures | `.ktx2`                                         |
| Environment maps    | `.hdr`                                          |
| Audio               | `.ogg`, `.wav`, `.mp3`, `.flac`                 |
| Fonts               | `.ttf`, `.otf`                                  |
| Shaders             | `.glsl`, `.vert`, `.frag`, `.metal`, `.wgsl`    |
| Worlds & data       | `.json`, `.jsonl`                               |
| Text & stories      | `.md`, `.txt`                                   |

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
[build-guide]: https://github.com/concinnitygg/concinnity/blob/main/docs/development/building.md
