# Build Guide

The native rendering backend follows the target, and the `vulkan` feature adds
Vulkan beside it:

| Platform | Backend    | Notes                                                |
| -------- | ---------- | ---------------------------------------------------- |
| macOS    | Metal      | `--features vulkan` adds Vulkan as a second choice.  |
| Windows  | DirectX 12 | `--features vulkan` adds Vulkan as a second choice.  |
| Linux    | Vulkan     | Only backend available.                              |
| iOS      | none       | No window layer yet; runs headless.                  |
| Android  | none       | No window layer yet; runs headless.                  |

A build with two backends renders with the native one unless told otherwise.
Use `--graphics-api vulkan` for runtime-related commands.

## Cargo features

The `concinnity` crate is the whole SDK, and its defaults are all of it: the
engine, the cook, the `concinnity-player` binary and the `concinnity` CLI and
editor. `cargo install concinnity` installs both binaries, and an application
depends on it with a plain version:

```toml
concinnity = "0.20"
```

| Feature  | Default | What it adds                                                                                                                                             |
| -------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `std`    | on      | The engine on an operating system, its render backend, heap counting, and the `concinnity-player` binary with the `concinnity::player` module behind it. |
| `cook`   | on      | `concinnity::cook`, which compiles authored assets into a world in process.                                                                              |
| `editor` | on      | The `concinnity` CLI and editor, along with `cook`.                                                                                                      |
| `vulkan` | off     | Vulkan beside Metal or DirectX 12. Linux renders with Vulkan already.                                                                                    |

A build that wants a smaller graph turns the defaults off and names what it
keeps. One that only plays an already-cooked application keeps the runtime and
leaves out the editor and the asset importers `cook` pulls in (glTF, FBX,
PNG and JPEG, KTX2 and Basis textures):

```toml
concinnity = { version = "0.20", default-features = false, features = ["std"] }
```

`vulkan` is mirrored by every crate between the facade and the device
backends, and by the `cn` CLI, so `--features vulkan` means the same thing
whichever of them a build targets. A blob a two-backend build writes carries
shaders for both, so one cook plays on either.

Dropping `std` leaves the `no_std` core: the component vocabulary, a `World`,
and the headless loop that steps it, with no operating system underneath.

## Common prerequisites

Install the [Rust toolchain](https://rustup.rs) on every platform. The workspace
requires Rust **1.100 or newer**:

```sh
rustup update
rustc --version
```

### Third-party dependencies

The build reads no third-party binary that is not a Rust crate. Running an app
from a checkout does: the [shader compiler](#shader-compiler), and on Windows
the [runtime SDKs](#optional-runtime-sdks). Vendor them once per checkout:

```sh
cargo run --bin concinnity -- vendor
```

To see what is vendored:

```sh
cargo run --bin concinnity -- vendor --status
```

### Shader compiler

Every engine shader is HLSL, compiled by the
[DirectX Shader Compiler](https://github.com/microsoft/DirectXShaderCompiler)
(`dxc`) to DXIL for DirectX 12 and to SPIR-V for Vulkan; on macOS the SPIR-V is
translated to MSL by spirv-cross, which the build links, and to a metallib by
the [Metal toolchain](#macos-metal). Nothing is compiled at build time. A run
from a checkout compiles each engine shader the first time it is used and caches
it under the project's `cache/`; `concinnity export` compiles all of them for
the player's graphics APIs and ships them beside its data, so an exported app
needs no compiler. `dxc` is looked for in this order:

1. `dxc/bin/` beside the running executable
2. `<vendor>/dxc-<version>-<os>-<arch>/`, the release `concinnity vendor dxc`
   pins, in the workspace's `vendor/` or the per-user vendor directory
3. `PATH`
4. `$VULKAN_SDK/bin`

Every host compiles with the one pinned release, since the shader cache's key
depends on it: Windows and Linux fetch Microsoft's archive, and macOS builds the
same commit from source, which needs `git`, `cmake` and `ninja`. Without
vendoring, install a
[DXC release](https://github.com/microsoft/DirectXShaderCompiler/releases) and
put its `bin/` on `PATH`.

`cn cook` also needs one to cook assets that declare a `Shader` or an
`SdfVolume`, whose authored HLSL compiles alongside the engine's.

## Third-party environment variables

| Variable     | Read at     | Default | Locates                                                                                                      |
| ------------ | ----------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `VULKAN_SDK` | run         | none    | The Vulkan SDK: `bin/dxc` as the last `dxc` candidate, and the loader, layers and `glslc` for a Vulkan build |

No other SDK is read by the build. The upscalers and the Agility D3D12 runtime
are loaded when an app runs; see
[Optional: runtime SDKs](#optional-runtime-sdks).

The remaining prerequisites are platform specific. Pick the section for your OS
below.

## macOS (Metal)

### Prerequisites

1. Install **Xcode** from the App Store (tested with Xcode 26.2) and select it as
   the active developer directory:

   ```sh
   sudo xcode-select -s /Applications/Xcode.app
   sudo xcodebuild -license accept
   ```

   The Apple frameworks the renderer links against (Metal, AppKit, MetalKit, ...)
   come from the Xcode SDK. The Command Line Tools alone are not sufficient.

2. Install the **Metal toolchain**. Since Xcode 16 it ships as a separate,
   downloadable component rather than being bundled. A run compiles the
   engine's shaders with `xcrun metal` / `xcrun metallib` on first use, and
   export compiles them for the player, so this is required to run or export
   (not to build):

   ```sh
   xcodebuild -downloadComponent MetalToolchain
   ```

   Verify it resolves:

   ```sh
   xcrun metal --version
   ```

3. Build `dxc`, the [shader compiler](#shader-compiler), with
   `concinnity vendor dxc`: DXC publishes no macOS release of its own, so this
   builds the pinned commit (`brew install cmake ninja`).

### Build

```sh
cargo build --release
```

## macOS (Vulkan)

Metal is the default on macOS; the `vulkan` feature adds Vulkan beside it,
which runs over the MoltenVK portability driver. This is a **testing
backend, not a shipping one**: it exists so Vulkan-backend changes can be
exercised on a Mac without a Linux or Windows host.

### Prerequisites

Xcode from the [macOS Metal prerequisites](#macos-metal) is still required to
build, since the build carries Metal beside Vulkan; the Metal toolchain is
needed only to run or export. In addition:

1. Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) from LunarG,
   choosing the **system-wide install** so the loader, the MoltenVK ICD, and the
   validation layers land under `/usr/local`. The engine looks for the loader
   through the dynamic linker first, then at `/usr/local/lib/libvulkan.dylib`
   and Homebrew's prefix, so no `DYLD_LIBRARY_PATH` or `VK_ICD_FILENAMES` needs
   to be set.

   Confirm the loader, the MoltenVK ICD, and the validation layer all resolve:

   ```sh
   vulkaninfo --summary
   ```

2. Shaders still compile with the vendored
   [shader compiler](#shader-compiler), not the SDK's `dxc`.

Windowing and input use the native AppKit window (shared with the Metal
backend), so no windowing-library install is involved and GLFW is not built.
GLFW remains the windowing layer on Linux only.

### Build

```sh
cargo build --release --features vulkan
```

### Known gaps

MoltenVK is not a native Vulkan driver, and the backend is newer here than on
Linux or Windows. Current deltas against the same scene on Metal:

- Directional lighting reads as fully shadowed, so the sun and ambient terms are
  missing; point, spot, and area lights are correct.
- `GraphicsConfig.shadows.map_size = 0` (the 1x1 fallback shadow array) renders
  corrupt geometry rather than an unshadowed scene.
- MoltenVK reports a `maxPerStageDescriptorSamplers` limit of 16, far below what
  desktop drivers allow. The geometry pipeline layout fits by binding fewer
  reflection probes (7 rather than 8); the bindless texture pool has no such
  headroom, so a world with more than one texture still exceeds the limit.
- Ray-traced reflections are unavailable: MoltenVK exposes neither
  `VK_KHR_ray_query` nor `VK_KHR_acceleration_structure`, so the renderer stays
  on screen-space reflections.

## Windows (DirectX 12)

DirectX 12 is the default backend on Windows.

### Prerequisites

1. Install Rust via [rustup](https://rustup.rs). On Windows, Rust uses the
   **MSVC** toolchain by default.

2. Install the **Microsoft C++ build tools** and a recent **Windows SDK**, either
   through Visual Studio 2022 (any edition) or the standalone
   [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/),
   selecting the **Desktop development with C++** workload. This provides the MSVC
   linker plus the Windows SDK. Shaders compile at run time with the pinned
   `dxc` that `concinnity vendor` vendors; see
   [Shader compiler](#shader-compiler).

### Build

```sh
cargo build --release
```

### Optional: runtime SDKs

The DirectX and Vulkan backends can use vendor temporal upscalers (AMD
FidelityFX FSR, Intel XeSS, NVIDIA DLSS through Streamline), and DirectX can run
on Microsoft's D3D12 Agility SDK, which FSR on DirectX needs. None of them is
needed to build: the renderer loads each when an app runs, and an upscaler whose
libraries are absent falls through to the next one, then to native resolution.

`concinnity vendor` (see [Third-party dependencies](#third-party-dependencies))
downloads the pinned releases into the workspace's `vendor/` directory (or a
per-user one outside a workspace) and stages their libraries into
`vendor/runtime-windows-x86_64/`, with their license notices under
`licenses/`. `concinnity run`, `bench` and `edit`, and the examples (through
`concinnity::dev::use_vendored`), load from there, and
`concinnity export` copies it beside the exported player. Streamline's DLSS and
Reflex terms are agreements, which the command shows and asks to accept;
`--accept-licenses` accepts them without asking, and without a terminal to ask
on, the command fails rather than skip them. `concinnity vendor --status`
reports what is vendored.

DLSS runs only on a machine with an NVIDIA GPU. The Agility runtime is used
when its `D3D12/` directory sits in the runtime directory or beside the
executable, and the system's `d3d12.dll` exposes `ID3D12SDKConfiguration1`,
which switching runtimes at launch needs (no Developer Mode is involved);
otherwise the app runs on the OS D3D12 runtime, without FSR on DirectX.

A library beside the executable is found before the runtime directory, so
DLLs an older build left in `target/<profile>/` (or `target/<profile>/examples/`)
take precedence over the staged ones. Delete them, or run `cargo clean`, after
updating an SDK.

## Windows (Vulkan)

Since DirectX is the default on Windows, the `vulkan` feature adds Vulkan beside
it.

### Prerequisites

In addition to the [DirectX prerequisites](#windows-directx-12) above (the MSVC
toolchain is still required):

1. Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) from LunarG. This
   provides the Vulkan loader and validation layers. Its `dxc` is only the
   last [shader compiler](#shader-compiler) candidate, after the vendored one.

Windowing and input use the native Win32 window (shared with the DirectX
backend), so no windowing-library install or runtime DLL is involved.

### Build

```sh
cargo build --release --features vulkan
```

## Linux (Vulkan)

Vulkan is the only backend on Linux. The package names below are for Debian /
Ubuntu; translate them to your distribution's equivalents as needed.

### Prerequisites

1. Install the build toolchain and the system development libraries:

   ```sh
   sudo apt update
   sudo apt install \
     build-essential cmake pkg-config git python3 \
     libasound2-dev \
     libglfw3 libglfw3-dev \
     libwayland-dev libwayland-bin wayland-protocols \
     libx11-dev libxkbcommon-dev libxrandr-dev libxinerama-dev \
     libxcursor-dev libxi-dev libudev-dev libdbus-1-dev
   ```

   - `build-essential`, `cmake`, `git`, `python3` — build GLFW from source when
     no prebuilt library is found.
   - `libasound2-dev` — ALSA, used by the audio backend.
   - `libglfw3` / `libglfw3-dev` and the `libx*` packages — windowing and input
     (GLFW's X11 backend).
   - `libwayland-dev`, `libwayland-bin`, `wayland-protocols` — needed when GLFW
     is built from source (no linkable system GLFW), which compiles both its X11
     and Wayland backends and so requires the Wayland scanner and protocol files.

2. Install the Vulkan loader and validation layers, either from your
   distribution or from the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home):

   ```sh
   sudo apt install libvulkan1 vulkan-validationlayers
   ```

3. Vendor `dxc`, the [shader compiler](#shader-compiler), with
   `concinnity vendor dxc`.

4. To **run** the engine you also need a Vulkan-capable GPU driver (e.g.
   `mesa-vulkan-drivers` for Intel/AMD, or the proprietary NVIDIA driver). The
   `vulkan-tools` package provides `vulkaninfo` to confirm a working ICD:

   ```sh
   sudo apt install vulkan-tools mesa-vulkan-drivers
   vulkaninfo | head
   ```

### Build

```sh
cargo build --release
```

## Testing

The workspace runs its tests under [nextest](https://nexte.st):

```sh
cargo nextest run --workspace
```

Doctests are a separate pass, because `--all-targets` skips them:

```sh
cargo test --doc --workspace
```

### Test tools

The capture, diff and snapshot tools live in the `tools/` submodule, which
nothing in the build needs. A plain clone leaves it empty; clone with
`--recurse-submodules`, or fetch it into an existing checkout:

```sh
git submodule update --init tools
python3 tools/cn.py --help
```

### Discipline scans

Three integration tests in `crates/concinnity-testing/tests/` scan the
workspace's own source rather than exercising it, and each fails the build on
a pattern rather than on a value. They are the one place a test reads beyond
its own package:

| Test                               | What it forbids                                                                                             |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `tests/headless_discipline.rs`     | A test that stands up a window, which hangs on an event loop the harness cannot end.                        |
| `tests/global_state_discipline.rs` | Reaching process-global state (working directory, open project, pending hot-reload flags) without the shared lock. |
| `tests/file_access_discipline.rs`  | Writing to a hand-picked path under the system temporary directory instead of a per-run one.                |
