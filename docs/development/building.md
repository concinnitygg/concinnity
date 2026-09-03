# Build Guide

The rendering backend is a Cargo feature. The default build carries `native`,
which is whatever the target renders with:

| Platform | `native` resolves to | Notes                                         |
| -------- | -------------------- | --------------------------------------------- |
| macOS    | Metal                | Build with `--features vulkan` to use Vulkan. |
| Windows  | DirectX 12           | Build with `--features vulkan` to use Vulkan. |
| Linux    | Vulkan               | Only backend available.                       |

## Cargo features

The `concinnity` facade crate, which is what an application depends on, has
these:

| Feature   | Default | What it adds                                                                |
| --------- | ------- | --------------------------------------------------------------------------- |
| `cook`    | off     | `concinnity::cook`, which compiles authored assets into a world in process. |
| `native`  | on      | The backend the target renders with: Metal, DirectX 12, or Vulkan.          |
| `metal`   | off     | The Metal backend, on macOS.                                                |
| `directx` | off     | The DirectX 12 backend, on Windows.                                         |
| `vulkan`  | off     | The Vulkan backend, on any platform that has one.                           |

A default build is the runtime alone. `cook` pulls in the asset importers
(glTF, FBX, textures, fonts) and 43 further dependencies in total, which is
build-time weight an application playing an already-compiled world does not
carry. Authoring through the `cn` CLI needs nothing extra; the feature is for
declaring a world in Rust, as `examples/bistro` does.

A backend feature naming one the target does not have is inert, so a build can
name more than one and get the one that applies; where two do, Vulkan wins.
Exactly one backend compiles into any binary.

The backend features are mirrored by every crate between the facade and the
device backends, and by the `cn` CLI, so `--features vulkan` means the same
thing whichever of them a build targets.

Turning every backend feature off is a supported configuration:

```sh
cargo build --no-default-features --features std
```

That is the CPU-only runtime. No GPU code is in the dependency graph at all,
and every world runs on the headless loop -- the simulation systems stepped on
a fixed virtual timestep with no window and no renderer. It is what a
simulation-only tool, or a build host with no GPU, wants. Dropping `std` as
well leaves the `no_std` core, which runs the same loop with no operating
system underneath.

## Common prerequisites

Install the [Rust toolchain](https://rustup.rs) on every platform. The workspace
uses the **2024 edition**, so Rust **1.85 or newer** is required:

```sh
rustup update
rustc --version
```

### Third-party dependencies

The build uses third-party binaries that are not Rust crates. Fetch them once
per checkout:

```sh
scripts/vendor.py fetch
```

To see what is vendored and what the build would otherwise pick:

```sh
scripts/vendor.py status
```

[Third-party environment variables](#third-party-environment-variables)
lists how to override vendor paths.

## Third-party environment variables

Every third-party toolchain the build reaches for is located the same way: the
environment variable if it is set, then `vendor/`, then nothing (the feature is
skipped and the build says which of the two was missing). Setting one is only
needed for an SDK installed somewhere `vendor.py` did not put it.

### Locating an SDK

| Variable            | Read at     | Default                                                                             | Locates                                                                                                          |
| ------------------- | ----------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `CN_SLANG_SDK`      | build + run | `slang/` beside the executable, then `vendor/`, then `PATH`, then `$VULKAN_SDK/bin` | The Slang shader compiler; expects `bin/slangc` under it                                                         |
| `VULKAN_SDK`        | build + run | none                                                                                | The Vulkan SDK: `bin/slangc` as the last slangc candidate, and the loader, layers and `glslc` for a Vulkan build |
| `CN_AGILITY_SDK`    | build       | `vendor/agility-*`                                                                  | Microsoft's D3D12 Agility SDK                                                                                    |
| `CN_FIDELITYFX_SDK` | build       | `vendor/fidelityfx-*`                                                               | AMD FidelityFX, for FSR 3. The Vulkan runtime prefers `vendor/fidelityfx-vk-*`, which has no variable of its own |
| `CN_XESS_SDK`       | build       | `vendor/xess-*`                                                                     | Intel XeSS                                                                                                       |
| `CN_STREAMLINE_SDK` | build       | `vendor/streamline-*`                                                               | NVIDIA Streamline, for DLSS                                                                                      |
| `CN_DXC_SDK`        | build       | the Windows SDK's `bin`, under `%ProgramFiles(x86)%`                                | A standalone DirectX Shader Compiler, over the Windows SDK's                                                     |
| `SHADERC_LIB_DIR`   | build       | none                                                                                | A prebuilt `shaderc`, skipping its from-source build (read by `shaderc-sys`, not by this workspace)              |

A vendored SDK needs no variable set. There is no third fallback to a stock
install path: these unpack wherever their user puts them, so any hardcoded
location would only be right by accident and would report a path nobody has
instead of saying nothing was found. The engine's
shaders are written once as `.slang` and compiled at build time, so a binary
built with `slangc` present carries them wherever it goes; one built without it
compiles them at renderer init instead and then needs `slangc` on every host
that runs it. Hot-reload (`cn debug`, `cn editor`) and a macOS Vulkan build
recompile shaders at runtime, so both need it present then too. The floor is
release **2026.1** -- earlier ones emit SPIR-V declaring capabilities the shaders
never use, which Vulkan rejects. The Windows Vulkan SDK bundles a new enough
one; the LunarG Linux packages have not.

### Turning a feature off

Each of these defaults to on and is disabled with `=0`. They gate whether the
build links or bundles the SDK at all, so they take effect at build time, not at
launch.

| Variable                | Default | Effect when off                                     |
| ----------------------- | ------- | --------------------------------------------------- |
| `CN_ENABLE_FFX_FSR3`    | on      | No FSR 3; upscaling falls back to native resolution |
| `CN_ENABLE_XESS`        | on      | No XeSS                                             |
| `CN_ENABLE_DLSS`        | on      | No DLSS                                             |
| `CN_ENABLE_DXC`         | on      | DXIL compiles through FXC's shader models only      |
| `CN_ENABLE_AGILITY_SDK` | **off** | Not applicable; this one is opt-_in_, see below     |

The three upscalers are loaded with `LoadLibrary` at runtime and degrade to a
fallback when their DLL is absent, so a build that includes one still runs on a
machine that has none. The Agility SDK is the exception in both directions: it
is off by default, opted into with `CN_ENABLE_AGILITY_SDK=1`, and turning it on
decides where the finished executable can run.

None of these are read by a shipped game. A player resolves its upscaler DLLs
beside its own executable, which is what `cn export` copies them there for.

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
   downloadable component rather than being bundled. The asset compiler invokes
   `xcrun metal` / `xcrun metallib` to compile shaders at build time, so this is
   required:

   ```sh
   xcodebuild -downloadComponent MetalToolchain
   ```

   Verify it resolves:

   ```sh
   xcrun metal --version
   ```

### Build

```sh
cargo build --release
```

## macOS (Vulkan)

Metal is the default on macOS; the `vulkan` feature selects a Vulkan build
instead, which runs over the MoltenVK portability driver. This is a **testing
backend, not a shipping one**: it exists so Vulkan-backend changes can be
exercised on a Mac without a Linux or Windows host.

### Prerequisites

The [macOS Metal prerequisites](#macos-metal) are still required for a workspace
build (the Metal toolchain compiles the Metal shaders for the crates that always
build them). In addition:

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

2. Building `shaderc` from source (which compiles GLSL to SPIR-V at runtime)
   needs **CMake**, **Python 3**, and **Git** on `PATH`. The first build takes a
   few minutes; the result is cached in `target/`.

   ```sh
   brew install cmake python git
   ```

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
- `GraphicsConfig.shadow_map_size = 0` (the 1x1 fallback shadow array) renders
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
   linker plus the Windows SDK, which supplies the HLSL shader compilers
   (`FXC` and `DXC`). The build script locates `dxcompiler.dll` / `dxil.dll` in
   the Windows SDK automatically.

### Build

```sh
cargo build --release
```

### Optional: temporal upscaling SDKs

The DirectX backend can use vendor temporal upscalers (AMD FidelityFX FSR 3,
Intel XeSS, NVIDIA DLSS) and Microsoft's DirectX 12 Agility SDK. These are all
**optional**: if an SDK is not present the build script prints a warning, skips
it, and the renderer falls back to native-resolution rendering. Install one and
point the build at it with the matching environment variable. The variables and
their defaults are listed under
[Third-party environment variables](#third-party-environment-variables); on a
machine with the default install paths, none of them need setting.

**The Agility SDK is the exception and is off by default**, because bundling it
decides where the finished executable can run (see below). Ask for it with:

```powershell
$env:CN_ENABLE_AGILITY_SDK = "1"
```

FSR 3 needs it, so a build without the opt-in renders at native resolution and
logs why when something requests FSR 3.

#### Why the Agility SDK is opt-in

Bundling it links `D3D12SDKVersion` / `D3D12SDKPath` into the executable, and
Windows' `d3d12.dll` reads those exports _before any engine code runs_. If
`D3D12Core.dll` is not in a `D3D12/` directory beside the executable, D3D12
device creation fails outright — there is no fallback to the OS runtime, and
every adapter then reports unsupported:

```
D3D12 init failed: no suitable D3D12 adapter found. This binary was built with
CN_ENABLE_AGILITY_SDK=1, so it bundles Microsoft's Agility SDK and needs
D3D12Core.dll in ...
```

So the opt-in decides _where the finished executable can run_, which is why an
installed SDK is not enough to trigger it. `cargo build` leaves the binary in
`target/<profile>/` with the staged directory beside it and everything works;
`cargo install` copies only the executable, and the copy is dead. Turning the
bundling on means committing to shipping that directory with the binary — a
packaging step, not something `cargo install` can do.

The runtime alternative, `ID3D12SDKConfiguration::SetSDKVersion`, cannot replace
the exports: Microsoft documents it as usable "only in Windows Developer Mode"
(it exists for tools such as PIX) and it returns `DXGI_ERROR_INVALID_CALL`
elsewhere.

So:

- **Installing or redistributing a lone executable** — leave the opt-in off.
  The binary runs anywhere on the OS D3D12 runtime, without FSR 3.
- **Packaging a distribution that carries `D3D12/`, or a local build you run
  out of `target/`** — set `CN_ENABLE_AGILITY_SDK=1` and get FSR 3.

## Windows (Vulkan)

Since DirectX is the default on Windows, the `vulkan` feature selects a Vulkan
build instead.

### Prerequisites

In addition to the [DirectX prerequisites](#windows-directx-12) above (the MSVC
toolchain is still required):

1. Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) from LunarG. This
   provides the Vulkan loader and validation layers, plus the prebuilt `shaderc`
   library used to compile GLSL to SPIR-V.

2. Point `shaderc` at the SDK's prebuilt library so it does not have to build from
   source:

   ```powershell
   $env:SHADERC_LIB_DIR = "$env:VULKAN_SDK\Lib"
   ```

   If `SHADERC_LIB_DIR` is unset, `shaderc` is compiled from source instead,
   which additionally requires **CMake**, **Python 3**, and **Git** on `PATH`.

Windowing and input use the native Win32 window (shared with the DirectX
backend), so no windowing-library install or runtime DLL is involved.

### Build

```sh
cargo build --release --features vulkan
```

### Optional: patched FidelityFX Vulkan runtime

FSR temporal upscaling on Vulkan uses AMD's FidelityFX runtime
(`amd_fidelityfx_vk.dll`). SDK v1.1.4 declares the FSR3 upscaler's
`rw_luma_history` storage image as `rgba8` while the C++ creates it as
`R16G16B16A16_FLOAT`, so every FSR dispatch trips a validation-layer
format-mismatch warning and the reads and stores are per-spec undefined
(upstream
[issue #161](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/issues/161),
open). Declaring `rgba16f` matches the view; output is visually identical.

Nothing upstream ships the fix and nothing will: SDK v2 dropped Vulkan
altogether ("Vulkan is currently not supported in SDK") and deleted the GLSL
callbacks the fix applies to, so **v1.1.4 is the last release with a Vulkan
backend**. Raising the `fidelityfx` pin past it removes Vulkan FSR from the
engine.

Build the patched runtime out of the SDK already under `vendor/`:

```powershell
python scripts/vendor.py fetch fidelityfx    # if not already vendored
python scripts/vendor.py build fidelityfx-vk
```

That applies the shader fix, builds `ffx-api` for `VK_X64` (which recompiles the
shader permutations), and installs the result at
`vendor/fidelityfx-vk-1.1.4-windows-x86_64/bin/`, which `build.rs` prefers over
the stock SDK copy. It is idempotent, needs **CMake**, the **Visual Studio Build
Tools** C++ x64 toolset, and the **Vulkan SDK** with `VULKAN_SDK` set (for
`glslc`), and takes `--generator` for a toolchain other than the default
`Visual Studio 18 2026`. The change it makes is recorded in
`scripts/patches/fsr3upscaler_luma_history_rgba16f.patch`.

Without it the build falls back to the stock SDK DLL, which works but warns
every dispatch; with neither, Vulkan FSR falls back to native resolution. To
return to the stock DLL, delete the `fidelityfx-vk-*` directory.

## Linux (Vulkan)

Vulkan is the only backend on Linux. The package names below are for Debian /
Ubuntu; translate them to your distribution's equivalents as needed.

### Prerequisites

1. Install the build toolchain and the system development libraries:

   ```sh
   sudo apt update
   sudo apt install \
     build-essential cmake pkg-config git python3 \
     libssl-dev libasound2-dev \
     libglfw3 libglfw3-dev \
     libwayland-dev libwayland-bin wayland-protocols \
     libx11-dev libxkbcommon-dev libxrandr-dev libxinerama-dev \
     libxcursor-dev libxi-dev libudev-dev libdbus-1-dev
   ```

   - `build-essential`, `cmake`, `git`, `python3` — build `shaderc` from source
     (and GLFW, when no prebuilt library is found). `shaderc` is always built
     from source and linked statically, so the binaries carry no
     `libshaderc_shared.so` dependency. The first build takes a few minutes; the
     result is cached in `target/`.
   - `libssl-dev` — TLS for the networking client.
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

3. To **run** the engine you also need a Vulkan-capable GPU driver (e.g.
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
