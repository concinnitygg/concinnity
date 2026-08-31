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
rustc --version   # should report 1.85.0 or later
```

### The Slang compiler

Every backend's engine shaders are written once as `.slang` and compiled by the
`slangc` binary. All three backends compile them at build time and embed the
result, so `slangc` is a **build-time** requirement and a binary built with one
present carries its shaders wherever it goes.

Building without `slangc` still succeeds. The binary then compiles its shaders
at renderer init instead, which needs `slangc` on every host that runs it — so
install one before building anything you intend to ship.

An embedded artifact is used only while the source still matches what it was
built from, which leaves two cases that compile at startup even on a binary that
has them. Hot-reload (`cn debug`, `cn editor`) prefers the checkout's copy of a
shader, so on Metal those two commands recompile the engine's shaders and need
`slangc`. And a Vulkan device that cannot seat the bindless texture pool at its
fixed ceiling sizes the pool to the world instead, which changes the source of
the two bindless main-pass programs; MoltenVK is such a device, so a macOS
Vulkan build needs `slangc` at runtime for that pair.

The engine requires a **2026.1 or newer** release. Earlier ones emit SPIR-V
declaring capabilities the shaders never use, which Vulkan rejects.

Install a release from
[shader-slang/slang](https://github.com/shader-slang/slang/releases) and put its
`slangc` first on `PATH`. `$VULKAN_SDK/bin` is searched too, but only the
Windows Vulkan SDK has tracked releases new enough; the LunarG Linux packages
have shipped well behind the floor (`slangc -version` on the Ubuntu package
prints its package tag rather than a release number, and the engine refuses a
compiler whose release it cannot read). Confirm what you have with:

```sh
slangc -version
```

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
point the build at it with the matching environment variable (defaults shown):

| SDK                | Environment variable  | Default install path                  | Default |
| ------------------ | --------------------- | ------------------------------------- | ------- |
| D3D12 Agility SDK  | `AGILITY_SDK_ROOT`    | `C:\microsoft.direct3d.d3d12.1.619.3` | **off** |
| FidelityFX (FSR 3) | `FIDELITYFX_SDK_ROOT` | `C:\FidelityFX-SDK-v1.1.4`            | on      |
| Intel XeSS         | `XESS_SDK_ROOT`       | `C:\XeSS_SDK_3.0.1`                   | on      |
| NVIDIA Streamline  | `STREAMLINE_SDK_ROOT` | `C:\streamline-sdk-v2.11.1`           | on      |

The three upscaler DLLs are loaded with `LoadLibrary` at runtime and degrade to
a fallback when absent, so they are picked up automatically wherever they are
installed, and each is disabled explicitly with `CN_ENABLE_FFX_FSR3=0`,
`CN_ENABLE_XESS=0`, or `CN_ENABLE_DLSS=0`.

**The Agility SDK is the exception and is off by default**, because bundling it
decides where the finished executable can run (see below). Ask for it with:

```powershell
$env:CN_ENABLE_AGILITY_SDK = "1"
```

FSR 3 needs it, so a build without the opt-in renders at native resolution and
logs why when something requests FSR 3.

#### Why the Agility SDK is opt-in

Bundling it links `D3D12SDKVersion` / `D3D12SDKPath` into the executable, and
Windows' `d3d12.dll` reads those exports *before any engine code runs*. If
`D3D12Core.dll` is not in a `D3D12/` directory beside the executable, D3D12
device creation fails outright — there is no fallback to the OS runtime, and
every adapter then reports unsupported:

```
D3D12 init failed: no suitable D3D12 adapter found. This binary was built with
CN_ENABLE_AGILITY_SDK=1, so it bundles Microsoft's Agility SDK and needs
D3D12Core.dll in ...
```

So the opt-in decides *where the finished executable can run*, which is why an
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
(`amd_fidelityfx_vk.dll`). The stock SDK v1.1.4 declares the FSR3 upscaler's
`rw_luma_history` storage image as `rgba8` while the C++ creates it as
`R16G16B16A16_FLOAT`, so every FSR dispatch trips a validation-layer
format-mismatch warning (upstream
[issue #161](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/issues/161)).
A pre-built patched DLL is already vendored at
`crates/concinnity-engine/third_party/ffx/amd_fidelityfx_vk.dll`, and `build.rs` prefers
it over the stock SDK copy automatically, so **most builds need no action**.

Run the helper script only when you need to rebuild the patched DLL yourself
(for example after updating the SDK). It applies the one-line shader fix
(`rgba8` -> `rgba16f`), rebuilds `ffx-api` for `VK_X64` from SDK source (which
recompiles the shader permutations), and copies the result into
`crates/concinnity-engine/third_party/ffx/`:

```powershell
# Uses the SDK at $env:FIDELITYFX_SDK_ROOT, else C:\FidelityFX-SDK-v1.1.4
pwsh scripts/setup_ffx_vk_dll.ps1

# Git-clone the v1.1.4 source first if the SDK is absent
pwsh scripts/setup_ffx_vk_dll.ps1 -CloneIfMissing

# Point at a custom SDK location
pwsh scripts/setup_ffx_vk_dll.ps1 -SdkRoot D:\ffx
```

The script requires **CMake**, the **Visual Studio Build Tools** C++ x64 toolset,
and the **Vulkan SDK** with `VULKAN_SDK` set (for `glslc`). It is idempotent:
re-running re-applies the patch only if needed and rebuilds. To fall back to the
unmodified SDK DLL, delete the vendored copy.

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
