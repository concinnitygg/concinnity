# Building Concinnity

The rendering backend is chosen automatically from the target platform:

| Platform | Default backend | Notes                                         |
| -------- | --------------- | --------------------------------------------- |
| macOS    | Metal           | Build with `--features vulkan` to use Vulkan. |
| Windows  | DirectX 12      | Build with `--features vulkan` to use Vulkan. |
| Linux    | Vulkan          | Only backend available.                       |

## Common prerequisites

Install the [Rust toolchain](https://rustup.rs) on every platform. The workspace
uses the **2024 edition**, so Rust **1.85 or newer** is required:

```sh
rustup update
rustc --version   # should report 1.85.0 or later
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
it, and the renderer falls back to native-resolution rendering. To enable one,
install it and point the build at it with the matching environment variable
(defaults shown):

| SDK                | Environment variable  | Default install path                  |
| ------------------ | --------------------- | ------------------------------------- |
| D3D12 Agility SDK  | `AGILITY_SDK_ROOT`    | `C:\microsoft.direct3d.d3d12.1.619.3` |
| FidelityFX (FSR 3) | `FIDELITYFX_SDK_ROOT` | `C:\FidelityFX-SDK-v1.1.4`            |
| Intel XeSS         | `XESS_SDK_ROOT`       | `C:\XeSS_SDK_3.0.1`                   |
| NVIDIA Streamline  | `STREAMLINE_SDK_ROOT` | `C:\streamline-sdk-v2.11.1`           |

Each can be disabled explicitly with `CN_ENABLE_AGILITY_SDK=0`,
`CN_ENABLE_FFX_FSR3=0`, `CN_ENABLE_XESS=0`, or `CN_ENABLE_DLSS=0`.

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
