# concinnity-shader

The HLSL shader toolchain shared by
[Concinnity](https://crates.io/crates/concinnity) build scripts and the
renderer.

The engine's single-source `.hlsl` shaders compile through one pipeline: `dxc`
emits DXIL for D3D12 and SPIR-V for Vulkan, and the Metal leg takes that same
SPIR-V through spirv-cross to MSL and the Metal toolchain to a metallib. Most of that
happens at build time, in the device build script; the renderer compiles the
rest, such as a hot-reload edit or a permutation no build-time artifact covers.
Every call site assembles the full source text first, so a compile is a pure
function of that text, the entry point, and the target, which is what lets the
renderer's content-addressed shader cache key it.

## Constraints

- The single invocation is the point: build script and runtime must produce
  byte-identical artifacts, so the flag list exists exactly once, here.
- Sits below `concinnity-toolchain` (which is build-script-only and never
  linked into a shipped binary) and holds no policy beyond the one thing no
  compiler can be asked for: the Metal binding table, derived from the source's
  own `register()` annotations rather than from a hand-maintained list.
- `dxc` resolves from `dxc/bin` beside the running executable, then the
  checkout's pinned release under `vendor/` (`scripts/vendor.py fetch dxc`, or
  `build dxc` on macOS), then `PATH`, then `$VULKAN_SDK/bin`. A build without
  one stops and says what to install; a built binary needs one only to compile
  something it did not embed.
- The MSL leg and the layout reflection are behind the `spirv-cross` feature,
  which links spirv-cross itself. A DirectX or Vulkan build needs neither and
  builds no C++.

## The Metal binding policy

A resource's Metal index is the number on its `register()`, in the Metal class
its SPIR-V kind implies. The three Metal namespaces are disjoint, so a
structured buffer at `t0` and a texture at `t0` are `buffer(0)` and
`texture(0)` and do not collide. A texture and its sampler are two
declarations, each with a `vk::binding` of its own, so a Vulkan descriptor set
binds them as a sampled image and a sampler, as D3D and Metal do. The
toolchain refuses `[[vk::combinedImageSampler]]`: dxc drops unread inputs and
unused resources from the entry interface of any source that declares one,
whatever it is asked to preserve.
