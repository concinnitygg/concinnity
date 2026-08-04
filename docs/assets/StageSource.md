<!-- Auto-generated - do not edit. -->

# StageSource

Source declaration for one stage of a [Shader](Shader.md).

Provide either `source` (single platform) or `sources` (multi-platform).
When both are present, `sources` takes priority for the current platform.

**Platform keys:** `"metal"` (macOS), `"hlsl"` (Windows), `"glsl"` (Linux/Vulkan).

## Parameters

- `source`: A string. Single-platform source path; used when `sources` is absent or lacks the current platform key.
- `sources`: An object. Per-platform source paths keyed by `"metal"`, `"hlsl"`, or `"glsl"`. Takes priority over `source`. Optional.
