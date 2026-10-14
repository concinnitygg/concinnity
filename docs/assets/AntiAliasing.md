<!-- Auto-generated - do not edit. -->

# AntiAliasing

Anti-aliasing and temporal upscaling: how edges are smoothed, and whether
the scene renders below native resolution and is reconstructed to full. The
`anti_aliasing` section of [GraphicsConfig](GraphicsConfig.md); a field left
unset takes the default below.

Temporal anti-aliasing is the default, which is the look on capable
hardware: the `Auto` graphics quality preset clamps it to `fxaa` below the
mid tier.

## Parameters

- `mode`: A string (see [AaMode](AaMode.md)). Anti-aliasing mode. `fxaa` applies a cheap composite-pass edge filter; `taa` (default) adds a temporal pass that jitters the projection and accumulates detail across frames for the cleanest edges, at the cost of a velocity pre-pass and a history buffer; `off` disables edge smoothing. Applies live, rebuilding the passes it gates.
- `upscaling`: A boolean. Temporal upscaling. Renders the 3D scene at a lower resolution (set by `upscale_quality`) and reconstructs a full-resolution image, trading some sharpness for performance. Replaces TAA while on. Defaults to `false`. Applies at the next launch.
- `upscale_quality`: A string (see [UpscaleQuality](UpscaleQuality.md)). Render-scale preset for `upscaling`; each step progressively lowers the internal resolution. No effect when `upscaling` is off. Defaults to `"quality"`. Applies at the next launch.
- `upscaler`: A string (see [UpscalerBackend](UpscalerBackend.md)). Which upscaler `upscaling` uses. `auto` (default) picks the best available at runtime (DLSS on NVIDIA RTX, else XeSS, else FSR3); `fsr3` / `dlss` / `xess` request a specific one and fall back when it is unavailable on the current GPU or build. No effect when `upscaling` is off. Metal treats every value as MetalFX. Applies at the next launch.
