<!-- Auto-generated - do not edit. -->

# AntiAliasing

Anti-aliasing and temporal upscaling: how edges are smoothed, and whether
the scene renders below native resolution and is reconstructed to full. The
`anti_aliasing` section of [`GraphicsConfig`](GraphicsConfig.md); a field left
unset takes the default below.

Temporal anti-aliasing is the default, which is the look on capable
hardware: the `Auto` graphics quality preset clamps it to `fxaa` below the
mid tier.

## Parameters

- `mode`: A string (see [AaMode](AaMode.md)). Anti-aliasing mode. `off` smooths nothing; `fxaa` applies a cheap composite-pass edge filter; `msaa` renders the scene multisampled, which smooths geometric edges but not shading; `taa` (default) jitters the projection and accumulates detail across frames for the cleanest edges and shading, at the cost of a velocity pre-pass and a history buffer. Where neither the world nor the user sets `upscaling` and the quality preset renders natively, `taa` turns `upscaling` on at the `native` render scale, so the platform's temporal upscaler anti-aliases, with the engine's own resolve as the fallback. Applies live, rebuilding the passes it gates.
- `upscaling`: A boolean. Temporal upscaling. Renders the 3D scene at the resolution `upscale_quality` sets and reconstructs a full-resolution image through the platform's temporal upscaler, which replaces TAA while on. Unset, the quality preset decides: on where it upscales, and on at `native` scale under TAA where it renders natively. `false` keeps the engine's TAA. Defaults to `false`. Applies at the next launch.
- `upscale_quality`: A string (see [UpscaleQuality](UpscaleQuality.md)). Render-scale preset for `upscaling`; each step progressively lowers the internal resolution. While `upscaling` is off the scene renders at native resolution whatever this says. Defaults to `"quality"`. Applies at the next launch.
- `upscaler`: A string (see [UpscalerBackend](UpscalerBackend.md)). Which upscaler `upscaling` uses. `auto` (default) picks the best available at runtime (DLSS on NVIDIA RTX, else XeSS, else FSR3); `fsr3` / `dlss` / `xess` request a specific one and fall back when it is unavailable on the current GPU or build. Also picks the upscaler `taa` runs through at native resolution. Metal treats every value as `MetalFX`. Applies at the next launch.
