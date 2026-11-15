<!-- Auto-generated - do not edit. -->

# UpscaleQuality

Render-scale preset for `GraphicsConfig.anti_aliasing.upscaling`. The ratio applies to both
axes (input pixel count = output * ratio per axis), so `Quality` renders at
4/9 of the output pixel count, `Performance` at 1/4, and `UltraPerformance`
at 1/9. `Native` renders every output pixel, so the upscaler only
anti-aliases.

## Values

- `native`: The full output pixel count.
- `quality`: 4/9 of the output pixel count.
- `balanced`: Roughly a third of the output pixel count.
- `performance`: A quarter of the output pixel count.
- `ultra_performance`: A ninth of the output pixel count.
