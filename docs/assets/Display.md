<!-- Auto-generated - do not edit. -->

# Display

How the finished frame is encoded for the display: standard dynamic range,
extended-range HDR, or PQ (HDR10). The `display` section of
[GraphicsConfig](GraphicsConfig.md); left unset, the output is standard
dynamic range.

## Parameters

- `hdr`: A boolean. HDR display output. On a capable display, emits extended-range HDR instead of the standard tonemapped output. Falls back to standard output when the display or platform doesn't support HDR. Auto-exposure meters toward perceptual middle gray while it is on. Defaults to `false`. Applies at the next launch.
- `pq`: A boolean. PQ (HDR10) output. When true, and `hdr` is on, and the display has HDR headroom, output is PQ-encoded for HDR10 panels. No effect when `hdr` is off. Defaults to `false`. Applies at the next launch.
