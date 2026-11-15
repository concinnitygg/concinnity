<!-- Auto-generated - do not edit. -->

# AaMode

Anti-aliasing mode for `GraphicsConfig.anti_aliasing.mode`. `Off` runs no edge smoothing;
`Fxaa` applies the composite's single-frame edge filter, which is nearly
free; `Msaa` renders the scene multisampled, which smooths geometric edges
but not shading; `Taa` adds a temporal pass that jitters the projection and
reprojects detail across frames for the cleanest edges and shading, at the
cost of a velocity pre-pass and a per-frame history buffer.

Only `Msaa` renders the scene multisampled, and only while temporal
upscaling is off. See `hdr_sample_count`.

## Values

- `off`: No edge smoothing.
- `fxaa`: Single-frame edge filter in the composite.
- `msaa`: Multisampled scene rendering.
- `taa`: Temporal anti-aliasing: jittered projection plus a reprojected history.
