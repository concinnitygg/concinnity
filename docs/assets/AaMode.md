<!-- Auto-generated - do not edit. -->

# AaMode

Anti-aliasing mode for `PostProcessConfig.aa_mode`. `Off` runs no edge
smoothing; `Fxaa` (default) applies the composite's single-frame edge
filter, which is nearly free; `Taa` adds a temporal pass that jitters the
projection and reprojects detail across frames for the cleanest edges, at
the cost of a velocity pre-pass and a per-frame history buffer.

The mode also decides whether the scene renders multisampled: `Off` and
`Fxaa` keep the multisampled target, `Taa` renders single-sampled because
the temporal filter reconstructs the same edges. Temporal upscaling does
the same. See `hdr_sample_count`.

## Values

- `off`: No edge smoothing.
- `fxaa`: Single-frame edge filter in the composite.
- `taa`: Temporal anti-aliasing: jittered projection plus a reprojected history.
