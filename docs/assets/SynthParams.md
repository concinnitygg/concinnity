<!-- Auto-generated - do not edit. -->

# SynthParams

Generator parameters for a synthesized target. Each generator reads the
fields it needs and ignores the rest.

## Parameters

- `amplitude`: A float. Displacement at full weight, in model units. Defaults to `0.02`.
- `along`: A float. `bulge`: centre along the bone, as a fraction of its length. Defaults to `0.5`.
- `sigma`: A float. `bulge`: width of the lobe along the bone, as a fraction of its length. Defaults to `0.15`.
- `direction`: An array of 3 floats. `bulge`: model-space direction of the lobe; zero means radially away from the bone. Defaults to `[0.0, 0.0, 0.0]`.
- `reverse`: A boolean. `taper`: ramp from the distal end toward the proximal end instead. Defaults to `false`.
- `source`: A string. `mirror` / `blend_mask`: the authored target to derive from.
- `span`: An array of 2 floats. `surface_offset`: the window along the region's first bone, as fractions of its length, outside which the offset fades to nothing. Defaults to `[0.0, 1.0]`.
- `falloff`: A float. `surface_offset`: width of the fade at each end of `span`. Defaults to `0.1`.
