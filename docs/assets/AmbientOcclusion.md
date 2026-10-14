<!-- Auto-generated - do not edit. -->

# AmbientOcclusion

Screen-space ambient occlusion: darkens creases and contact areas where
ambient light is occluded. One per world; with none declared, the defaults
below apply.

On by default, which is the look on capable hardware: the `Auto` graphics
quality preset forces it off on the lowest tier. A ceiling only ever
reduces, so turning it off here holds on every tier.

## Parameters

- `enabled`: A boolean. Whether the occlusion pass runs. Defaults to `true`. Applies live, rebuilding the passes it gates.
- `radius`: A float. How far the search reaches for occluders, in world units. Larger values pick up broader, softer occlusion. Defaults to `0.5`. At least `0.001`. Applies live.
- `intensity`: A float. Occlusion strength. 1.0 is the natural amount; higher values exaggerate the contact darkening. Defaults to `1.0`. Clamped to `[0, 4]`. Applies live.
