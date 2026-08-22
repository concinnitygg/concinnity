<!-- Auto-generated - do not edit. -->

# PointLight

A spherical point light with quadratic distance attenuation.

The forward renderer lights every surface from all declared point lights (up
to a large per-scene budget). Secondary effects (volumetric fog, SDF
raymarching, and reflection-probe capture) still consider only the first 8.

## Parameters

- `position`: An array of 3 floats. World-space position of the light source. Defaults to `[0.0, 2.5, 0.0]`.
- `color`: An array of 3 floats. Linear-space RGB colour of the light. Defaults to `[1.0, 1.0, 1.0]`.
- `intensity`: A float. Intensity multiplier applied to the colour. Defaults to `8.0`.
- `range`: A float. Maximum reach in world units; attenuation is zero at this distance. Defaults to `6.0`.
