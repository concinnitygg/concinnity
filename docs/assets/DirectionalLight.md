<!-- Auto-generated - do not edit. -->

# DirectionalLight

An infinitely distant directional light (sun, moon, or sky fill).

Up to 4 directional lights may be declared; extras beyond 4 are silently ignored.
When no directional light is present, a built-in warm sun is used as a fallback.

## Parameters

- `direction`: An array of 3 floats. Direction pointing toward the light source. Does not need to be normalised. Defaults to `[-0.3, 0.85, 0.4]`.
- `color`: An array of 3 floats. Linear-space RGB colour of the light. Defaults to `[1.0, 1.0, 1.0]`.
- `intensity`: A float. Intensity multiplier applied to the colour. Defaults to `1.0`.
