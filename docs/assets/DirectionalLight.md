<!-- Auto-generated - do not edit. -->

# DirectionalLight

An infinitely distant directional light (sun, moon, or sky fill).

Up to 4 directional lights may be declared; extras beyond 4 are silently ignored.
When no directional light is present, a built-in warm sun is used as a fallback.
Under an [Atmosphere](Atmosphere.md) the first one is the sun: its color is the
sunlight above the air, and it lights the scene dimmed and reddened by the air.

## Parameters

- `direction`: An array of 3 floats. Direction pointing toward the light source. Does not need to be normalized. Defaults to `[-0.3, 0.85, 0.4]`.
- `color`: An array of 3 floats. Linear-space RGB color of the light. Defaults to `[1.0, 1.0, 1.0]`.
- `intensity`: A float. Intensity multiplier applied to the color. Defaults to `1.0`.
- `angular_radius_deg`: A float. Angular radius of the light's disc in the sky, in degrees: a smooth surface reflects the disc's image rather than a point. 0 is a point. Under an [Atmosphere](Atmosphere.md) the sun takes the atmosphere's disc. Defaults to `0.0`.
