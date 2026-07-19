<!-- Auto-generated - do not edit. -->

# SpotLight

A cone-shaped local light: a point light restricted to the cone around
`direction`, with a soft edge between `inner_angle` and `outer_angle`.

Distance attenuation matches [PointLight](PointLight.md); the cone adds an
angular falloff that is full brightness inside the inner cone and fades to
black at the outer cone. Spot lights share the same per-scene local-light
budget as point lights and are culled by the same clustered pass. Secondary
effects (volumetric fog, SDF raymarching, and reflection-probe capture) do
not consider them.

```jsonl
{"name":"lantern","type":"SpotLight","args":{"position":[0.0,4.0,-2.0],"direction":[0.0,-1.0,0.0],"color":[1.0,0.9,0.7],"intensity":20.0,"range":10.0,"inner_angle":18.0,"outer_angle":30.0}}
```

## Parameters

- `position`: An array of 3 floats. World-space position of the light source. Defaults to `[0.0, 4.0, 0.0]`.
- `direction`: An array of 3 floats. Direction the cone points, away from the light. Does not need to be normalised; defaults to straight down when degenerate.
- `color`: An array of 3 floats. Linear-space RGB colour of the light. Defaults to `[1.0, 1.0, 1.0]`.
- `intensity`: A float. Intensity multiplier applied to the colour. Defaults to `20.0`.
- `range`: A float. Maximum reach in world units; attenuation is zero at this distance. Defaults to `10.0`.
- `inner_angle`: A float. Half-angle in degrees of the fully lit inner cone. Clamped to `outer_angle`. Defaults to `18.0`.
- `outer_angle`: A float. Half-angle in degrees at which the cone fades to black. Clamped to (0, 89.9]. Defaults to `30.0`.
