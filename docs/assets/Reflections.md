<!-- Auto-generated - do not edit. -->

# Reflections

Reflections on glossy surfaces (water, polished floors): screen-space,
hardware ray-traced, or neither. The `reflections` section of
[`GraphicsConfig`](GraphicsConfig.md); a field left unset takes the default
below.

Both methods are on by default, which is the look on capable hardware. The
`Auto` graphics quality preset forces screen-space reflections off below
the high tier and permits ray tracing only on the top tier; where ray
tracing is not permitted or the GPU lacks it, `screen_space` takes over. A
ceiling only ever reduces, so turning a method off here holds on every tier.

## Parameters

- `screen_space`: A boolean. Screen-space reflections: mix reflected scene color over glossy surfaces, from what is already on screen. Defaults to `true`. Applies live, rebuilding the passes it gates.
- `ray_traced`: A boolean. Hardware ray-traced reflections. When the GPU supports ray tracing, traces real reflection rays so off-screen geometry still appears. Takes precedence over `screen_space`, falling back to it where ray tracing isn't available. Defaults to `true`. Applies live, rebuilding the passes it gates.
- `intensity`: A float. Reflection blend strength. Scales the Fresnel-weighted reflection mixed over the base shading. Defaults to `0.7`. Clamped to `[0, 1]`. Applies live.
- `max_distance`: A float. How far a reflection reaches, in world units. Longer reaches catch more distant reflections, more coarsely. Defaults to `40.0`. Clamped to `[1, 1000]`. Applies live.
- `ray_traced_resolution`: A string (see [PassResolution](PassResolution.md)). Internal resolution the ray-traced reflection rays are cast at. `half` (default) traces a quarter of the pixels and upsamples with a depth- and normal-aware filter; `full` traces every pixel for the sharpest mirrors; `quarter` is the cheapest. Only matters when `ray_traced` is on. Applies live, rebuilding the passes it gates.
- `ray_traced_shadows`: A boolean. Whether surfaces seen in a ray-traced reflection are shadowed from the sun. Each reflected hit then casts a second ray toward the sun, which roughly doubles the trace cost; off lights every reflected surface as if the sun reached it. Only matters when `ray_traced` is on. Defaults to `true`. Applies live, rebuilding the passes it gates.
- `blur_resolution`: A string (see [PassResolution](PassResolution.md)). Internal resolution of the roughness-aware reflection blur. `half` (default) blurs at a quarter of the pixels for a large saving and bilinearly upsamples; `full` blurs at native resolution; `quarter` is the cheapest. Smooth mirror surfaces stay sharp at any setting (the composite keeps the sharp reflection for low roughness). Applies live, rebuilding the passes it gates.
