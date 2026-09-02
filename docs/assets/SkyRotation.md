<!-- Auto-generated - do not edit. -->

# SkyRotation

Turns the whole celestial sphere: the sky, the image-based lighting it
casts, every [DirectionalLight](DirectionalLight.md), and any
[Prop](Prop.md) hung on it.

One per world. The rotation at elapsed time `t` is `angle_deg +
degrees_per_second * t` about `axis`, taken in the sense a planet's own
spin gives the sky: with the default axis a body rises from `+Z`, passes
overhead through `+Y`, and sets toward `-Z`.

The component's own entity carries that rotation as its transform, so a
`Prop` naming this asset as its `parent` orbits with the sky. Reflection
probes are baked once and do not turn.

## Parameters

- `axis`: An array of 3 floats. The celestial pole in world space: the axis the sphere turns about. Does not need to be normalised. Defaults to `[1.0, 0.0, 0.0]`.
- `degrees_per_second`: A float. Turn rate in degrees per second. Negative runs the sky backwards. Defaults to `1.0`.
- `angle_deg`: A float. The angle the sky starts at, in degrees. Defaults to `0.0`.
