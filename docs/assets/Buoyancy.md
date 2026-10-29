<!-- Auto-generated - do not edit. -->

# Buoyancy

Makes a [`PropBody`](PropBody.md) float in water.

A body with `buoyancy` floats in every [`WaterSurface`](WaterSurface.md) it
reaches, on the waves where they are drawn: it rises and falls with the
swell, pitches and rolls with the slope of the water under it, drifts with
the water, and in the surf of a planet sea the breaking waves carry it in
toward the beach. On calm water it settles and comes to rest.

How high it floats follows its density, the body's `mass` over the volume
of its collider, against the water's `density`: a body half as dense as
the water floats half submerged, one denser than the water sinks. Give the
body a `mass`: one left at 0 takes the mass of its collider at 1 kg per
cubic meter, a thousandth of water's density, so it rides with all but a
sliver of it above the water.

The water slows what moves through it: `linear_drag` and `angular_drag`
set how quickly the body comes to move with the water and stops spinning
once it is fully under, and less while it is only partly submerged.

```json
["PropBody",{"prop_name":"crate_a","mass":300.0,"buoyancy":{"linear_drag":1.5}}]
```

## Parameters

- `linear_drag`: A float. How quickly the water brings the body to its own speed, per second, fully submerged. At least 0. Defaults to `3.0`.
- `angular_drag`: A float. How quickly the water stops the body's spin, per second, fully submerged. At least 0. Defaults to `3.0`.
