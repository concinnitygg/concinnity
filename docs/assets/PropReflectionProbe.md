<!-- Auto-generated - do not edit. -->

# PropReflectionProbe

A reflection probe that rides on a [Prop](Prop.md) and is captured again
all the time, so a mirror-like prop (a chrome ball, a car's paint)
reflects what is around it wherever it goes: the ground it rolls over,
the shore and the sea, the props it passes.

The cube is rendered from the center of the prop's bounds, all six faces
every frame. The prop itself is left out of its own capture. Its glossy
surface reflects the cube at full strength, a surface right beside it
blends the cube with any static [`ReflectionProbe`](ReflectionProbe.md)
there. Its influence ends a fifth of the prop's smallest half-size past
its bounds, enough to cover the prop's own surface, and beyond that the
reflection hands over to the sky within about half the prop's size.

The capture draws the opaque scene within `capture_distance` of the prop,
200 m by default, and the environment's sky beyond it, which on a planet
carries the far ground and sea below the horizon. Grass is drawn as the
far field's meadow rather than as blades, and a planet's sea as a calm
surface at sea level. It is skipped where ray-traced reflections already
reflect the scene.

```json
["PropReflectionProbe",{"prop":"marble"}]
```

Each face is `resolution` texels square. The default suits a prop that
covers a few hundred pixels at most; a larger reflector seen up close
(the side of a car) may want 256, which shades four times the pixels.
A shorter `capture_distance` costs less to capture, a longer one draws
more of the far scene itself; `null` draws all of it.

```json
["PropReflectionProbe",{"prop":"car","resolution":256,"capture_distance":400.0}]
```

A world uses at most one.

## Parameters

- `prop`: A string. The [Prop](Prop.md) the probe rides on and reflects in. Optional.
- `resolution`: An integer. Edge of each cube face, in texels: a power of two from 16 to 512. Another value is rounded to the nearest power of two in that range. Defaults to `128`.
- `capture_distance`: A float. How far the capture reaches, in meters: an object lying wholly farther than this from the prop along the axis a cube face looks down is left out of that face, and the sky shows in its place. `null` captures without limit. Defaults to `200.0`.
