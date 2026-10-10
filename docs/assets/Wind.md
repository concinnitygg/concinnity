<!-- Auto-generated - do not edit. -->

# Wind

The wind blowing across the world: a steady breeze along one horizontal
direction, broken up by gusts that roll through with it.

One per world. Anything that sways reads the same wind, so grass, foliage
and other moving surfaces lean and ripple together. With none declared the
air is still.

Gusts are patches of stronger and weaker air, about `gust_scale` meters
across, carried downwind at the wind's own speed. `gustiness` sets how far
they swing the strength: 0 is a steady wind, 1 lets it drop to calm and
peak at double between gusts.

## Parameters

- `direction`: An array of 2 floats. The horizontal direction the wind blows toward, `[x, z]`. Does not need to be normalized. Defaults to `[1.0, 0.0]`.
- `strength`: A float. Mean wind speed, in meters per second. 0 is still air. Defaults to `3.0`.
- `gustiness`: A float. How strongly gusts vary the speed, in [0, 1]. Defaults to `0.5`.
- `gust_scale`: A float. Typical width of one gust, in meters. Defaults to `20.0`.
