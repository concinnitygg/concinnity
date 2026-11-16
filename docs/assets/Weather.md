<!-- Auto-generated - do not edit. -->

# Weather

The weather over a world's cloud layers: systems of cloud and clear sky
hundreds of kilometers across on Earth, spiraling round their storms and
drawn out along the latitudes, over a climate that runs from a wet equator
through dry subtropics to stormy middle latitudes.

Every [`Cloud`](Cloud.md) layer covers its own `coverage` of the sky on
average, and the weather moves that cover about: overcast under a system,
clear between them. It also sets how tall the clouds grow (the tropics
and the storms heap them up), how dense they are (from thin gray veils the
ground shows through to white cores) and how far separate clouds merge
into sheets, as along a front. Without one, every layer covers the sky
evenly.

`system_size` is the span of a weather system, in meters; systems smaller
than about a twelfth of the world's radius are finer than the weather
holds and are drawn at that size. `storminess` is how common storms are,
in `[0, 1]`, and `swirl` how far the cloud around each winds into a
spiral: the way the world spins north of the equator, the other way south
of it, and less toward the equator. `belts` is how strongly latitude
shapes the cover. The systems travel east at `drift` meters per second.
Latitude and east are taken about the axis of the world's
[`SkyRotation`](SkyRotation.md), or straight up without one; on a world with
no [`Planet`](Planet.md) they lie on the sphere the
[`Atmosphere`](Atmosphere.md) wraps. `seed` places the systems: each seed is
a different sky.

## Parameters

- `system_size`: A float. Span of one weather system, in meters. At least 1000. Defaults to `1500000.0`.
- `storminess`: A float. How common storms are, in `[0, 1]`. Defaults to `0.5`.
- `swirl`: A float. How far the cloud round a storm winds into a spiral, in `[0, 1]`. Defaults to `0.6`.
- `belts`: A float. How strongly latitude shapes the cover, in `[0, 1]`: 0 spreads the systems evenly over the world. Defaults to `0.6`.
- `drift`: A float. How fast the systems travel east, in meters per second. At least 0. Defaults to `10.0`.
- `seed`: An integer. Places the systems: each seed is a different sky. Defaults to `0`.
