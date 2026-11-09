<!-- Auto-generated - do not edit. -->

# PlanetSea

The ocean of a [Planet](Planet.md): a sphere of water `level` meters above
the planet's radius, so every valley whose ground lies lower floods, and
the coast is wherever the hills rise out of it.

The water looks and moves the way its `water` [`WaterSurface`](WaterSurface.md)
describes: colors, reflections, how the wind raises its waves, whitecaps
and spray. It is always a simulated ocean, its waves fixed to the planet's
ground wherever on it the camera roams, and it curves away to the horizon
with the ground. Its gusts follow the wind in step with the planet's
grass. It reflects the scene by ray tracing, reflection probes or the sky,
never by a planar mirror, and it takes no authored swell: the surface's
`waves` must be empty. The surface's `center`, `extent`, `subdivisions`
and `depth_meters` do not apply, and the surface is not drawn anywhere
else.

Where the planet has continents and the sea meets their beaches, its waves
come ashore as at a real coast. Over the shelf the open sea's longest
waves give way to long swell lines running in toward the land; they slow
and bunch up as the water shallows, turn to meet the beach nearly square,
a little askew so the breakers peel along it, and grow taller until they
break, about where a wave stands four fifths as tall as the water under
it. Bigger waves come in sets and break farther out. A breaking crest
pitches forward into a bore of white water that trails foam, and the wash
of each wave runs on up the sand and drains back, leaving the beach darker
and glossy where it reached, drying back between sets. The height and
rhythm of the surf follow the wind sea: a stronger wind or a higher
`wind_response` brings bigger, slower waves ashore. A sea standing over
the planet's beaches, or around a planet without seas, has no surf.

## Parameters

- `level`: A float. Height of the water above the planet's radius, in meters, at least 0. Defaults to `0.0`.
- `water`: A string. The [`WaterSurface`](WaterSurface.md) whose look the water takes. Defaults to `0`.
