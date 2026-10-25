<!-- Auto-generated - do not edit. -->

# Planet

A whole world to stand on: a sphere of ground around `center`, raised
into hills and continents and sunk into ocean basins, rendered with
`material`, and pulling every body toward its center.

The ground is generated from `seed`: hills up to `amplitude` meters above
the radius, the broadest about `feature_size` meters across, with
`octaves` layers of finer detail on them, each half the width and half the
height of the one before. The same seed gives the same planet on every run.
The ground is built around the camera as it moves, finer near it and
coarser toward the horizon, so a planet costs the same to load whatever
its size.

Its `edits` change the generated ground where they lie: a file of pages,
each over one square of the planet's quadtree, holding per point a height
change in meters and a detail weight that scales the generated hills, 1
keeping them whole and 0 leaving the smooth continents and coasts. A point
reads the finest page over it, so detail painted into a small area sits on
a coarse page over a wider one.

With a `land_share` under 1, the planet divides into land and sea around
its `radius`, which becomes sea level: continents about `continent_size`
meters across rise above it over `land_share` of the surface, and ocean
basins up to `ocean_depth` meters deep sink below it over the rest. The
coasts are gentle: from the waterline a beach `beach_width` meters wide
rises a few meters, the hills grow in behind it over half their width,
and out to sea the floor falls away at the beach's slope across a shallow
shelf before it levels off toward the ocean depth. The ground then spans
from `ocean_depth` below the radius to `amplitude` above it, or to the top
of the beach where the hills are lower than that. Its `sea` fills the
basins with water; a `level` of 0 fills them to the coasts. A planet
with no `sea` keeps its basins dry.

Gravity points at the center from wherever a body is, `gravity` meters per
second squared, so a first-person camera with a [`RigidBody`](RigidBody.md)
walks all the way around. Up is always away from the center: looking
around turns about it, and the horizon stays level.

Far from where the world started, positions stay as precise as they were
near it: the world is simulated around the camera, wherever on the planet
it is. A world has at most one planet.

The ground takes the look of `material`, overlaid in order by each of
`covers` where its rule holds, at most 4 of them: sand along the coasts,
rock on cliffs. The covers are drawn whether or not the planet grows grass
in its `layers`, and each layer's own rule keeps its blades off the covers
they should not grow on.

Places on the planet are named by latitude and longitude, in degrees.
Latitude 0, longitude 0 is straight up from the center, where a world
whose center is below the origin starts; the north pole lies toward `-Z`
from the center, so a camera at the start facing `-Z` faces north, and
longitude grows eastward, toward `+X`. Altitude is measured above
`radius`, height above the ground above the hills.

## Parameters

- `center`: An array of 3 floats. World-space center. The default puts the top of a planet of the default radius at the origin.
- `radius`: A float. Radius of sea level, or of the lowest ground on a planet that is all land, in meters. At least 100. Defaults to `50000.0`.
- `amplitude`: A float. Highest hilltop above `radius`, in meters. An `amplitude` of 0 gives a smooth sphere. Defaults to `40.0`.
- `feature_size`: A float. Width of the broadest hills, in meters. Defaults to `2000.0`.
- `octaves`: An integer. Layers of ever finer detail on the hills. Clamped to [1, 16]. Defaults to `8`.
- `seed`: An integer. Shapes the ground: each seed gives a different planet. Defaults to `0`.
- `land_share`: A float. Share of the surface that is land, in [0, 1]; the rest lies under sea level. 1 makes the whole planet land. Defaults to `1.0`.
- `continent_size`: A float. Width of the broadest continents and oceans, in meters. At least 100. Defaults to `20000.0`.
- `ocean_depth`: A float. Deepest the sea floor lies below sea level, in meters. Defaults to `60.0`.
- `beach_width`: A float. Width of the gentle rise from the waterline to where the hills begin, in meters. At least 1. Defaults to `60.0`.
- `sea`: A [PlanetSea](PlanetSea.md) object. The [`PlanetSea`](PlanetSea.md) that floods the low ground. None leaves it dry.
- `material`: A string. The [Material](Material.md) the ground renders with. Optional.
- `covers`: An array of [GroundCover](GroundCover.md) objects. Looks laid over `material` in order, each where its rule holds. At most 4; later ones are dropped. Defaults to `[]`.
- `gravity`: A float. Pull toward the center, in meters per second squared. Defaults to `20.0`.
- `layers`: An array of [TerrainLayer](TerrainLayer.md) objects. The grass that grows on the planet, one entry per look. Each grows over the whole planet as a meadow: soft-edged bare patches a couple of dozen meters across are laid over the ground, different for every layer, and the same every run. A layer's `density_mask` is not supported on a planet: a world that sets one fails validation. Defaults to `[]`.
- `edits`: A string. Path to the file of edits laid over the generated ground, relative to the project root. None leaves the ground as generated. A file that is missing or does not read as edits fails validation.
