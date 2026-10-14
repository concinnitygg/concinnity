<!-- Auto-generated - do not edit. -->

# Planet

A whole world to stand on: a sphere of ground `radius` meters around
`center`, raised into hills, rendered with `material`, and pulling every
body toward its center.

The ground is generated from `seed`: hills up to `amplitude` meters above
the radius, the broadest about `feature_size` meters across, with
`octaves` layers of finer detail on them, each half the width and half the
height of the one before. The same seed gives the same planet on every run.
Nothing about the ground is stored: it is built around the camera as it
moves, finer near it and coarser toward the horizon, so a planet costs the
same to load whatever its size.

Gravity points at the center from wherever a body is, `gravity` meters per
second squared, so a first-person camera with a [RigidBody](RigidBody.md)
walks all the way around. Up is always away from the center: looking
around turns about it, and the horizon stays level.

Far from where the world started, positions stay as precise as they were
near it: the world is simulated around the camera, wherever on the planet
it is. A world has at most one planet.

Places on the planet are named by latitude and longitude, in degrees.
Latitude 0, longitude 0 is straight up from the center, where a world
whose center is below the origin starts; the north pole lies toward `-Z`
from the center, so a camera at the start facing `-Z` faces north, and
longitude grows eastward, toward `+X`. Altitude is measured above
`radius`, height above the ground above the hills.

A [WaterSurface](WaterSurface.md) whose `shape` is `planet_sea` floods the
planet to its `sea_level`.

## Parameters

- `center`: An array of 3 floats. World-space center. The default puts the top of a planet of the default radius at the origin.
- `radius`: A float. Radius of the lowest ground, in meters. At least 100. Defaults to `50000.0`.
- `amplitude`: A float. Highest hilltop above `radius`, in meters. An `amplitude` of 0 gives a smooth sphere. Defaults to `40.0`.
- `feature_size`: A float. Width of the broadest hills, in meters. Defaults to `2000.0`.
- `octaves`: An integer. Layers of ever finer detail on the hills. Clamped to [1, 16]. Defaults to `8`.
- `seed`: An integer. Shapes the ground: each seed gives a different planet. Defaults to `0`.
- `material`: A string. The [Material](Material.md) the ground renders with. Optional.
- `gravity`: A float. Pull toward the center, in meters per second squared. Defaults to `20.0`.
- `layers`: An array of [TerrainLayer](TerrainLayer.md) objects. The grass that grows on the planet, one entry per look. Each grows over the whole planet as a meadow: soft-edged bare patches a couple of dozen meters across are laid over the ground, different for every layer, and the same every run. A layer's `density_mask` is not supported on a planet: a world that sets one fails validation. Defaults to `[]`.
