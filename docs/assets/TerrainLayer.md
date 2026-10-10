<!-- Auto-generated - do not edit. -->

# TerrainLayer

One ground cover a [Terrain](Terrain.md) grows: a [Grass](Grass.md) look, and
where on the terrain it grows.

Without a `density_mask` the grass covers the whole terrain. With one, the
mask's red channel scales the grass's density across the terrain: white
grows the full density, black leaves the ground bare. The mask is stretched
over the terrain's extent, its first row along the terrain's `-Z` edge and
its first column along `-X`.

## Parameters

- `grass`: A string. The [Grass](Grass.md) whose blades this layer grows. Defaults to `0`.
- `density_mask`: A string. A [Texture](Texture.md) whose red channel scales the density across the terrain. Unset grows the grass everywhere.
