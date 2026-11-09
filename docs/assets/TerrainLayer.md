<!-- Auto-generated - do not edit. -->

# TerrainLayer

One ground cover a [Terrain](Terrain.md) grows: a [Grass](Grass.md) look, and
where on the terrain it grows.

Without a `density_mask` the grass covers the whole terrain. With one, the
mask's red channel scales the grass's density across the terrain: white
grows the full density, black leaves the ground bare. The mask is stretched
over the terrain's extent, its first row along the terrain's `-Z` edge and
its first column along `-X`.

On a [Planet](Planet.md) a layer grows over the planet's own meadow instead,
and `density_mask` is not supported. There the layer grows only where its
`rule` holds, thinning out over the rule's soft edges, so a meadow can end
above a beach. A terrain does not read `rule`.

## Parameters

- `grass`: A string. The [Grass](Grass.md) whose blades this layer grows. Defaults to `0`.
- `density_mask`: A string. A [Texture](Texture.md) whose red channel scales the density across the terrain. Unset grows the grass everywhere.
- `rule`: A [GroundRule](GroundRule.md) object. Where on a planet the grass grows. The default grows it everywhere.
