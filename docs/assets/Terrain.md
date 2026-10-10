<!-- Auto-generated - do not edit. -->

# Terrain

A rectangle of ground shaped by a height grid: rendered as a mesh with
`material`, collided by physics, and covered by grass `layers`.

The terrain spans `extent` (half-width and half-depth) around `center`, whose
height is the terrain's base. `resolution` cells along each side make up the
grid, so every corner is one height sample. The heights come from one of two
sources:

- Generated: rolling hills of up to `amplitude` meters above the base,
  shaped by `seed`. An `amplitude` of 0 gives a flat field.
- A `heightmap` [Texture](Texture.md): its red channel maps black to
  `elevation_min` and white to `elevation_max` above the base, stretched
  over the extent the same way as a layer's density mask.

The rendered surface, the surface bodies collide with, and the ground every
grass blade roots in are the same triangles. Grass grows only on terrain:
each entry in `layers` grows one [Grass](Grass.md) look over the terrain,
optionally masked. Several terrains, and several layers on one terrain, may
overlap.

## Parameters

- `center`: An array of 3 floats. World-space center; its height is the terrain's base. Defaults to `[0.0, 0.0, 0.0]`.
- `extent`: An array of 2 floats. Half-width and half-depth `[x, z]`, in meters. Defaults to `[50.0, 50.0]`.
- `resolution`: An integer. Grid cells along each side. Clamped to [4, 1024]. Defaults to `128`.
- `material`: A string. The [Material](Material.md) the surface renders with. Optional.
- `amplitude`: A float. Tallest generated hill above the base, in meters. Ignored with a `heightmap`. Defaults to `2.0`.
- `seed`: An integer. Shapes the generated hills: each seed gives a different landscape. Ignored with a `heightmap`. Defaults to `0`.
- `heightmap`: A string. A [Texture](Texture.md) whose red channel sets the heights. Unset generates them from `amplitude` and `seed`.
- `elevation_min`: A float. Height of a black `heightmap` texel above the base, in meters. Defaults to `0.0`.
- `elevation_max`: A float. Height of a white `heightmap` texel above the base, in meters. Defaults to `10.0`.
- `layers`: An array of [TerrainLayer](TerrainLayer.md) objects. The grass that grows on this terrain, one entry per look. Defaults to `[]`.
