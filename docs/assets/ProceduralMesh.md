<!-- Auto-generated - do not edit. -->

# ProceduralMesh

Geometry built by a named generator at compile time. Use for standard shapes.

For custom / hand-authored geometry use [Mesh](Mesh.md) instead.

**Built-in generators:**

## Parameters

- `generator`: A string. Built-in generator name (required), e.g. `room`, `box`, `cylinder`, `sphere`, `plane`, `water_grid`, or `extrude`. Defaults to `""`.
- `half_width`: A float. Half-width along X (room / plane / water grid), in world units. Defaults to `8.0`.
- `half_depth`: A float. Half-depth along Z (room / plane / water grid), in world units. Defaults to `10.0`.
- `ceiling_height`: A float. Ceiling height for the `room` generator, in world units. Defaults to `3.5`.
- `half_extents`: An array of 3 floats. Half-extents `[x, y, z]` for the `box` generator. Optional.
- `radius`: A float. Radius for the `cylinder` and `sphere` generators. Optional.
- `height`: A float. Height for the `cylinder` and `extrude` generators. Optional.
- `segments`: An integer. Number of radial segments around the `cylinder` and `sphere` generators. Optional.
- `rings`: An integer. Number of horizontal rings on the `sphere` generator. Optional.
- `subdivisions`: An integer. Grid subdivisions for the `water_grid` generator. Higher is more detailed. Optional.
- `profile`: An array of arrays of 2 floats. 2D outline `[[x, z], ...]` extruded by the `extrude` generator. Optional.
- `corner_radius`: A float. Corner-rounding radius for the `extrude` generator. 0 keeps sharp corners. Optional.
- `corner_segments`: An integer. Number of segments used to round each corner in the `extrude` generator. Optional.
- `lod_levels`: An integer. Number of level-of-detail versions to generate, including the original. `1` (the default) generates none; values are clamped to `[1, 8]`.
- `lod_distances`: An array of floats. Camera distances at which to switch to each lower-detail version; length should be `lod_levels - 1`. Empty lets the build choose defaults.
