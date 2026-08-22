<!-- Auto-generated - do not edit. -->

# PhysicsConfig

Configures the world's physics floor / terrain.

Optional: a world with physics bodies but no `PhysicsConfig` simulates over a
flat floor at Y = 0. Physics runs whenever the world declares a
`PhysicsConfig`, a [RigidBody](RigidBody.md), or a [PropBody](PropBody.md).
Declare a `PhysicsConfig` to put bodies on terrain or a non-zero floor.

For terrain-based outdoor scenes the terrain parameters must match the
terrain mesh exactly.

## Parameters

- `floor_y`: A float. Y coordinate of the floor. When left at 0.0 it is auto-detected from the camera; set it explicitly to override. Defaults to `0.0`.
- `terrain_half_width`: A float. Half-width of the terrain mesh along X. Must match the terrain mesh. Leave at 0.0 (with `terrain_subdivisions` = 0) for flat-floor scenes. Defaults to `0.0`.
- `terrain_half_depth`: A float. Half-depth of the terrain mesh along Z. Must match the terrain mesh. Defaults to `0.0`.
- `terrain_subdivisions`: An integer. Subdivision count of the terrain mesh. When 0, a flat floor at Y = 0 is used instead of a heightfield. Defaults to `0`.
- `terrain_amplitude`: A float. Height variation of the terrain mesh. Must match the terrain mesh. Defaults to `0.0`.
- `terrain_offset_y`: A float. World-space Y offset of the terrain: the height of the prop that renders the terrain mesh. Leave at 0.0 when the terrain sits at the origin. Defaults to `0.0`.
- `terrain_mesh`: A string. Name of a [ProceduralMesh](ProceduralMesh.md) with `generator: "heightfield"`. When set, the physics surface is built from that mesh's source image so props rest on the visible terrain. Takes precedence over the `terrain_*` values above. Optional.
- `layers`: An array of strings. Extra collision layer names beyond the built-ins (`world`, `prop`, `character`, `trigger`). At most 28; referenced by collider `layer` fields and `no_collide` pairs.
- `no_collide`: An array of arrays of 2 strings. Unordered layer-name pairs that do not collide. Everything collides by default; each pair here disables collision (and contact solving) between its two layers symmetrically. Pairs naming `character` also filter the character controller's movement.
- `contact_min_impulse`: A float. Minimum contact impulse (mass times velocity change) for a collision to publish a contact event. Resting contact stays below it; raise to hear only hard impacts. Defaults to `1.0`.
