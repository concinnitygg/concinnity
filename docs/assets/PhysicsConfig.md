<!-- Auto-generated - do not edit. -->

# PhysicsConfig

Configures the world's physics floor / terrain.

Optional: a world with physics bodies but no `PhysicsConfig` simulates over a
flat floor at Y = 0, and receives one carrying these values at start so the
settings are a component rather than a fallback. Physics runs whenever the world
declares a `PhysicsConfig`, a [RigidBody](RigidBody.md), a
[PropBody](PropBody.md), a [TriggerVolume](TriggerVolume.md), or a
[SkinnedMesh](SkinnedMesh.md) with a `capsule`. Declare a `PhysicsConfig` to
put bodies on terrain or a non-zero floor.

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
- `spawn_headroom`: An integer. Extra physics bodies reserved for props created while the world runs (by a [Spawner](Spawner.md), a [Behavior](Behavior.md) `spawn` node, or the host). Physics reserves every body it will ever need when the world loads and never grows: once the declared bodies plus this many are live, a further spawn gets no physics body and is reported as an error. This is a floor beneath what the build reserves on its own, not the whole reservation. Every [Spawner](Spawner.md) whose `interval` and `lifetime` bound how many copies can be alive at once is already reserved for, and the larger of the two numbers wins. Set a value here for the sources the build cannot count: a `Spawner` with `lifetime: 0` (its copies live forever), a `spawn` node in a behavior, and spawns the host drives itself. Defaults to `0`.
