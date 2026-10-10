<!-- Auto-generated - do not edit. -->

# PhysicsConfig

Configures the world's physics: the floor, collision layers, and how many
bodies to reserve.

Optional: a world with physics bodies but no `PhysicsConfig` simulates over a
flat floor at Y = 0, and receives one carrying these values at start so the
settings are a component rather than a fallback. Physics runs whenever the world
declares a `PhysicsConfig`, a [RigidBody](RigidBody.md), a
[PropBody](PropBody.md), a [TriggerVolume](TriggerVolume.md), or a
[SkinnedMesh](SkinnedMesh.md) with a `capsule`.

Every [Terrain](Terrain.md) in the world is solid ground. A world with no
terrain stands on a flat floor at Y = 0.

## Parameters

- `floor_y`: A float. Y coordinate of the floor. When left at 0.0 it is auto-detected from the camera; set it explicitly to override. Defaults to `0.0`.
- `layers`: An array of strings. Extra collision layer names beyond the built-ins (`world`, `prop`, `character`, `trigger`). At most 28; referenced by collider `layer` fields and `no_collide` pairs. Defaults to `[]`.
- `no_collide`: An array of arrays of 2 strings. Unordered layer-name pairs that do not collide. Everything collides by default; each pair here disables collision (and contact solving) between its two layers symmetrically. Pairs naming `character` also filter the character controller's movement.
- `contact_min_impulse`: A float. Minimum contact impulse (mass times velocity change) for a collision to publish a contact event. Resting contact stays below it; raise to hear only hard impacts. Defaults to `1.0`.
- `spawn_headroom`: An integer. Extra physics bodies reserved for props created while the world runs (by a [Spawner](Spawner.md), a [Behavior](Behavior.md) `spawn` node, or the host). Physics reserves every body it will ever need when the world loads and never grows: once the declared bodies plus this many are live, a further spawn gets no physics body and is reported as an error. This is a floor beneath what the build reserves on its own, not the whole reservation. Every [Spawner](Spawner.md) whose `interval` and `lifetime` bound how many copies can be alive at once is already reserved for, and the larger of the two numbers wins. Set a value here for the sources the build cannot count: a `Spawner` with `lifetime: 0` (its copies live forever), a `spawn` node in a behavior, and spawns the host drives itself. Defaults to `0`.
