<!-- Auto-generated - do not edit. -->

# TriggerVolume

An invisible sensor region that reports when something enters or leaves it.

A trigger volume senses overlap and never collides: nothing bounces off
it and it blocks no movement. [Reaction](Reaction.md)s listen for its
crossings with an `enter` or `exit` source, so "when the player steps into
this area, open that door" is two declared assets. `detects` filters what
sets it off: the player character, dynamic props, or anything. Volumes
sense at their authored position; they do not move at runtime.

```jsonl
{"name":"vault_zone","type":"TriggerVolume","args":{"position":[4,1,-2],"collider":{"shape":"cuboid","half_extents":[2,1.5,2]}}}
{"name":"vault_opens","type":"Reaction","args":{"on":{"enter":"vault_zone"},"actions":[{"despawn":{"target":"vault_door"}}],"once":true}}
```

## Parameters

- `position`: An array of 3 floats. World-space position of the volume's center.
- `rotation_deg`: An array of 3 floats. Euler rotation of the volume in degrees.
- `collider`: A [PropCollider](PropCollider.md) object. The sensed region, in the same shape vocabulary as a [PropCollider](PropCollider.md): a `cuboid` with `half_extents`, a `ball` with `radius`, or a `capsule`.
- `detects`: A string (see [TriggerFilter](TriggerFilter.md)). What sets the volume off: the `player` character, dynamic `props`, or `any` of them.
