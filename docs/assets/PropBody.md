<!-- Auto-generated - do not edit. -->

# PropBody

Makes a companion [Prop](Prop.md) a dynamic physics body.

Attach a `PropBody` to give a [Prop](Prop.md) real physics: it falls, collides,
stacks, tumbles, and (with `pickup: true` on the prop) can be carried and
thrown. A Prop with a `collider` but no `PropBody` is a static, immovable
obstacle.

```json
["PropBody",{"prop_name":"crate_a","mass":4.0,"friction":0.6}]
```

## Parameters

- `prop_name`: A string. The [Prop](Prop.md) this body drives. Must match a Prop declared in the same world. Optional.
- `mass`: A float. Mass in kilograms. 0 lets the simulation derive mass from the collider shape and a default density.
- `friction`: A float. Friction coefficient used for contacts with this body. Defaults to `0.5`.
- `restitution`: A float. Bounciness in [0, 1]. 0 is fully inelastic. Defaults to `0.0`.
- `gravity_scale`: A float. Multiplier applied to world gravity for this body. 1.0 is normal. Defaults to `1.0`.
- `linear_damping`: A float. Linear velocity damping, modeling air drag. Defaults to `0.05`.
- `angular_damping`: A float. Angular velocity damping, slowing the body's spin. Defaults to `0.05`.
- `rolling_resistance`: A float. Rolling resistance: the share of its load a rolling contact resists with, acting at the collider's rolling radius (a ball's or a capsule's radius). It stops a ball rolling on level ground. 0 rolls freely; a box does not roll. Defaults to `0.0`.
- `impact_clip`: A string. Optional [`AudioClip`](AudioClip.md) played at the contact point when this body collides hard enough to pass the world's `contact_min_impulse` (see [`PhysicsConfig`](PhysicsConfig.md)). Louder impacts play louder.
- `impact_volume`: A float. Linear gain applied to the impact clip at full impulse. Defaults to `1.0`.
- `asleep`: A boolean. Start the body asleep: it holds its authored pose, ignoring gravity, until something strikes it or a [Behavior](Behavior.md)'s `wake` node wakes it. A body leaning on an awake one wakes with it. Defaults to `false`.
- `buoyancy`: A [Buoyancy](Buoyancy.md) object. Makes the body float in water, as [`Buoyancy`](Buoyancy.md) describes. Unset, it sinks.
