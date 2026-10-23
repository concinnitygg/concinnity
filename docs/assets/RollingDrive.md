<!-- Auto-generated - do not edit. -->

# RollingDrive

Rolls a companion [Prop](Prop.md)'s body the way it is steered.

The prop needs a ball or capsule `collider` and a
[`PropBody`](PropBody.md). Steering turns the body about the axis square to
the ground's up and the heading, so it rolls toward the heading by the
grip its contact has, under whatever gravity pulls it: down `-Y`, or
toward a [`Planet`](Planet.md)'s center. In water the roll gives way to a
paddle by the share of the body that floats, since a spinning ball does
not swim. A third-person `Camera3D` that follows the prop steers it from
the movement keys or the left stick, relative to where it looks.

```json
["RollingDrive",{"prop_name":"ball","acceleration":8.0,"max_speed":12.0}]
```

## Parameters

- `prop_name`: A string. The [Prop](Prop.md) this drive rolls. Must match a Prop with a [`PropBody`](PropBody.md) declared in the same world. Optional.
- `acceleration`: A float. How hard the drive rolls the body on level ground, in meters per second squared, whatever it weighs. Defaults to `8.0`.
- `max_speed`: A float. Speed along the heading, in meters per second, past which the drive stops adding to it. A slope or a push can still carry the body faster. Defaults to `10.0`.
- `sprint_multiplier`: A float. What sprinting multiplies `acceleration` and `max_speed` by. Defaults to `1.8`.
- `jump_speed`: A float. Speed a jump gives the body away from the ground, in meters per second. 0 never jumps. A jump only leaves the ground. Defaults to `5.0`.
- `paddle`: A float. Share of the drive that pushes a floating body through the water: fully afloat it paddles at `paddle` times `acceleration`. Defaults to `0.3`.
