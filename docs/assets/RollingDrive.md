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

A `ramp` builds the drive up the longer a direction is held. A
`climb_rate` lets a held jump fly the body: it climbs away from the ground
by a steady share of its height each second, from meters to kilometers in
even tenfolds, and the movement keys carry it along as fast as it climbs,
so forward and jump together fly it up and ahead, though never so fast
along the ground that gravity cannot bring it back. Let go, the climb
eases to a stop and the body falls under gravity, held back by the air.
Holding descend in the air flies it down the same way, slowing as it
nears the ground.

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
- `ramp`: A float. How much `acceleration` and `max_speed` grow for each second a direction is held, as a share of them: 0.5 doubles them two seconds in. Letting go of the direction starts them over. 0 holds them steady. Defaults to `0.0`.
- `ramp_limit`: A float. The most a held direction multiplies `acceleration` and `max_speed` by. Defaults to `4.0`.
- `climb_rate`: A float. Share of its height over the ground a held jump climbs the body by each second. 0 never climbs. Defaults to `0.0`.
- `climb_max_speed`: A float. The fastest the body climbs, or flies along the ground, in meters per second. Defaults to `1000.0`.
