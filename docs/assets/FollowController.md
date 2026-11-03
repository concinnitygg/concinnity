<!-- Auto-generated - do not edit. -->

# FollowController

Third-person follow settings carried on a [`CameraController`](CameraController.md).

When `follow` is set the camera becomes a third-person orbit camera: the
mouse orbits around what it follows, and WASD steers it (camera-relative).
It follows either a character, a [`SkinnedMesh`](skinned_mesh.md) with a
`capsule` named by `target`, or a [Prop](Prop.md) named by `prop`, which a
[`RollingDrive`](RollingDrive.md) on it lets the keys roll. Up is away from
the pull: a [`Planet`](Planet.md)'s center, or `+Y`.

## Parameters

- `target`: A string. Name of the followed [`SkinnedMesh`](skinned_mesh.md). It must declare a `capsule`. Optional.
- `prop`: A string. Name of the followed [Prop](Prop.md), in place of a `target`. Optional.
- `distance`: A float. Orbit distance from the pivot to the camera, in world units. Defaults to `4.0`.
- `height`: A float. Pivot height above the character's feet, or above a prop's center, in world units. Defaults to `1.5`.
- `lag_seconds`: A float. Seconds the pivot takes to catch up with a change in how what it follows moves, smoothing a bumpy ride. Steady motion, however fast, is followed exactly, and the pivot never falls behind by more than a quarter of `distance`. `0` follows exactly. Defaults to `0.0`.
- `drive`: A string (see [FollowDrive](FollowDrive.md)). How the character moves; see [`FollowDrive`](FollowDrive.md). Defaults to `"root_motion"`.
- `turn_speed`: A float. Character turn rate toward the input heading, in radians per second. Defaults to `10.0`.
- `speed_parameter`: A string. Name of the character's [`AnimationGraph`](anim_graph.md) float parameter that receives the current travel speed in world units per second (drives a locomotion blendspace). Empty disables parameter writes, leaving the graph externally driven. Defaults to `"speed"`.
- `jump_height`: A float. Jump apex height in world units when the jump key is pressed while grounded. `0` disables jumping. Defaults to `0.0`.
