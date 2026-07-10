<!-- Auto-generated - do not edit. -->

# FollowController

Third-person follow settings carried on a [CameraController](CameraController.md).

When `follow` is set the camera becomes a third-person orbit camera: the
mouse orbits around the followed character, and WASD steers the character
itself (camera-relative). The character must be a
[SkinnedMesh](skinned_mesh.md) with a `capsule`, so it has a kinematic
character capsule to move.

## Parameters

- `target`: A string. Name of the followed [SkinnedMesh](skinned_mesh.md). It must declare a `capsule`. Optional.
- `distance`: A float. Orbit distance from the pivot to the camera, in world units. Defaults to `4.0`.
- `height`: A float. Pivot height above the character's feet, in world units. Defaults to `1.5`.
- `drive`: A string (see [FollowDrive](FollowDrive.md)). How the character moves; see [FollowDrive](FollowDrive.md).
- `turn_speed`: A float. Character turn rate toward the input heading, in radians per second. Defaults to `10.0`.
- `speed_parameter`: A string. Name of the character's [AnimGraph](anim_graph.md) float parameter that receives the current travel speed in world units per second (drives a locomotion blendspace). Empty disables parameter writes, leaving the graph externally driven. Defaults to `"speed"`.
- `jump_height`: A float. Jump apex height in world units when the jump key is pressed while grounded. `0` disables jumping. Defaults to `0.0`.
