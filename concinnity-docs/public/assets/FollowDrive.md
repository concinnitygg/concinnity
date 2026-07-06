<!-- Auto-generated - do not edit. -->

# FollowDrive

How a followed character converts movement input into displacement.

## Values

- `root_motion`: The controller only writes the speed parameter and the facing; the character moves by the displacement its animation clips carry (clips baked with [root_motion](animation.md)). Clips must travel along local -Z so the facing yaw and the travel direction agree.
- `direct`: The controller moves the character capsule directly at the camera controller's `move_speed`, for characters whose clips animate in place. The speed parameter is still written, so a locomotion blendspace matches the visual gait to the travel speed.
