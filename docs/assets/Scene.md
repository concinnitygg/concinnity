<!-- Auto-generated - do not edit. -->

# Scene

A named group of world content.

A [Prop](Prop.md) belongs to a Scene by naming it in the prop's `scene`. A
prop that names no scene is visible in every scene.

The first declared Scene is active at world start. Scene changes are driven
by actions: a UI `scene:<name>` action ([HitRegion](HitRegion.md) /
[KeyBinding](KeyBinding.md)) or a [Behavior](Behavior.md) scene node jumps to
the named scene, with the transition ("Cut" or "FadeBlack") declared on the
jump.

## Parameters

- `camera_shot`: A string. A [CameraShot](CameraShot.md) or [Camera3D](Camera3D.md) to activate when this scene becomes active. `None` keeps the current camera unchanged. Optional.
