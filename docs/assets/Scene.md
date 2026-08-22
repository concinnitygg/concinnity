<!-- Auto-generated - do not edit. -->

# Scene

A named group of world content.

[Prop](Prop.md)s belong to a Scene by naming convention: props whose `name`
begins with `<scene_name>_` are associated with that Scene. Props not
prefixed by any scene name are visible in every scene.

The first declared Scene is active at world start. Scene changes are driven
by actions: a UI `scene:<name>` action ([HitRegion](HitRegion.md) /
[KeyBinding](KeyBinding.md)) or a [Behavior](Behavior.md) scene node jumps to
the named scene, with the transition ("Cut" or "FadeBlack") declared on the
jump.

## Parameters

- `camera_shot`: A string. A [CameraShot](CameraShot.md) or [Camera3D](Camera3D.md) to activate when this scene becomes active. `None` keeps the current camera unchanged. Optional.
