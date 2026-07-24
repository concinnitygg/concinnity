<!-- Auto-generated - do not edit. -->

# CameraShot

A reusable [Camera3D](Camera3D.md) preset: reference it from a
[Scene](Scene.md)'s `camera_shot`, or use it standalone.

Used standalone, it expands into a [Camera3D](Camera3D.md) with the same
parameters.

**Examples**

With Scenes: camera switches per scene (declared on each Scene):
```jsonl
{"name":"wide", "type":"CameraShot","args":{"fov_y_degrees":80,"position":[0,1.75,8],"yaw":3.14}}
{"name":"close","type":"CameraShot","args":{"fov_y_degrees":55,"position":[0,1.5,3],"yaw":3.14}}
{"name":"intro", "type":"Scene","args":{"camera_shot":"wide"}}
{"name":"detail","type":"Scene","args":{"camera_shot":"close"}}
```

From library preset (standalone, replaces Camera3D):
```jsonl
{"name":"cam","type":"CameraShot","args":{"preset":"shot_outdoor_wide"}}
```

## Parameters

- `preset`: A string. Name of a built-in or file-backed preset (e.g. "shot_eye_level"). Preset values are used as defaults; any inline fields override them.
- `fov_y_degrees`: A float. Vertical field of view in degrees. Defaults to `75.0`.
- `near`: A float. Near clip plane distance in world units. Defaults to `0.05`.
- `far`: A float. Far clip plane distance in world units. Defaults to `200.0`.
- `position`: An array of 3 floats. World-space camera position. Defaults to `[0.0, 0.0, 0.0]`.
- `yaw`: A float. Yaw rotation in radians (Y-axis, applied first). Defaults to `0.0`.
- `pitch`: A float. Pitch rotation in radians (X-axis, applied second). Defaults to `0.0`.
