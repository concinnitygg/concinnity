<!-- Auto-generated - do not edit. -->

# Camera3D

Declares the 3D camera. One per scene.

## Parameters

- `fov_y_degrees`: A float. Vertical field-of-view in degrees. Defaults to `75.0`.
- `near`: A float. Near clip plane distance. Defaults to `0.05`.
- `view_distance`: A float. How far the camera sees, in world units. Objects lying wholly beyond this distance along the view direction are not drawn, and shadows reach no farther; an object that straddles it is drawn whole, never cut. `null` (the default) sees without limit: there is no far clip plane.
- `position`: An array of 3 floats. Initial eye position in world space [x, y, z]. Defaults to `[0.0, 1.7, 0.0]`.
- `yaw`: A float. Initial yaw in radians (0 = looking toward -Z). Defaults to `0.0`.
- `pitch`: A float. Initial pitch in radians. Defaults to `0.0`.
- `controller`: A [CameraController](CameraController.md) object. Input controller settings, or `null` to leave the camera uncontrolled (driven by a [CameraShot](CameraShot.md) / [Scene](Scene.md) cutscene). Omitted defaults to a free-fly inspector controller.
