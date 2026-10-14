<!-- Auto-generated - do not edit. -->

# Culling

How the renderer culls objects the camera cannot see. The `culling` section
of [GraphicsConfig](GraphicsConfig.md); a field left unset takes the default
below.

## Parameters

- `two_pass_occlusion`: A boolean. Two-pass occlusion culling. Reduces objects popping in a frame late when they're revealed by camera or occluder motion, at the cost of extra culling work each frame. Needs the bindless GPU-cull path. Defaults to `true`. Applies at the next launch.
