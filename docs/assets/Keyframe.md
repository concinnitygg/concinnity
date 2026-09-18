<!-- Auto-generated - do not edit. -->

# Keyframe

One keyframe in an animation track: a joint pose sampled at `time` seconds.
The pose fields (`translation`, `rotation_deg`, `scale`) are given directly
on the keyframe, each defaulting to the identity transform when omitted.

## Parameters

- `time`: A float. Time of this keyframe in seconds from the clip start.
- `translation`: An array of 3 floats. Local translation. Defaults to `[0.0, 0.0, 0.0]`.
- `rotation_deg`: An array of 3 floats. Local YXZ Euler rotation in degrees. Defaults to `[0.0, 0.0, 0.0]`.
- `scale`: An array of 3 floats. Per-axis local scale. Defaults to `[1.0, 1.0, 1.0]`.
