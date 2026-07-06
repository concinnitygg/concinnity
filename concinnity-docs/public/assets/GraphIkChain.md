<!-- Auto-generated - do not edit. -->

# GraphIkChain

One two-bone IK chain, pinning the chain's end joint (typically a foot)
to the ground the physics scene finds beneath it.

`joints` names the chain root, middle, and end in the target skeleton --
e.g. a hip, knee, and foot. The middle joint must be the direct child of
the root and the end the direct child of the middle. Every frame the
runtime probes straight down from the animated end joint; when a surface
is within range, the chain bends so the end lands `foot_height` above it.
Pinning pauses automatically while the character is airborne.

## Parameters

- `joints`: An array of strings. Names of the chain's root, middle, and end joints, in order. Exactly three are required, matching the target skeleton's joint names.
- `pole`: An array of 3 floats. Bend direction in mesh space: the middle joint bows toward this vector (a knee points forward, an elbow backward). Defaults to `[0.0, 0.0, 1.0]`.
- `weight_parameter`: A string. Name of a declared graph parameter scaling the solve in `[0, 1]`; empty pins at full strength. Lets gameplay fade IK in and out.
- `foot_height`: A float. Height the end joint rests above the probed surface, in mesh units (the sole-to-ankle offset for a foot). Defaults to `0.0`.
