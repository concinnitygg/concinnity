<!-- Auto-generated - do not edit. -->

# JointProportion

One joint's proportion change.

## Parameters

- `joint`: A string. Name of the joint in the target mesh's `skeleton`.
- `scale`: A float. Uniform scale applied to the joint (and, through the hierarchy, everything below it). `1` leaves it alone. Defaults to `1.0`.
- `length`: A float. Extra length along the bone, in model units: every child joint is pushed that far along its bind direction from this joint. `0` leaves it alone. Defaults to `0.0`.
