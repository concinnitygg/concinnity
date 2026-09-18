<!-- Auto-generated - do not edit. -->

# ProportionGroup

A proportion slider: one value in `[-1, 1]` written as a scale and / or
length change on every listed joint.

## Parameters

- `name`: A string. Group name (the panel row). Defaults to `""`.
- `caption`: A string. Panel caption; the name when empty. Defaults to `""`.
- `region`: A string. The region the row belongs to (panel grouping). Defaults to `""`.
- `joints`: An array of strings. Joints the row writes; only those the skeleton has are written. Defaults to `[]`.
- `scale`: A float. Scale change at full deflection (`0` leaves scale alone). Defaults to `0.0`.
- `length`: A float. Length change at full deflection, in model units (`0` leaves it alone). Defaults to `0.0`.
