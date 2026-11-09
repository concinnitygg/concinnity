<!-- Auto-generated - do not edit. -->

# GroundRule

Where on the ground something belongs: between two altitudes and between
two slopes, fading in and out over soft edges.

`altitude` is measured in meters above a [Planet](Planet.md)'s `radius`, its
sea level, and `slope` in degrees from level ground. Either end of either
band may be left unset (`null`), which leaves the band open on that side;
a rule with every end unset holds everywhere.

At each set bound the rule fades out over `altitude_edge` meters or
`slope_edge` degrees. The boundary also wanders by about as much again,
along a pattern fixed to the ground and the same every run, so two covers
meeting at one altitude interlock in ragged tongues rather than along a
contour line. Rules that meet at the same bound with the same edge share
that pattern, so where one ends the other begins.

## Parameters

- `altitude`: An array of 2 floats. Lowest and highest altitude, in meters above sea level. Unset ends are open. Defaults to `[null, null]`.
- `slope`: An array of 2 floats. Gentlest and steepest slope, in degrees from level, within [0, 90]. Unset ends are open. Defaults to `[null, null]`.
- `altitude_edge`: A float. How wide the fade at an altitude bound is, in meters. At least 0.01. Defaults to `1.0`.
- `slope_edge`: A float. How wide the fade at a slope bound is, in degrees. At least 0.1. Defaults to `5.0`.
