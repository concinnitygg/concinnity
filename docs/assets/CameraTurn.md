<!-- Auto-generated - do not edit. -->

# CameraTurn

One turn on a [CameraTrack](CameraTrack.md)'s turn track.

The angles are absolute headings rather than deltas, so a leg reads as the
direction the camera ends up looking. Yaw takes the shorter way round.
Either angle may be left out to keep the one the camera already holds,
which is what makes a leg with only `seconds` a hold rather than a turn to
zero.

## Parameters

- `yaw_deg`: A float. Heading the turn ends at, in degrees, or unset to hold the current one. `0` looks toward -Z.
- `pitch_deg`: A float. Elevation the turn ends at, in degrees, or unset to hold the current one. Positive looks up.
- `degrees_per_second`: A float. Turn rate in degrees per second, taken over whichever of yaw and pitch has further to go so the two arrive together. The leg's duration follows from it, and cannot be resolved until the world starts and the heading the first leg turns away from is known.
- `seconds`: A float. Duration in seconds, overriding the one `degrees_per_second` implies. A leg with only this set holds the heading for that long.
- `ease`: A string (see [Ease](Ease.md)). How the turn is paced.
