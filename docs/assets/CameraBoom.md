<!-- Auto-generated - do not edit. -->

# CameraBoom

One change of framing on a [`CameraTrack`](CameraTrack.md)'s boom track: how
far the camera sits from the track's `subject` and where the subject sits
in the frame.

Either value may be left out to keep the one the boom already holds. Until
the first leg that sets a value the boom holds that value, so a track can
open on the framing its first leg names.

## Parameters

- `distance`: A float. Distance from the subject the camera ends at, in world units, or unset to hold the current one.
- `offset_deg`: An array of 2 floats. Where the subject sits in the frame the leg ends at, in degrees right of and above the center of view, or unset to hold the current place. `[0, -10]` keeps the subject a little below the center.
- `seconds`: A float. Duration in seconds. A leg with only this set holds the framing for that long. Defaults to `0.0`.
- `ease`: A string (see [Ease](Ease.md)). How the change is paced. Defaults to `"linear"`.
- `cut`: A boolean. Snap to the leg's framing instead of moving to it, as at a shot change. A leg that takes no time anyway is a cut as soon as it changes the framing. Defaults to `false`.
