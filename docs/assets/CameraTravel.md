<!-- Auto-generated - do not edit. -->

# CameraTravel

One straight run on a [CameraTrack](CameraTrack.md)'s travel track.

The direction is world space rather than camera relative, so a turn running
over the same span does not bend the path: the two tracks stay independent,
which is what lets them be read separately.

## Parameters

- `direction`: An array of 3 floats. World-space direction of the run. Need not be unit length. Defaults to `[0.0, 0.0, 0.0]`.
- `distance`: A float. How far the run travels along `direction`, in world units. Defaults to `0.0`.
- `speed`: A float. Travel rate in world units per second, which fixes the leg's duration as `distance` divided by it. An eased leg still covers `distance` in that duration, so this is the average rate rather than the peak. Defaults to `0.0`.
- `seconds`: A float. Duration in seconds, overriding the one `distance` and `speed` imply. A leg with only this set holds the camera still for that long. Defaults to `0.0`.
- `ease`: A string (see [Ease](Ease.md)). How the run is paced. Defaults to `"linear"`.
- `cut`: A boolean. Jump to where the leg ends instead of running to it. The leg takes no time, whatever `speed` or `seconds` say, and the first frame drawn from the new pose starts afresh rather than blending with the frames before the jump, as at a shot change. A leg that takes no time anyway is a cut as soon as it moves the camera. Defaults to `false`.
- `segment`: A string. Names the segment of the path this leg opens, for whatever reads where the track has reached. An empty label continues whichever segment the previous leg was in. Defaults to `""`.
