<!-- Auto-generated - do not edit. -->

# CameraTravel

One straight run on a [CameraTrack](CameraTrack.md)'s travel track.

The direction is world space rather than camera relative, so a turn running
over the same span does not bend the path: the two tracks stay independent,
which is what lets them be read separately.

## Parameters

- `direction`: An array of 3 floats. World-space direction of the run. Need not be unit length.
- `distance`: A float. How far the run travels along `direction`, in world units.
- `speed`: A float. Travel rate in world units per second, which fixes the leg's duration as `distance` divided by it. An eased leg still covers `distance` in that duration, so this is the average rate rather than the peak.
- `seconds`: A float. Duration in seconds, overriding the one `distance` and `speed` imply. A leg with only this set holds the camera still for that long.
- `ease`: A string (see [Ease](Ease.md)). How the run is paced.
- `segment`: A string. Names the reporting segment this leg opens. An empty label continues whichever segment the previous leg was in.
