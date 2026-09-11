<!-- Auto-generated - do not edit. -->

# CameraTrack

Drives the world's [Camera3D](Camera3D.md) along a scripted path, so a
fly-through visits the same poses on every machine that runs it.

One per world, and its presence takes the camera away from the input
controller: a world declaring a track is driven by the track, whatever
`controller` the camera carries.

The track has two independent lists played against one clock. `travel` is
where the camera goes and `turn` is where it looks, so a leg of each runs
at the same time and the camera can turn toward one thing while traveling
toward another. Each list runs from the camera's authored pose; when one
runs out the camera holds that list's last value while the other finishes.

The clock is the fixed simulation step, not the frame delta. A machine that
renders half as fast visits the same poses at the same track times and
simply samples fewer of them, which is what makes two runs comparable. The
track freezes while a world-pausing screen is open, like the rest of the
simulation.

## Parameters

- `travel`: An array of [CameraTravel](CameraTravel.md) objects. Where the camera goes, leg by leg.
- `turn`: An array of [CameraTurn](CameraTurn.md) objects. Where the camera looks, leg by leg. Played against the same clock as `travel` and independent of it, so the camera can turn one way while traveling another.
