<!-- Auto-generated - do not edit. -->

# CameraTrack

Drives the world's [`Camera3D`](Camera3D.md) along a scripted path, so a
fly-through visits the same poses on every machine that runs it.

One per world, and its presence takes the camera away from the input
controller: a world declaring a track is driven by the track, whatever
`controller` the camera carries. Nor does the physics collide the camera
while a track drives it, so the path is flown exactly as written.

The track has two independent lists played against one clock. `travel` is
where the camera goes and `turn` is where it looks, so a leg of each runs
at the same time and the camera can turn toward one thing while traveling
toward another. Each list runs from the camera's authored pose; when one
runs out the camera holds that list's last value while the other finishes.

A leg marked `cut` jumps rather than runs, which is how a track changes
shot: the frame drawn from the new pose does not blend with the old one.

A track with a `subject` films a [Prop](Prop.md) from behind a boom, as a
third-person camera does. `travel` then carries the subject rather than the
camera, `turn` is still where the camera looks, and a third list, `boom`,
sets how far back the camera sits and where in the frame the subject
appears. The camera is placed from the subject on every frame, so the
subject holds its place in the frame exactly while either moves, and a
sweep of `turn` circles the camera round it. Without boom legs the camera
is placed at its authored distance from the subject, on the line of view
through it, from the first frame.

The subject must be a Prop with no `parent`, since travel writes its
position in world space, and is moved by the track alone: give it no
`collider` and no [`PropBody`](PropBody.md), or the physics pushes against
the track. A subject that is missing, parented or later removed leaves the
track driving the camera itself from wherever it stands, with a warning.

The clock is the fixed simulation step, not the frame delta. A machine that
renders half as fast visits the same poses at the same track times and
simply samples fewer of them, which is what makes two runs comparable. The
track freezes while a world-pausing screen is open, like the rest of the
simulation.

## Parameters

- `travel`: An array of [CameraTravel](CameraTravel.md) objects. Where the camera goes, leg by leg. Defaults to `[]`.
- `turn`: An array of [CameraTurn](CameraTurn.md) objects. Where the camera looks, leg by leg. Played against the same clock as `travel` and independent of it, so the camera can turn one way while traveling another. Defaults to `[]`.
- `subject`: A string. The [Prop](Prop.md) the track carries and films, or unset for a track that carries the camera itself.
- `boom`: An array of [CameraBoom](CameraBoom.md) objects. How the camera frames the `subject`, leg by leg, on the same clock as the other two lists. Used only with a `subject`. Defaults to `[]`.
