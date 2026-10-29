<!-- Auto-generated - do not edit. -->

# BenchmarkConfig

Customizes how a world is benchmarked.

A world is benchmarked when it is launched with `bench`, whether it
declares one of these or not. Without one the camera holds the pose the
world starts it at for ten seconds after the warm-up, or, on a world with
a [`Planet`](Planet.md), tours the planet as [`PlanetTour`](PlanetTour.md)
describes. A config that authors any of its route (`start`, `travel`,
`turn`, `boom`, `phases`, `drive` or `carry_from_seconds`) keeps it unless
it asks for the tour by `route`. The run says which route it took as it starts. Declaring one
does nothing on an ordinary launch, so a world can ship its benchmark
beside it.

A benchmarked run takes the camera away from the player: it holds still
through the warm-up, then follows `travel` and `turn` from where it stood.
The legs are the same ones a [`CameraTrack`](CameraTrack.md) takes, and played
on the same fixed simulation clock, so a slower machine visits the same
poses at the same times and simply samples fewer of them.

With a `subject` the run films a [Prop](Prop.md) in third person, as a track
with a subject does: `travel` carries the prop, `turn` aims the camera and
`boom` frames the prop. `start` then places the prop, and the camera is
framed from it as the boom says from the first frame.

The run is cut into `phases`, named stretches of time that the report
gives one by one as well as whole, so a cost that belongs to one stretch
says so instead of spreading itself thinly over the average. The run ends
when the last phase does: a path shorter than that holds its last pose, and
a longer one is cut off.

The report leads with frame time, because that is the number a player
feels, and gives its median and tail percentiles rather than its mean: a
mean hides exactly the stutter that matters. Beside it are the frame's CPU
work and its GPU time, so a phase reads as CPU-bound or GPU-bound at a
glance, and under it the render passes and the systems that owned each,
with their shares.

## Parameters

- `warmup_seconds`: A float. Seconds the camera holds its start pose before the first phase, with every frame drawn in them left out of the measurement. Shader compilation, streaming residency, temporal-antialiasing history and auto-exposure all converge over the opening seconds of a run. Without a discard, two runs of identical code disagree, so this is what makes one run comparable with the next. Defaults to `2.0`.
- `budget_ms`: A float. The frame budget, in milliseconds, that the over-budget count is taken against. Defaults to a 60 Hz frame.
- `fps_cap`: An integer. The frame-rate cap the run holds to, whatever the world or the player's settings choose. `0` leaves it uncapped, since a capped GPU idles between frames and lowers its clocks. Defaults to `0`.
- `start`: A [CameraPose](CameraPose.md) object. Where the camera starts, or unset to start it where the world does. With a `subject`, where the subject starts.
- `subject`: A string. The [Prop](Prop.md) the run carries and films, or unset to carry the camera itself.
- `route`: A string (see [BenchmarkRoute](BenchmarkRoute.md)). Which way the run goes: the legs below, or a tour of the world's [`Planet`](Planet.md). Left to choose, a world with a planet tours it when the config authors none of its route. Defaults to `"auto"`.
- `tour`: A [PlanetTour](PlanetTour.md) object. How the run tours the planet, when it does.
- `drive`: An array of [DriveKey](DriveKey.md) objects. How the subject's [`RollingDrive`](RollingDrive.md) is steered after the warm-up, key by key, while the subject moves by itself. Defaults to `[]`.
- `carry_from_seconds`: A float. Seconds after the warm-up from which `travel` carries the subject. Until then the run only films it, and its physics and its `drive` move it; from then the run picks it up where it is, as fast as it is going, and eases it onto travel's path. `0` carries it from the start. Defaults to `0.0`.
- `travel`: An array of [CameraTravel](CameraTravel.md) objects. Where the camera goes after the warm-up, leg by leg. Defaults to `[]`.
- `turn`: An array of [CameraTurn](CameraTurn.md) objects. Where the camera looks after the warm-up, leg by leg. Played against the same clock as `travel` and independent of it. Defaults to `[]`.
- `boom`: An array of [CameraBoom](CameraBoom.md) objects. How the camera frames the `subject` after the warm-up, leg by leg, on the same clock as `travel` and `turn`. Defaults to `[]`.
- `phases`: An array of [BenchmarkPhase](BenchmarkPhase.md) objects. The named stretches the run is measured in, back to back from the end of the warm-up. Left empty, the run is one ten-second phase. Defaults to `[]`.
