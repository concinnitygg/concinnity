<!-- Auto-generated - do not edit. -->

# FrameReport

Times every frame of a run and prints what they cost when it ends.

One per world. The report leads with frame time, because that is the number
a player feels, and gives its median and tail percentiles rather than its
mean: a mean hides exactly the stutter that matters. Beside it are the
frame's CPU work and its GPU time, so a stretch reads as CPU-bound or
GPU-bound at a glance, and under it the render passes and the systems that
owned each, with their shares.

A run that names segments as it goes is reported segment by segment as well
as whole, so a cost that belongs to one stretch of it says so instead of
spreading itself thinly over the average.

## Parameters

- `warmup_seconds`: A float. Frames earlier than this many seconds into the run are dropped. Shader compilation, streaming residency, temporal-antialiasing history and auto-exposure all converge over the opening seconds of a run. Without a discard, two runs of identical code disagree, so this is what makes one run comparable with the next. Defaults to `2.0`.
- `budget_ms`: A float. The frame budget, in milliseconds, that the over-budget count is taken against. Defaults to a 60 Hz frame.
- `stop_when_complete`: A boolean. Whether the run ends as soon as the world says the measurement is complete. `true` reports and stops. `false` keeps the world running and reports when it closes instead, for a world watched rather than left to itself. A world that never says it is complete runs on either way, and reports whatever it gathered on the way out. A [CameraTrack](CameraTrack.md) is one thing that says so, when it reaches the end of its path. Defaults to `true`.
