<!-- Auto-generated - do not edit. -->

# AudioCue

Plays audio when a [Screen](Screen.md) is shown.

A cue links a [Screen](Screen.md) to an [AudioClip](AudioClip.md): whenever UI
navigation makes the screen active (a `screen:show` or `screen:toggle` action, a
[KeyBinding](KeyBinding.md), dismissing an overlay back to it, or being the
world's initial screen), the clip plays. Cues play flat on the main mix with
no 3D position; use an [AudioEmitter](AudioEmitter.md) for positional sound.

The `kind` decides the playback behavior:

- `music`: loops until replaced. Showing a screen whose music cue is already
  playing leaves the track running, so navigating between screens that share
  a cue is seamless. A screen with a *different* music cue replaces the
  track; a screen with *no* music cue leaves the current music playing.
- `sound`: a one-shot effect, played every time the screen is shown.

## Parameters

- `screen`: A string. The [Screen](Screen.md) whose activation triggers this cue. Optional.
- `clip`: A string. The [AudioClip](AudioClip.md) to play. Optional.
- `kind`: A string (see [CueKind](CueKind.md)). Playback behavior: a looping `music` track or a one-shot `sound`.
- `volume`: A float. Linear gain applied to the clip (1.0 leaves it unchanged). Defaults to `1.0`.
- `bus`: A string (see [AudioBus](AudioBus.md)). Mix bus the cue routes through. Defaults to `music` for a music cue and `sfx` for a sound cue; set `voice` for dialogue.
- `priority`: An integer. Voice priority for a `sound` cue. When all voice slots are busy, a new sound silences the oldest lowest-priority voice; a sound outranked by everything playing is skipped. Higher wins; the default is 0.
