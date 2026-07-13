<!-- Auto-generated - do not edit. -->

# AudioCue

Plays audio when a [View](View.md) is shown.

A cue links a [View](View.md) to an [AudioClip](AudioClip.md): whenever UI
navigation makes the view active (a `view:show` or `view:toggle` action, a
[KeyBinding](KeyBinding.md), dismissing an overlay back to it, or being the
world's initial view), the clip plays. Cues play flat on the main mix with
no 3D position; use an [AudioEmitter](AudioEmitter.md) for positional sound.

The `kind` decides the playback behavior:

- `music`: loops until replaced. Showing a view whose music cue is already
  playing leaves the track running, so navigating between views that share
  a cue is seamless. A view with a *different* music cue replaces the
  track; a view with *no* music cue leaves the current music playing.
- `sound`: a one-shot effect, played every time the view is shown.

```jsonl
{"name":"inn_theme","type":"AudioCue","args":{"view":"page_inn","clip":"theme","kind":"music"}}
{"name":"door_sfx","type":"AudioCue","args":{"view":"page_door","clip":"door_creak"}}
```

## Parameters

- `view`: A string. The [View](View.md) whose activation triggers this cue. Optional.
- `clip`: A string. The [AudioClip](AudioClip.md) to play. Optional.
- `kind`: A string (see [CueKind](CueKind.md)). Playback behavior: a looping `music` track or a one-shot `sound`.
- `volume`: A float. Linear gain applied to the clip (1.0 leaves it unchanged). Defaults to `1.0`.
