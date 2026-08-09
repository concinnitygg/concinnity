<!-- Auto-generated - do not edit. -->

# AudioEmitter

A point source of sound in the world.

Plays its `clip` (an [AudioClip](AudioClip.md) reference) from `position`,
attenuated and panned relative to the camera. When `prop` names a
[Prop](Prop.md), the emitter tracks that prop's position every frame, so the
sound follows a moving object.

The sound is at full volume inside `min_distance`, fades according to
`rolloff` between `min_distance` and `max_distance`, and is inaudible
beyond `max_distance`.

```jsonl
{"name":"fire_sound","type":"AudioEmitter","args":{"clip":"fire_loop","position":[6.0,4.0,-6.0]}}
{"name":"waterfall","type":"AudioEmitter","args":{"clip":"falls","position":[0,2,8],"min_distance":3.0,"max_distance":80.0,"rolloff":"linear"}}
```

## Parameters

- `clip`: A string. The [AudioClip](AudioClip.md) this emitter plays. Optional.
- `position`: An array of 3 floats. World-space position of the sound source.
- `volume`: A float. Linear gain multiplier applied to the clip. Defaults to `1.0`.
- `looping`: A boolean. Whether the clip restarts when it ends. Defaults to `true`.
- `prop`: A string. Optional [Prop](Prop.md) whose position the emitter tracks each frame.
- `min_distance`: A float. Distance from the listener at which the sound plays at full volume. Defaults to `1.0`.
- `max_distance`: A float. Distance from the listener beyond which the sound is inaudible. Must exceed `min_distance`. Defaults to `50.0`.
- `rolloff`: A string (see [Rolloff](Rolloff.md)). How volume falls between `min_distance` and `max_distance`.
- `bus`: A string (see [AudioBus](AudioBus.md)). Mix bus the emitter routes through. Defaults to `sfx`.
