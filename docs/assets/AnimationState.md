<!-- Auto-generated - do not edit. -->

# AnimationState

One state of the graph: while active it plays either a single
[Animation](Animation.md) `clip` or a `blend` (a blendspace mixing several
clips by parameter value). Exactly one of the two must be set.

## Parameters

- `name`: A string. State name, referenced by `initial` and by transitions.
- `clip`: A string. The [Animation](Animation.md) clip this state plays. Must target the same [SkinnedMesh](SkinnedMesh.md) as the graph. Leave unset when the state plays a `blend` instead.
- `blend`: An object. A blendspace to play instead of a single `clip`. Optional.
- `rate`: A float. Playback speed scale; 1.0 plays at authored speed. Defaults to `1.0`.
- `loop_override`: A boolean. Overrides the loop mode while this state plays: a single `clip` defaults to its own `looping` flag, a `blend` defaults to looping.
