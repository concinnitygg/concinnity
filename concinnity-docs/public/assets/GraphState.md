<!-- Auto-generated - do not edit. -->

# GraphState

One state of the graph: a single [Animation](Animation.md) clip played on a
loop (or once) while the state is active.

## Parameters

- `name`: A string. State name, referenced by `initial` and by transitions.
- `clip`: A string. The [Animation](Animation.md) clip this state plays. Must target the same [SkinnedMesh](SkinnedMesh.md) as the graph. Optional.
- `rate`: A float. Playback speed scale; 1.0 plays the clip at its authored speed. Defaults to `1.0`.
- `loop_override`: A boolean. Overrides the clip's own `looping` flag while this state plays. Leave unset to keep the clip's flag.
