<!-- Auto-generated - do not edit. -->

# AnimationGraph

An animation state machine for one [SkinnedMesh](SkinnedMesh.md).

While a plain set of [Animation](Animation.md) clips blends every clip all
the time, a graph plays exactly one *state* at a time and moves between
states along declared transitions, crossfading poses over each
transition's `duration_secs`. Transitions fire when their conditions --
comparisons against the graph's named float `parameters` -- pass. Gameplay
systems write parameter values each frame (the `anim-param` debug command
does the same from a `cn debug` session).

A graph owns its target: every [Animation](Animation.md) targeting the
graph's mesh must be referenced by exactly one state, and at most one
graph may target a given mesh (both are build errors otherwise). Clip
`weight` and `fade_in_secs` have no effect under a graph.

Transitions are checked in declaration order and the first match wins.
A state with no outgoing transitions (or none passing) keeps playing;
looping states wrap, non-looping states hold their final pose.

## Parameters

- `target`: A string. The [SkinnedMesh](SkinnedMesh.md) asset this graph animates. Optional.
- `parameters`: An array of [AnimationParam](AnimationParam.md) objects. Named float parameters transitions compare against.
- `initial`: A string. Name of the state the graph starts in. Defaults to the first state.
- `states`: An array of [AnimationState](AnimationState.md) objects. The graph's states. At least one is required.
- `transitions`: An array of [AnimationTransition](AnimationTransition.md) objects. Directed transitions between states.
- `ik_chains`: An array of [AnimationIkChain](AnimationIkChain.md) objects. Two-bone IK chains applied on top of every state's pose; see [AnimationIkChain](AnimationIkChain.md).
