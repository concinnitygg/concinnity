<!-- Auto-generated - do not edit. -->

# AnimationTransition

One directed transition between two states.

## Parameters

- `from`: A string. Source state name.
- `to`: A string. Destination state name.
- `duration_secs`: A float. Crossfade length in seconds between the outgoing and incoming poses. Zero snaps to the new state's pose immediately.
- `exit_time`: A float. When set (0 to 1), the transition waits until the source state has played this fraction of its clip. On a looping state the gate re-opens every loop; on a non-looping state it stays open once reached. Useful for letting a clip finish before leaving, e.g. `0.9` on a jump. Optional.
- `conditions`: An array of [AnimationCondition](AnimationCondition.md) objects. Conditions that must all pass (in addition to any `exit_time` gate). An empty list always passes.
