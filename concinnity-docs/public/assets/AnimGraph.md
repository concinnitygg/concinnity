<!-- Auto-generated - do not edit. -->

# AnimGraph

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

```jsonl
// Two single-clip states:
{"name":"hero_graph","type":"AnimGraph","args":{
  "target":"hero",
  "parameters":[{"name":"speed","default":0.0}],
  "initial":"idle",
  "states":[
    {"name":"idle","clip":"hero_idle"},
    {"name":"run","clip":"hero_run","rate":1.1}
  ],
  "transitions":[
    {"from":"idle","to":"run","duration_secs":0.2,
     "conditions":[{"parameter":"speed","op":"gt","value":0.5}]},
    {"from":"run","to":"idle","duration_secs":0.3,
     "conditions":[{"parameter":"speed","op":"le","value":0.5}]}
  ]
}}
// One locomotion blendspace state mixing idle/walk/run by speed:
{"name":"hero_graph","type":"AnimGraph","args":{
  "target":"hero",
  "parameters":[{"name":"speed","default":0.0}],
  "states":[
    {"name":"locomotion","blend":{"kind":"blend1d","parameter":"speed","sync":true,
     "points":[
       {"value":0.0,"clip":"hero_idle"},
       {"value":1.6,"clip":"hero_walk"},
       {"value":5.0,"clip":"hero_run"}
     ]}}
  ]
}}
```

## Parameters

- `target`: A string. The [SkinnedMesh](SkinnedMesh.md) asset this graph animates. Optional.
- `parameters`: An array of [GraphParam](GraphParam.md) objects. Named float parameters transitions compare against.
- `initial`: A string. Name of the state the graph starts in. Defaults to the first state.
- `states`: An array of [GraphState](GraphState.md) objects. The graph's states. At least one is required.
- `transitions`: An array of [GraphTransition](GraphTransition.md) objects. Directed transitions between states.
