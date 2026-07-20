<!-- Auto-generated - do not edit. -->

# Reaction

A declarative logic rule: when an event fires and its conditions pass, run
a list of actions.

A reaction is the world's when/if/then unit. `on` names the event that
fires it, `conditions` gate it against shared integer variables (all must
pass), and `actions` run in order when it fires. Variables start each run
at `0` and are written by the `set` action, so a flag is a variable holding
`1`. Rules chain: a reaction with a `variable` source fires when another
reaction (or any system) changes that variable.

`once` limits a reaction to a single firing; `cooldown` enforces a minimum
number of seconds between firings; `delay` postpones the actions after the
firing decision (conditions are checked at fire time, not after the delay).
Timers, delays, and cooldowns freeze while a menu is open, like the rest of
the world clock.

```jsonl
{"name":"greet","type":"Reaction","args":{"on":"start","actions":[{"set":{"name":"visits","value":1,"add":true}}]}}
{"name":"drip","type":"Reaction","args":{"on":{"timer":{"interval":5.0,"repeat":true}},"actions":[{"spawn":{"template":"drop","position":[0,3,0],"lifetime":4.0}}]}}
{"name":"chime","type":"Reaction","args":{"on":{"variable":"visits"},"conditions":[{"name":"visits","op":"ge","value":3}],"actions":[{"sound":{"clip":"bell"}}],"once":true}}
```

## Parameters

- `on`: An object. The event that fires this reaction: `"start"` (world start), `{"timer": {"interval": seconds, "repeat": bool}}`, or `{"variable": "name"}` (the named variable changed value).
- `conditions`: An array of [Condition](Condition.md) objects. Conditions on shared variables; every one must pass for the reaction to fire. An empty list always passes.
- `actions`: An array of objects. The actions run, in order, each time the reaction fires.
- `once`: A boolean. Fire at most once per run.
- `delay`: A float. Seconds between the firing decision and the actions running (`0` runs them immediately).
- `cooldown`: A float. Minimum seconds between firings (`0` allows every firing).
