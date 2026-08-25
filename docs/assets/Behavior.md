<!-- Auto-generated - do not edit. -->

# Behavior

A unit of world logic: an event source, the entities it runs against, its
state, the world it reads, and the nodes it runs.

A behavior with an empty `scope` runs once per firing, world-scoped. A
behavior with a `scope` runs once per matching entity, each with its own
copy of `locals`, and `"self"` resolves to that entity.

`locals` and `queries` are declared here and resolved to dense slots once,
when the world starts, so nothing is looked up by name while it runs. A
query naming an unknown component is a build error.

`once` limits a behavior to a single firing; `cooldown` enforces a minimum
number of seconds between firings; `delay` postpones the nodes after the
firing decision, which is made at fire time rather than after the delay.
Timers, delays, and cooldowns freeze while a menu is open, like the rest of
the world clock.

## Parameters

- `on`: An object. The event that fires this behavior.
- `scope`: An array of strings. Component names selecting the entities this behavior runs against. An empty list runs it once, world-scoped, with no `"self"`. The names are matched against the components entities carry while the world runs, which are not always the ones a world declares: the build expands some types away, compiles others into the resource stream, and a load-time pass decomposes the rest. `"Prop"` is the common case and works, resolving to the marker decomposition leaves on every prop's entity, model- and mesh-backed alike. A name with no runtime counterpart is a build error rather than a scope that silently matches nothing.
- `locals`: An array of [BehaviorLocal](BehaviorLocal.md) objects. Per-entity state. Each matching entity gets its own copy, reset to the declared value when the world starts. Locals are never persisted.
- `queries`: An array of [BehaviorQuery](BehaviorQuery.md) objects. World reads, resolved once per tick into a stable-ordered entity list.
- `do`: An array of objects. The nodes run, in order, each time the behavior fires.
- `once`: A boolean. Fire at most once per run.
- `delay`: A float. Seconds between the firing decision and the nodes running (`0` runs them immediately).
- `cooldown`: A float. Minimum seconds between firings (`0` allows every firing).
