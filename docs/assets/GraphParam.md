<!-- Auto-generated - do not edit. -->

# GraphParam

A named float parameter driving a graph's transitions. Gameplay systems
(or the `anim-param` debug command) write parameter values at runtime;
transitions compare against them. Flag-like parameters use 0 and 1.

## Parameters

- `name`: A string. Parameter name, referenced by transition conditions.
- `default`: A float. Initial value at world start.
