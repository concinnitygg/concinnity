<!-- Auto-generated - do not edit. -->

# Condition

A test against one shared variable, gating a [Reaction](Reaction.md). An
unset variable reads as `0`, so a plain flag test is `ne 0` and its
negation `eq 0`.

## Parameters

- `name`: A string. The variable the condition tests.
- `op`: A string (see [CmpOp](CmpOp.md)). How the variable compares against `value`: `eq`, `ne`, `lt`, `le`, `gt`, or `ge`.
- `value`: An integer. The literal compared against.
