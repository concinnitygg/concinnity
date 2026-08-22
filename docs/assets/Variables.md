<!-- Auto-generated - do not edit. -->

# Variables

The world's shared variables: the state [Behavior](Behavior.md)s read with
`var` and write with `set`, and the state a `save` node persists.

Declaring this asset makes the table authoritative: every variable a
behavior names must appear here, and its declared value fixes both the
variable's type and its starting value. A world without a `Variables` asset
keeps every variable implicit and integer-typed, so declaring one is how a
world opts into typed variables and into catching misspelled names at build
time.

Variables are world-scoped and shared. Per-entity state belongs in a
behavior's `locals`, which are typed the same way but private to one entity
and never persisted.

## Parameters

- `vars`: An array of [VariableDecl](VariableDecl.md) objects. Every variable the world declares.
