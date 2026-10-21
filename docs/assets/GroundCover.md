<!-- Auto-generated - do not edit. -->

# GroundCover

A look a [Planet](Planet.md)'s ground takes where its `rule` holds: sand on
a beach, a duller seabed under deeper water, rock on steep slopes.

The cover's `material` replaces the planet's own where the rule holds and
fades back to it over the rule's soft edges. Its color (`tint` over its
`albedo` texture), `roughness` and `metallic` are used; the planet's
`material` keeps the normal map and every other map.

## Parameters

- `material`: A string. The [Material](Material.md) the ground takes where the rule holds. Optional.
- `rule`: A [GroundRule](GroundRule.md) object. Where the cover lies.
