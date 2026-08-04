<!-- Auto-generated - do not edit. -->

# PrefabKind

Which kind of asset a [PrefabEntry](PrefabEntry.md) expands into.

## Values

- `prop`: A [Prop](Prop.md) built from the entry's `model` / `mesh` / `material` / `texture` and transform fields.
- `point_light`: A [PointLight](PointLight.md) built from the entry's `light_*` fields at the entry's `position`.
- `prefab`: A nested prefab named by the entry's `prefab` field, expanded relative to this entry's transform.
