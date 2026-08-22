<!-- Auto-generated - do not edit. -->

# Prefab

A reusable template of [Prop](Prop.md)s, [PointLight](PointLight.md)s, and nested
prefabs.

Placed as a unit at a world-space transform. Add a `prefab` field to a
[Prop](Prop.md) to instantiate it; each instance expands into concrete assets
positioned relative to the instance's transform.

**Expanded asset names:** `<instance_name>_<entry_name>` (nested:
`<instance>_<outer>_<inner>`).

**Instantiation:** add a `prefab` field to a [Prop](Prop.md). The prop's other
fields (`position`, `rotation_deg`, `scale`) act as the instance's world
transform.

**Library presets** (JSON files in `assets/prefabs/`):

## Parameters

- `props`: An array of [PrefabEntry](PrefabEntry.md) objects. Ordered list of entries. Each is a prop, a point light, or a nested prefab (selected by `kind`), placed relative to the instance transform.
