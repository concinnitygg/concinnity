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

```jsonl
// Define the template:
{"type":"Prefab","name":"table_set","args":{"props":[
  {"name":"table","kind":"prop","model":"model_table","position":[0,0,0]},
  {"name":"chair_n","kind":"prop","model":"model_chair","position":[0,0,0.7],"rotation_deg":[0,180,0]},
  {"name":"chair_s","kind":"prop","model":"model_chair","position":[0,0,-0.7]},
  {"name":"lamp","kind":"point_light","position":[0,2.2,0],"light_color":[1.0,0.9,0.7],"light_intensity":6.0,"light_range":5.0}
]}}

// Place two instances:
{"type":"Prop","name":"dining_a","args":{"prefab":"table_set","position":[3,0,-5]}}
{"type":"Prop","name":"dining_b","args":{"prefab":"table_set","position":[-3,0,-5],"rotation_deg":[0,45,0]}}
```

**Library presets** (JSON files in `assets/prefabs/`):

```jsonl
// From library preset:
{"type":"Prop","name":"table_a","args":{"prefab":"prefab_table_4chair","position":[0,0,-6]}}
```

## Parameters

- `props`: An array of [PrefabEntry](PrefabEntry.md) objects. Ordered list of entries. Each is a prop, a point light, or a nested prefab (selected by `kind`), placed relative to the instance transform.
