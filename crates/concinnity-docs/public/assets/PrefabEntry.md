<!-- Auto-generated - do not edit. -->

# PrefabEntry

One entry in a [Prefab](Prefab.md)'s `props` list. The fields consulted depend on
`kind`: a `prop` uses the render / collision / transform fields, a
`point_light` uses the `light_*` fields, and a `prefab` uses `prefab`. Names
in `model` / `mesh` / `material` / `texture` / `parent` / `prefab` are
unresolved references to other assets, resolved when the entry expands.

## Parameters

- `name`: A string. Entry name; the expanded asset is named `<instance>_<name>`.
- `kind`: A string (see [PrefabKind](PrefabKind.md)). Which asset this entry expands into.
- `position`: An array of 3 floats. Local position relative to the instance transform. Defaults to `[0.0, 0.0, 0.0]`.
- `rotation_deg`: An array of 3 floats. Local rotation, Euler degrees [pitch, yaw, roll], YXZ order. Defaults to `[0.0, 0.0, 0.0]`.
- `scale`: An array of 3 floats. Local scale. Defaults to `[1.0, 1.0, 1.0]`.
- `model`: A string. `prop`: [Model](Model.md) name.
- `mesh`: A string. `prop`: [Mesh](Mesh.md) / [ProceduralMesh](ProceduralMesh.md) name.
- `material`: A string. `prop`: [Material](Material.md) name.
- `texture`: A string. `prop`: [Texture](Texture.md) name (older path; `material` takes priority).
- `parent`: A string. `prop`: parent asset name for the expanded prop.
- `collider`: A [PropCollider](PropCollider.md) object. `prop`: optional collision shape for the expanded prop.
- `interactable`: A boolean. `prop`: whether the expanded prop is interactable. Defaults to `false`.
- `pickup`: A boolean. `prop`: whether the expanded prop is a pickup. Defaults to `false`.
- `light_color`: An array of 3 floats. `point_light`: linear-space RGB colour. Defaults to `[1.0, 1.0, 1.0]`.
- `light_intensity`: A float. `point_light`: intensity multiplier. Defaults to `8.0`.
- `light_range`: A float. `point_light`: maximum reach in world units. Defaults to `6.0`.
- `prefab`: A string. `prefab`: name of another [Prefab](Prefab.md) to expand at this entry's transform.
