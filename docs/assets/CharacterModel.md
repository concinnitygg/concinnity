<!-- Auto-generated - do not edit. -->

# CharacterModel

A character body that conforms to a [CharacterSchema](CharacterSchema.md).

The build validates the source against the schema (joint names and
parentage, complete `+` / `-` pairs for bipolar keys), imports it,
generates the schema's synthesized targets, and emits one
[SkinnedMesh](SkinnedMesh.md) under this asset's name. A
[CharacterShape](CharacterShape.md) or [Animation](Animation.md) targets the
model by this name exactly as it would a `SkinnedMesh`. Lower levels of
detail come from `lod_levels`, as on a `SkinnedMesh`.

The source's extra shape keys (ones the schema does not list) are
imported and appear under the editor panel's "Other" section.

## Parameters

- `schema`: A string. The [CharacterSchema](CharacterSchema.md) the source conforms to, by asset name or the reserved `builtin:humanoid`.
- `source`: A string. Path to the `.glb` / `.gltf` body.
- `skin_index`: An integer. Which skinned mesh of `source` to import, in file order. Defaults to `0`.
- `material`: A string. [Material](Material.md) of the emitted mesh. Optional.
- `position`: An array of 3 floats. World-space position. Defaults to `[0.0, 0.0, 0.0]`.
- `rotation_deg`: An array of 3 floats. World rotation, Euler degrees [pitch, yaw, roll], YXZ order. Defaults to `[0.0, 0.0, 0.0]`.
- `scale`: An array of 3 floats. World scale. Defaults to `[1.0, 1.0, 1.0]`.
- `lod_levels`: An integer. Number of level-of-detail versions to generate, including the original. `1` (the default) generates none; values are clamped to `[1, 8]`.
- `lod_distances`: An array of floats. Camera distances at which to switch to each lower-detail version. When non-empty, must have exactly `lod_levels - 1` entries; empty lets the build choose defaults.
- `max_instances`: An integer. Runtime copies the mesh may spawn beyond the authored one. Defaults to `0`.
- `capsule`: A [CharacterCapsule](CharacterCapsule.md) object. Character capsule of the emitted mesh. Optional.
