<!-- Auto-generated - do not edit. -->

# Model

An ordered list of sub-meshes, each with its own material.

Use via the `model` field on a [Prop](Prop.md) instead of `mesh`. Each
sub-mesh is drawn with its own material, all sharing the prop's transform.

Each `mesh` must name a [Mesh](Mesh.md) or [ProceduralMesh](ProceduralMesh.md)
asset present in the scene. `material` may be empty to use the default
material.

## Parameters

- `meshes`: An array of [SubMeshRef](SubMeshRef.md) objects. Ordered list of sub-meshes that make up this model.
