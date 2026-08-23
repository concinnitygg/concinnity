<!-- Auto-generated - do not edit. -->

# CharacterShape

Shape sliders and joint proportions applied to one [SkinnedMesh](SkinnedMesh.md).

Every characteristic of the shape is data on the mesh, not code: a slider
drives one or two of the mesh's morph targets, and a proportion scales or
lengthens one joint of its skeleton. The deformation is static and sits
under any [Animation](Animation.md) playing on the same mesh: clip morph
tracks are added on top of the slider weights, and clip poses are
re-proportioned every frame.

**Sliders** resolve to morph targets by name. A target named exactly
`name` is unipolar and receives the slider value clamped to `[0, 1]`. A
pair named `name+` / `name-` is bipolar: a positive value drives `name+`,
a negative value drives `name-` by its magnitude. A slider with no matching
target is reported as a build warning and ignored.

**Proportions** resolve to joints by name. `scale` is uniform (the
skinning shaders transform normals with the plain joint matrix, so a
non-uniform scale would shade incorrectly) and propagates to the joint's
descendants; `length` moves only the joint's children along the bone, so a
longer thigh does not also stretch the shin. Proportions change the posed
skeleton, not the bind pose, so clips with translation tracks on the
affected joints fight them; keep such rigs rotation-only. When the mesh
declares a `capsule`, the capsule's half-height follows the skeleton's
height change and its radius follows the root joint's scale.

`target` may name a [CharacterModel](CharacterModel.md) as well as a
`SkinnedMesh`; the model's emitted mesh is what the shape deforms.

**Baking.** With `bake` set, the build flattens the shape into its target:
the sliders' deformation is applied to the vertices and the morph targets
dropped, the bind pose is rewritten through the proportions, the capsule
is resized, and this asset is consumed. The result is a plain `SkinnedMesh`
with no per-frame shape work, for characters that never change shape.

## Parameters

- `target`: A string. The [SkinnedMesh](SkinnedMesh.md) this shape deforms. Optional.
- `sliders`: An array of [ShapeSlider](ShapeSlider.md) objects. Named shape values, each resolved to the mesh's morph targets.
- `proportions`: An array of [JointProportion](JointProportion.md) objects. Per-joint scale and length changes.
- `bake`: A boolean. Flatten the shape into the target mesh at build time and drop this asset, instead of deforming at runtime.
