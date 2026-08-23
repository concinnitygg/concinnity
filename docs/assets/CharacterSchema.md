<!-- Auto-generated - do not edit. -->

# CharacterSchema

The contract between a character body and everything that uses it.

A schema names the joints a conforming skeleton must have (with their
parents), the shape keys a conforming mesh carries, the regions those keys
and the editor group by, the proportion rows the editor offers, the morph
targets the build synthesizes from the mesh, the panel's section order,
and the presets it offers. A [CharacterModel](CharacterModel.md) names one
schema and is validated against it at build time, so any conforming body
gets the same sliders, panel, and animations.

**Regions** are joint groups. A vertex belongs to a region by the skin
weight it gives the region's joints, which needs no authoring and holds
at any vertex count. Regions scope every synthesized target and group the
panel.

**Synthesized targets** are ordinary morph targets the build generates:
`girth` pushes a region's vertices away from its bone axes, `taper` ramps
that push along each bone, `bulge` raises a gaussian lobe at a point along
a bone, `mirror` reflects an authored target across X, `blend_mask`
restricts an authored whole-body target to a region, and `surface_offset`
pushes along the vertex normal. Normals are recomputed from the displaced
mesh. At runtime they are indistinguishable from sculpted keys.

The reserved name `builtin:humanoid` is the schema of the humanoid body
the `customize_character` example ships (`base_humanoid.glb`), bundled
with the build so any body with the same 25 joints and 21 shape keys
conforms to it.

## Parameters

- `joints`: An array of [SchemaJoint](SchemaJoint.md) objects. Required (and optional) joints with their parents.
- `keys`: An array of [SchemaKey](SchemaKey.md) objects. Shape keys a conforming source carries.
- `regions`: An array of [SchemaRegion](SchemaRegion.md) objects. Named joint groups.
- `proportion_groups`: An array of [ProportionGroup](ProportionGroup.md) objects. Proportion rows.
- `synthesized`: An array of [SynthesizedTarget](SynthesizedTarget.md) objects. Targets the build generates from the mesh.
- `panel`: An array of [PanelSection](PanelSection.md) objects. Panel sections in display order. Regions no section lists, and keys the schema does not know, show under a trailing "Other" section.
- `presets`: An array of [ShapePreset](ShapePreset.md) objects. Named slider vectors offered as buttons.
