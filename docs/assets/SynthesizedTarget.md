<!-- Auto-generated - do not edit. -->

# SynthesizedTarget

A morph target the build generates from the mesh instead of reading from
the source.

## Parameters

- `name`: A string. Slider name. A bipolar target emits `name+` and its negation `name-`. Defaults to `""`.
- `generator`: A string. Generator: `girth`, `taper`, `bulge`, `mirror`, `blend_mask`, or `surface_offset`. Defaults to `""`.
- `region`: A string. The region the generator works in and the key is grouped under. Defaults to `""`.
- `polarity`: A string (see [KeyPolarity](KeyPolarity.md)). One target or a pair. Defaults to `"unipolar"`.
- `caption`: A string. Panel caption; the name when empty. Defaults to `""`.
- `params`: A [SynthParams](SynthParams.md) object. Generator parameters.
