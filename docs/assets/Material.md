<!-- Auto-generated - do not edit. -->

# Material

A Material bundles the surface parameters that control how a [Prop](Prop.md) is
lit and shaded.

Reference it from a [Prop](Prop.md)'s `material` field. The `material` field takes
precedence over the older `texture` field.

# Shader parameters

`params` holds eight numbers of your own for a [Shader](Shader.md) to read.
The engine gives them no meaning and its own lighting ignores them; they
exist so that one Shader can be shared by many materials that each set it
up differently: a stripe count, a pulse speed, a blend amount. Every
parameter defaults to 0.

A Shader's `shade` and `transform` read them with `material_param(i)`,
where `i` runs from 0 to 7 and picks `params[i]` of the material the
surface being drawn uses. A surface drawn without a material reads 0 for
all eight.

A Shader's `fragment` file reading a stripe count from `params[0]` and a
scroll speed from `params[1]`:

```hlsl
float4 shade(VertexOut v, GpuObjectData od)
{
    float stripes = material_param(0);
    float speed = material_param(1);
    float band = step(0.5, frac(v.uv.x * stripes + VIEW.elapsed * speed));
    return shade_surface(v, od) * lerp(0.4, 1.0, band);
}
```

## Parameters

- `albedo`: A string. The [Texture](Texture.md) asset used as the base color (albedo) map. Optional.
- `normal_map`: A string. The [Texture](Texture.md) asset used as a tangent-space normal map. Optional.
- `emissive_map`: A string. The [Texture](Texture.md) asset used as an emissive map. Multiplied by `emissive_factor` to drive the glow; when omitted, only the scalar `emissive_factor` is used. Pair a textured emissive with an `emissive_factor` above 1 to make the bright parts bloom.
- `orm_map`: A string. The [Texture](Texture.md) asset used as a packed surface map: green = roughness, blue = metalness. When present it overrides the scalar `roughness` and `metallic` per-texel; when omitted those scalars are used. The red channel is reserved and not read as ambient occlusion: packed maps in the wild (glTF metallic-roughness, FBX specular maps) leave red empty, so treating it as occlusion would darken indirect light to black. Ambient occlusion comes from the screen-space pass.
- `roughness`: A float. Perceptual roughness in [0, 1]. 0 = mirror, 1 = fully diffuse. Controls the width of the specular highlight. Defaults to `0.8`.
- `metallic`: A float. Metallic factor in [0, 1]. 0 = dielectric (plastic/stone), 1 = metal. Metallic surfaces tint their reflections with the albedo color and show almost no diffuse; dielectrics keep a neutral, dim reflection. Defaults to `0.0`.
- `tint`: An array of 3 floats. Linear-space RGB multiplier applied to the albedo sample. Useful for tinting a shared texture without a separate asset (e.g. colored brick). Defaults to `[1.0, 1.0, 1.0]`.
- `emissive_factor`: An array of 3 floats. Additive emission color in linear space. Non-zero values make the surface appear to glow independently of the scene lighting. Defaults to `[0.0, 0.0, 0.0]`.
- `alpha_cutoff`: A float. Alpha-cutout threshold in [0, 1]. When non-zero, a texel whose `albedo` alpha falls below it is discarded outright, punching a hole in the surface: this is how foliage, chain-link, and decal cards are drawn as one opaque quad. 0 (the default) disables the test and keeps every texel. Cutout is not glass: the surface still renders in the opaque pass, so leave `transparent` and `see_through` off.
- `opacity`: A float. Surface opacity in [0, 1]. 1 = fully opaque (the default). Only meaningful when `transparent` is set: it drives how much of the scene behind the surface shows through the glass.
- `transparent`: A boolean. When true, the surface is a translucent dielectric (glass): it renders in the engine's transparent pass instead of the opaque pass, refracting and reflecting the scene rather than writing solid color + depth. The importer sets this for materials it detects as glass; authored materials can opt in directly. Defaults to false (opaque).
- `see_through`: A boolean. When true, the glass is rendered as genuinely see-through: the scene behind it shows through with a sharp per-pixel reflection (requires a ray-tracing-capable GPU). When false (the default), a `transparent` surface still renders as low-roughness reflective glass that hides whatever is behind it. See-through only looks right when the space behind the glass is actually modeled, so it is opt-in per material. Setting it implies `transparent`.
- `shader`: A string. The [Shader](Shader.md) asset that shades surfaces using this material. When omitted, the world's default shader is used. Referencing a shader from a material ties that shader's lifetime to the material's: a shader referenced only by scene-exclusive materials loads and unloads with the scene.
- `params`: An array of 8 floats. Eight numbers for the material's [Shader](Shader.md) to read, as `material_param(0)` through `material_param(7)`. What each one means is up to the Shader; the engine's own shading ignores them. All 0 by default.
