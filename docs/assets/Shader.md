<!-- Auto-generated - do not edit. -->

# Shader

Replaces how surfaces are shaded, and optionally how vertices are placed,
with functions of your own. Written in Slang, one source for every
backend.

**A Shader is entirely optional.** The engine ships its own lighting and
projection and uses them for every draw a Shader does not claim, so a world
that wants standard lighting declares no Shader at all. The shadow pass and
the depth pre-pass are engine-internal and take no Shader stage; enable or
size shadows with `shadow_map_size` in [GraphicsConfig](GraphicsConfig.md).

# The two hooks

A Shader file defines a function, not an entry point. The engine owns every
entry point, binding and pipeline on every backend, and calls the world's
functions from inside its own:

```hlsl
// the `fragment` file, required
float4 shade(VertexOut in, GpuObjectData od);

// the `vertex` file, optional; without one the engine projects the vertex itself
VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
                    float3 color, float2 uv);
```

`shade` returns the surface's linear-light colour with alpha. `od` is the
surface's material record whichever path drew it: `tint_roughness`,
`emissive_metallic`, `albedo_index`, `normal_index`, `emissive_map_index`,
`orm_map_index` and `bb_max_alpha_cutoff.w` are the fields a surface
reads. `transform` receives the model matrix and the model-space
attributes, after skinning for a [SkinnedMesh](SkinnedMesh.md) and per
instance for an [InstancedProp](InstancedProp.md), and returns the projected
vertex; the engine's own is `project_vertex`, so a displacement is
`return project_vertex(model, pos + offset, normal, tangent, color, uv);`.

Both files are compiled inside the engine's own main-pass source, so they
see the same vocabulary the engine's shading uses and declare no layout,
binding, register, attribute or varying of their own:

- `shade_surface(in, od)`: the engine's PBR lighting, so
  `return shade_surface(in, od) * tint;` starts from it.
- `project_vertex(model, pos, normal, tangent, color, uv)`: the engine's
  projection.
- `pool_sample(index, uv)`: a texture from the world's pool by the record's
  index.
- `decode_normal_map(rg)`: a tangent-space normal from a normal-map texel.
- `shadow_factor_cascaded(world_pos, view_depth, screen_xy)`: the sun's
  cascaded shadow term.
- `environment_specular(world_pos, reflected, lod)`: the reflection
  environment.
- `irradiance_sample(normal)`: the diffuse environment.
- `VIEW`: the view block, with `vp`, `view_mat`, `elapsed`, `cam_x` /
  `cam_y` / `cam_z` and `sky_rot`.
- `LIGHTS`: the light block, with `dir[]`, `pt[]`, `num_dir`, `num_pt` and
  `ambient_intensity`.
- `SKY_DIR(d)`: a world direction in the environment map's frame.

`VertexOut` is the engine's varying block: `position` (clip), `world_pos`,
`normal`, `tangent`, `bitangent`, `uv`, `view_depth` and `color`. A `shade`
must not read `in.object_id`; the record is `od`.

# More than one Shader

The first declared Shader is the world's default: everything renders with it
unless a [Material](Material.md) names another one through its `shader` field.
A world may declare up to 8 Shaders in total.

- **Instanced, skinned, and voxel-chunk draws always use the world default.**
  A Material naming a Shader cannot be used by an
  [InstancedProp](InstancedProp.md), a [SkinnedMesh](SkinnedMesh.md), or a
  [VoxelWorld](VoxelWorld.md); give those a Material without one.
- **At most 8 Shaders**, the world default included.

Planar reflections are the one case with no build-time signal: a surface
reflected in a mirror is drawn with the world default Shader regardless of
its Material. Reflection probe cubes capture it the same way.

A Shader referenced only by materials belonging to one [Scene](Scene.md) is
owned by that scene: its pipeline is built when the scene loads (behind the
loading screen, alongside that scene's textures and meshes) and released when
the scene unloads. A Shader used across scenes, or by the world default,
loads at startup.

# Compilation

`cn build` compiles both files for the backend it cooks for and stores the
result in the world; a player needs no shader compiler. A file that fails
to compile, or omits its hook, fails the build naming the Shader and the
hook. Under `cn debug` a save to either file recompiles it and swaps the
live pipelines.

## Parameters

- `fragment`: A string. Path to the `.slang` file defining `shade`. Required.
- `vertex`: A string. Path to the `.slang` file defining `transform`. Omit to keep the engine's own projection.
