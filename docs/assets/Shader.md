<!-- Auto-generated - do not edit. -->

# Shader

Replaces how surfaces are shaded, and optionally how vertices are placed,
with functions of your own. Written in HLSL, one source for every
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
float4 shade(VertexOut v, GpuObjectData od);

// the `vertex` file, optional; without one the engine projects the vertex itself
VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
                    float3 color, float2 uv);
```

`shade` returns the surface's linear-light color with alpha. `od` is the
surface's material record whichever path drew it: `tint_roughness`,
`emissive_metallic`, `albedo_index`, `normal_index`, `emissive_map_index`,
`orm_map_index` and `bb_max_alpha_cutoff.w` are the fields a surface
reads. `transform` receives the model matrix and the model-space
attributes, after skinning for a [SkinnedMesh](SkinnedMesh.md) and per
instance for an [InstancedProp](InstancedProp.md), and returns the projected
vertex; the engine's own is `project_vertex`, so a displacement is
`return project_vertex(model, pos + offset, normal, tangent, color, uv);`.
The same `transform` places the surface's depth, normal and motion for the
screen-space effects. Its motion comes from the returned `world_pos`, not
its clip-space `position`, found by running it again with the previous
frame's model, position, `VIEW.elapsed` and camera position, so a
displacement should depend on nothing else that changes between frames.

Those effects (ambient occlusion, screen-space reflections and global
illumination, and an upscaler's depth) see the surface the material
record describes: its alpha cutout, normal map and roughness. A `discard`
or a normal or roughness of `shade`'s own changes the lit color only.

Both files are compiled inside the engine's own main-pass source, so they
see the same vocabulary the engine's shading uses and declare no layout,
binding, register, attribute or varying of their own:

- `shade_surface(v, od)`: the engine's PBR lighting, so
  `return shade_surface(v, od) * tint;` starts from it.
- `project_vertex(model, pos, normal, tangent, color, uv)`: the engine's
  projection.
- `material_param(index)`: parameter `index`, 0 to 7, of the `params` the
  surface's [Material](Material.md) sets, from either hook; 0 for a surface
  drawn without a material.
- `pool_sample(index, uv)`: a texture from the world's pool by the record's
  index.
- `decode_normal_map(rg)`: a tangent-space normal from a normal-map texel.
- `shadow_factor_cascaded(world_pos, view_depth, screen_xy)`: the sun's
  cascaded shadow term.
- `environment_specular(probe_mask_all(), world_pos, reflected, roughness,
  radiance)`: the reflection environment for a surface of `roughness` into
  `radiance`, false where the world has neither a reflection probe nor an
  environment map.
- `irradiance_sample(normal)`: the diffuse environment.
- `VIEW`: the view block, with `vp`, `view_mat`, `elapsed`, `cam_x` /
  `cam_y` / `cam_z` and `sky_rot`.
- `LIGHTS`: the light block, with `dir[]`, `pt[]`, `num_dir`, `num_pt` and
  `ambient_intensity`.
- `SKY_DIR(d)`: a world direction in the environment map's frame.

`VertexOut` is the engine's varying block: `position` (clip), `world_pos`,
`normal`, `tangent`, `bitangent`, `uv`, `view_depth` and `color`. A `shade`
must not read `v.object_id`; the record is `od`.

# Parameters from the Material

A Shader declares no inputs of its own. Each [Material](Material.md) instead
carries eight numbers, its `params`, which `material_param(0)` through
`material_param(7)` read for the surface being drawn, so one Shader can be
set up differently by every material that uses it. What each parameter
means is the Shader's to decide, and worth a comment at the top of its
file:

```hlsl
// material_param(0): glow strength, material_param(1): pulses per second
float4 shade(VertexOut v, GpuObjectData od)
{
    float pulse = 0.5 + 0.5 * sin(VIEW.elapsed * 6.2831 * material_param(1));
    return shade_surface(v, od) + float4(od.tint_roughness.rgb * material_param(0) * pulse, 0.0);
}
```

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
hook, with each compiler error reported at its file and line; a compiler
warning is logged the same way and the build goes on.

## Parameters

- `fragment`: A string. Path to the `.hlsl` file defining `shade`. Required.
- `vertex`: A string. Path to the `.hlsl` file defining `transform`. Omit to keep the engine's own projection.
