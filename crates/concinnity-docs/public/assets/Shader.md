<!-- Auto-generated - do not edit. -->

# Shader

Declares a complete shader program: the vertex and fragment stages every
rendered world needs, plus the optional GPU-instanced vertex stage.

**A rendering world needs at least one Shader.** When a world declares
none, the bundled default set is injected automatically. The shadow pass is
engine-internal (no Shader stage of its own); enable or size it with
`shadow_map_size` in [GraphicsConfig](GraphicsConfig.md).

**Bundled sources:**

- `"default.metal"` / `"default_vert.hlsl"` / `"default_frag.hlsl"`: standard diffuse/specular lighting.

The engine-internal shadow map covers a ±20 m world-space region centred at
the origin with 80 m depth. For larger scenes, increase `shadow_map_size` in
[GraphicsConfig](GraphicsConfig.md) to maintain resolution.

```jsonl
// Multi-platform standard scene:
{"name":"scene_shader","type":"Shader","args":{
  "vertex":{"sources":{"metal":"default.metal","hlsl":"default_vert.hlsl"}},
  "fragment":{"sources":{"metal":"default.metal","hlsl":"default_frag.hlsl"}}}}

// Single-platform (macOS only):
{"name":"scene_shader","type":"Shader","args":{
  "vertex":{"source":"my.metal"},"fragment":{"source":"my.metal"}}}
```

# More than one Shader

The first declared Shader is the world's default: everything renders with it
unless a [Material](Material.md) names another one through its `shader` field.
A world may declare up to 8 Shaders in total.

```jsonl
{"name":"scene_shader","type":"Shader","args":{
  "vertex":{"source":"default.metal"},"fragment":{"source":"default.metal"}}}
{"name":"water_shader","type":"Shader","args":{
  "vertex":{"source":"water.metal"},"fragment":{"source":"water.metal"}}}
{"name":"pond_mat","type":"Material","args":{"shader":"water_shader"}}
```

Three rules come with the second Shader, all enforced at build time:

- **Every fragment stage must define `fragment_main_bindless`.** Multi-Shader
  worlds render through the GPU-driven bindless path, which is the only path
  that can switch programs per draw. A single-Shader world has no such
  requirement and may define just `fragment_main`. This applies to `.metal`
  sources, which carry one program per entry point; an `.hlsl` or GLSL stage
  compiles a single `main`, so there is no entry point to pick -- what it must
  match instead is the bindless binding layout (see below).
- **Instanced, skinned, and voxel-chunk draws always use the world default.**
  A Material naming a Shader cannot be used by an
  [InstancedProp](InstancedProp.md), a [SkinnedMesh](SkinnedMesh.md), or a
  [VoxelWorld](VoxelWorld.md); give those a Material without one.
- **At most 8 Shaders**, the default included.

Planar reflections are the one case with no build-time signal: a surface
reflected in a mirror is drawn with the world default Shader regardless of
its Material. Reflection probe cubes capture it the same way.

A non-default Shader's stages must be written against the engine's **bindless**
binding layout, not the per-draw one: the material, transform, and texture
indices come from the per-frame object buffer rather than per-draw constants.
A Shader that names the engine's own built-in sources (`default.metal`, or
`default_vert.hlsl` + `default_frag.hlsl`) is understood as "render this
material with the engine default program" and is wired to the engine's
bindless program, whichever backend is in use.

A Shader referenced only by materials belonging to one [Scene](Scene.md) is
owned by that scene: its pipeline is built when the scene loads (behind the
loading screen, alongside that scene's textures and meshes) and released when
the scene unloads. A Shader used across scenes, or by the world default,
loads at startup.

**Custom shader vertex layout**: the engine always supplies vertices with 5
attributes at a fixed 56-byte stride. Any custom `.metal` shader **must** declare
`struct Vertex` exactly as shown below: wrong attribute indices cause tangent
data to be read as vertex colour, producing red/green/blue geometry:

```metal
struct Vertex {
    float3 pos     [[attribute(0)]];  // offset  0
    float3 normal  [[attribute(1)]];  // offset 12
    float3 tangent [[attribute(2)]];  // offset 24
    float3 color   [[attribute(3)]];  // offset 36
    float2 uv      [[attribute(4)]];  // offset 48
};
```

Buffer and texture bindings that must match:

```metal
struct DirectionalLightData {
    packed_float3 direction;
    float         intensity;
    packed_float3 color;
    float         _pad;
};

struct PointLightData {
    packed_float3 position;
    float         range;
    packed_float3 color;
    float         intensity;
};

struct ShadowUniforms {
    float4x4 light_vp;
};
```

## Parameters

- `vertex`: A [StageSource](StageSource.md) object. The vertex stage. Required.
- `fragment`: A [StageSource](StageSource.md) object. The fragment stage. Required.
- `vertex_instanced`: A [StageSource](StageSource.md) object. The GPU-instanced vertex stage. Required only for worlds with [InstancedProp](InstancedProp.md) components. Optional.
