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
