// Shader-stage schema.
//
// CRITICAL: `packed_float3` in light structs. In MSL constant buffers `float3`
// has size=16, but Rust `[f32; 3]` (what the engine sends) has size=12. If you
// declare `DirectionalLightData` or `PointLightData` with plain `float3`, the
// color field will read as zeros (black light) and `num_directional` will read
// garbage, causing ambient-only rendering. Always use `packed_float3` for vector
// fields in these structs.

use crate::PayloadLocator;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};

/// Which stage in the render pipeline this shader drives.
///
/// `VertexInstanced` is the GPU-instanced sibling of `Vertex`, reading per-
/// instance model matrices instead of a per-draw transform. Required for any
/// world containing [InstancedProp](#instancedprop) components; otherwise
/// unused.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ShaderKind {
    #[default]
    Vertex,
    Fragment,
    #[serde(rename = "vertex_instanced", alias = "vertexinstanced")]
    VertexInstanced,
}

impl ShaderKind {
    /// The compile kind string expected by ShaderCompileArgs.
    pub fn compile_kind(&self) -> &'static str {
        match self {
            ShaderKind::Vertex | ShaderKind::VertexInstanced => "vertex",
            ShaderKind::Fragment => "fragment",
        }
    }
}

/// Declares a compiled shader stage.
///
/// **Vertex and fragment stages are required for anything to render.** The
/// shadow pass is engine-internal (no `ShaderStage` of its own); enable or
/// size it with `shadow_map_size` in [GraphicsConfig](#graphicsconfig).
///
/// Provide either `source` (single platform) or `sources` (multi-platform). When both are
/// present, `sources` takes priority for the current platform.
///
/// **Platform keys:** `"metal"` (macOS), `"hlsl"` (Windows), `"glsl"` (Linux/Vulkan).
///
/// **Bundled shaders:**
///
/// - `"default.metal"` / `"default_vert.hlsl"` / `"default_frag.hlsl"`: standard diffuse/specular lighting.
///
/// The engine-internal shadow map covers a ±20 m world-space region centred at
/// the origin with 80 m depth. For larger scenes, increase `shadow_map_size` in
/// [GraphicsConfig](#graphicsconfig) to maintain resolution.
///
/// ```jsonl
/// // Multi-platform standard scene:
/// {"name":"vert","type":"ShaderStage","args":{"kind":"vertex","sources":{"metal":"default.metal","hlsl":"default_vert.hlsl"}}}
/// {"name":"frag","type":"ShaderStage","args":{"kind":"fragment","sources":{"metal":"default.metal","hlsl":"default_frag.hlsl"}}}
///
/// // Single-platform (macOS only):
/// {"name":"vert","type":"ShaderStage","args":{"kind":"vertex","source":"default.metal"}}
/// ```
///
/// **Custom shader vertex layout**: the engine always supplies vertices with 5
/// attributes at a fixed 56-byte stride. Any custom `.metal` shader **must** declare
/// `struct Vertex` exactly as shown below: wrong attribute indices cause tangent
/// data to be read as vertex colour, producing red/green/blue geometry:
///
/// ```metal
/// struct Vertex {
///     float3 pos     [[attribute(0)]];  // offset  0
///     float3 normal  [[attribute(1)]];  // offset 12
///     float3 tangent [[attribute(2)]];  // offset 24
///     float3 color   [[attribute(3)]];  // offset 36
///     float2 uv      [[attribute(4)]];  // offset 48
/// };
/// ```
///
/// Buffer and texture bindings that must match:
///
/// ```metal
/// struct DirectionalLightData {
///     packed_float3 direction;
///     float         intensity;
///     packed_float3 color;
///     float         _pad;
/// };
///
/// struct PointLightData {
///     packed_float3 position;
///     float         range;
///     packed_float3 color;
///     float         intensity;
/// };
///
/// struct ShadowUniforms {
///     float4x4 light_vp;
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShaderStage {
    /// Which stage this shader drives.
    pub kind: ShaderKind,
    /// Single-platform source path; used when `sources` is absent or lacks the current platform key.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
    /// Per-platform source paths keyed by `"metal"`, `"hlsl"`, or `"glsl"`. Takes priority over `source`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<BTreeMap<String, String>>,
    /// Injected at load time from BlobAssetDef::payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for ShaderStage {
    fn default() -> Self {
        let mut sources = BTreeMap::new();
        sources.insert("metal".to_string(), "default.metal".to_string());
        sources.insert("hlsl".to_string(), "default_vert.hlsl".to_string());
        Self {
            kind: ShaderKind::Vertex,
            source: String::new(),
            sources: Some(sources),
            locator: None,
        }
    }
}
