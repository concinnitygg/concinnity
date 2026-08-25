// Shader schema: one asset = one complete shader program (all stages).
//
// CRITICAL: `packed_float3` in light structs. In MSL constant buffers `float3`
// has size=16, but Rust `[f32; 3]` (what the engine sends) has size=12. If you
// declare `DirectionalLightData` or `PointLightData` with plain `float3`, the
// color field will read as zeros (black light) and `num_directional` will read
// garbage, causing ambient-only rendering. Always use `packed_float3` for vector
// fields in these structs.

use crate::{AssetId, PayloadLocator};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A stage slot within a [Shader](#shader).
///
/// `VertexInstanced` is the GPU-instanced sibling of `Vertex`, reading per-
/// instance model matrices instead of a per-draw transform. Required for any
/// world containing [InstancedProp](#instancedprop) components; otherwise
/// unused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ShaderKind {
    /// Vertex stage for a per-draw transform.
    #[default]
    Vertex,
    /// Fragment stage.
    Fragment,
    /// Vertex stage reading per-instance model matrices.
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

/// Source declaration for one stage of a [Shader](#shader).
///
/// Provide either `source` (single platform) or `sources` (multi-platform).
/// When both are present, `sources` takes priority for the current platform.
///
/// **Platform keys:** `"metal"` (macOS), `"hlsl"` (Windows), `"glsl"` (Linux/Vulkan).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StageSource {
    /// Single-platform source path; used when `sources` is absent or lacks the current platform key.
    #[serde(default)]
    pub source: String,
    /// Per-platform source paths keyed by `"metal"`, `"hlsl"`, or `"glsl"`. Takes priority over `source`.
    #[serde(default)]
    pub sources: Option<BTreeMap<String, String>>,
}

impl StageSource {}

/// Declares a custom shader program: the vertex and fragment stages, plus the
/// optional GPU-instanced vertex stage.
///
/// **A Shader is entirely optional.** The engine ships its own main-pass
/// program and uses it for every draw a Shader does not claim, so a world that
/// wants standard lighting declares no Shader at all. Declare one only to
/// replace that program with your own. The shadow pass is engine-internal (no
/// Shader stage of its own); enable or size it with `shadow_map_size` in
/// [GraphicsConfig](#graphicsconfig).
///
/// The engine-internal shadow map covers a ±20 m world-space region centred at
/// the origin with 80 m depth. For larger scenes, increase `shadow_map_size` in
/// [GraphicsConfig](#graphicsconfig) to maintain resolution.
///
/// A stage that resolves no source for the running backend falls back to the
/// engine's own program for that stage, so a Shader may cover only the
/// platforms it has sources for.
///
/// # More than one Shader
///
/// The first declared Shader is the world's default: everything renders with it
/// unless a [Material](#material) names another one through its `shader` field.
/// A world may declare up to 8 Shaders in total.
///
/// Three rules come with the second Shader, all enforced at build time:
///
/// - **Every fragment stage must define `fragment_main_bindless`.** Multi-Shader
///   worlds render through the GPU-driven bindless path, which is the only path
///   that can switch programs per draw. A single-Shader world has no such
///   requirement and may define just `fragment_main`. This applies to `.metal`
///   sources, which carry one program per entry point; an `.hlsl` or GLSL stage
///   compiles a single `main`, so there is no entry point to pick -- what it must
///   match instead is the bindless binding layout (see below).
/// - **Instanced, skinned, and voxel-chunk draws always use the world default.**
///   A Material naming a Shader cannot be used by an
///   [InstancedProp](#instancedprop), a [SkinnedMesh](#skinnedmesh), or a
///   [VoxelWorld](#voxelworld); give those a Material without one.
/// - **At most 8 Shaders**, the world default included.
///
/// Planar reflections are the one case with no build-time signal: a surface
/// reflected in a mirror is drawn with the world default Shader regardless of
/// its Material. Reflection probe cubes capture it the same way.
///
/// A non-default Shader's stages must be written against the engine's **bindless**
/// binding layout, not the per-draw one: the material, transform, and texture
/// indices come from the per-frame object buffer rather than per-draw constants.
///
/// A Shader referenced only by materials belonging to one [Scene](#scene) is
/// owned by that scene: its pipeline is built when the scene loads (behind the
/// loading screen, alongside that scene's textures and meshes) and released when
/// the scene unloads. A Shader used across scenes, or by the world default,
/// loads at startup.
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Shader {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The vertex stage. Required.
    pub vertex: StageSource,
    /// The fragment stage. Required.
    pub fragment: StageSource,
    /// The GPU-instanced vertex stage. Required only for worlds with
    /// [InstancedProp](#instancedprop) components.
    #[serde(default)]
    pub vertex_instanced: Option<StageSource>,
    /// Injected at load time from BlobAssetDef::payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Shader {
    /// The declared source for `kind`, if that stage is present.
    pub fn stage(&self, kind: ShaderKind) -> Option<&StageSource> {
        match kind {
            ShaderKind::Vertex => Some(&self.vertex),
            ShaderKind::Fragment => Some(&self.fragment),
            ShaderKind::VertexInstanced => self.vertex_instanced.as_ref(),
        }
    }
}

/// The compiled payload a [`Shader`] carries in the blob: every compiled stage,
/// tagged by kind. Written by the cook, decoded once by the renderer at load.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShaderPayload {
    /// The compiled bytes of each stage, tagged by kind.
    pub stages: Vec<(ShaderKind, Vec<u8>)>,
}

impl ShaderPayload {
    /// Serialize the payload for the blob.
    pub fn encode(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Read a payload back out of the blob.
    pub fn decode(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    /// The compiled bytes for `kind`, if that stage was compiled.
    pub fn stage(&self, kind: ShaderKind) -> Option<&[u8]> {
        self.stages
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, b)| b.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn payload_round_trips_and_indexes_by_kind() {
        let payload = ShaderPayload {
            stages: vec![
                (ShaderKind::Vertex, vec![1, 2, 3]),
                (ShaderKind::Fragment, vec![4, 5]),
            ],
        };
        let bytes = payload.encode().expect("encode");
        let decoded = ShaderPayload::decode(&bytes).expect("decode");
        assert_eq!(decoded, payload);
        assert_eq!(decoded.stage(ShaderKind::Vertex), Some(&[1u8, 2, 3][..]));
        assert_eq!(decoded.stage(ShaderKind::Fragment), Some(&[4u8, 5][..]));
        assert_eq!(decoded.stage(ShaderKind::VertexInstanced), None);
    }

    #[test]
    fn stage_lookup_covers_every_kind() {
        let s = Shader::default();
        assert!(s.stage(ShaderKind::Vertex).is_some());
        assert!(s.stage(ShaderKind::Fragment).is_some());
        assert!(s.stage(ShaderKind::VertexInstanced).is_none());
    }

    #[test]
    fn the_instanced_vertex_stage_compiles_as_a_vertex_stage() {
        assert_eq!(ShaderKind::Vertex.compile_kind(), "vertex");
        assert_eq!(ShaderKind::VertexInstanced.compile_kind(), "vertex");
        assert_eq!(ShaderKind::Fragment.compile_kind(), "fragment");
        assert_eq!(ShaderKind::default(), ShaderKind::Vertex);
    }

    #[test]
    fn stage_kinds_parse_from_their_authored_spellings() {
        let kind = |s: &str| serde_json::from_str::<ShaderKind>(s).unwrap();
        assert_eq!(kind(r#""vertex""#), ShaderKind::Vertex);
        assert_eq!(kind(r#""fragment""#), ShaderKind::Fragment);
        assert_eq!(kind(r#""vertex_instanced""#), ShaderKind::VertexInstanced);
        // The unseparated spelling is accepted as an alias.
        assert_eq!(kind(r#""vertexinstanced""#), ShaderKind::VertexInstanced);
        assert_eq!(
            serde_json::to_string(&ShaderKind::VertexInstanced).unwrap(),
            r#""vertex_instanced""#
        );
    }

    #[test]
    fn a_shader_parses_from_authored_args() {
        let s: Shader = serde_json::from_str(
            r#"{"vertex":{"sources":{"metal":"my.metal"}},"fragment":{"source":"my.metal"}}"#,
        )
        .unwrap();
        assert_eq!(s.fragment.source, "my.metal");
        assert_eq!(
            s.vertex.sources.as_ref().expect("per-platform")["metal"],
            "my.metal"
        );
        // A world with no instanced props declares no instanced vertex stage.
        assert!(s.vertex_instanced.is_none());
        // The identity and payload locator are injected, never authored.
        assert_eq!(s.asset_id, AssetId::default());
        assert!(s.locator.is_none());

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: Shader = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.fragment.source, "my.metal");
    }

    #[test]
    fn an_empty_payload_has_no_stages() {
        let payload = ShaderPayload::default();
        assert!(payload.stages.is_empty());
        assert_eq!(payload.stage(ShaderKind::Vertex), None);
        assert_eq!(
            ShaderPayload::decode(&payload.encode().unwrap()),
            Ok(payload)
        );
    }

    #[test]
    fn decoding_garbage_is_an_error_not_a_panic() {
        assert!(ShaderPayload::decode(&[0xff, 0xff, 0xff]).is_err());
    }
}
