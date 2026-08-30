//! The Shader asset: the authored schema (Shader, StageSource, ShaderKind, and
//! the ShaderPayload container), the `Component` impl, and the
//! `StageSourceExt::current_platform_source` extension the engine init and
//! hot-reload paths use. The JSON-args source selection and validation live in
//! concinnity-cook (`authoring::source_args`, `check::shader`).

use crate::ecs::Component;
use crate::ecs::PayloadLocator;
use crate::ecs::asset_id::AssetId;
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

/// Resolve the source filename for the current build platform from a stage's
/// declared `source` / `sources`. Mirrors the build-time selection
/// (concinnity-cook `authoring::source_args`) so the hot-reload subsystem picks the
/// same per-platform source the build read at compile time. Returns `None` when
/// no current-platform source is declared (e.g. a stage that only declares `glsl`
/// running on the Metal backend, which loads the embedded GLSL fallback at init
/// and has no on-disk file to hot-reload). Exposed as an extension trait because
/// the schema type is declared above.
pub trait StageSourceExt {
    /// The source path declared for the running platform, or `None` when the
    /// stage declares none.
    fn current_platform_source(&self) -> Option<String>;
}

impl StageSourceExt for StageSource {
    fn current_platform_source(&self) -> Option<String> {
        let platform = crate::platform::Platform::current();
        if let Some(sources) = &self.sources
            && let Some(src) = sources.get(platform.key())
        {
            return Some(src.clone());
        }
        if self.source.is_empty() {
            return None;
        }
        let ext = super::path_extension(&self.source).unwrap_or("");
        if platform.accepts_ext(ext) {
            Some(self.source.clone())
        } else {
            None
        }
    }
}

impl Component for Shader {
    const NAME: &'static str = "Shader";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(crate::blob::decode_exact(bytes)?)
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }

    fn inject_name(&mut self, id: crate::ecs::asset_id::AssetId) {
        self.asset_id = id;
    }
}

/// Returns the platform key used to look up entries in the `sources` map.
pub fn platform_key() -> &'static str {
    crate::platform::Platform::current().key()
}

#[cfg(test)]
mod runtime_tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn compile_kind_maps_each_stage() {
        assert_eq!(ShaderKind::Vertex.compile_kind(), "vertex");
        assert_eq!(ShaderKind::VertexInstanced.compile_kind(), "vertex");
        assert_eq!(ShaderKind::Fragment.compile_kind(), "fragment");
        assert_eq!(ShaderKind::default(), ShaderKind::Vertex);
    }

    #[test]
    fn current_platform_source_resolves_for_any_backend() {
        // Declaring every platform source resolves on whichever backend the
        // test build targets.
        let stage = StageSource {
            sources: Some(
                [
                    ("metal".to_string(), "v.metal".to_string()),
                    ("hlsl".to_string(), "v.hlsl".to_string()),
                    ("glsl".to_string(), "v.glsl".to_string()),
                ]
                .into_iter()
                .collect(),
            ),
            ..Default::default()
        };
        assert!(stage.current_platform_source().is_some());
    }

    #[test]
    fn single_source_resolves_only_for_matching_extensions() {
        let stage = StageSource {
            source: "v.metal".to_string(),
            sources: None,
        };
        let platform = crate::platform::Platform::current();
        assert_eq!(
            stage.current_platform_source().is_some(),
            platform.accepts_ext("metal")
        );
    }

    // The map is consulted first; a stage declaring only a bare `source` falls
    // back to it, but only when the file's extension is one this platform can
    // actually load. A stage declaring nothing resolves to nothing rather than
    // handing back an empty path.
    #[test]
    fn a_bare_source_resolves_only_when_the_platform_accepts_its_extension() {
        let bare = |source: &str| StageSource {
            source: source.to_string(),
            sources: None,
        };

        assert_eq!(bare("").current_platform_source(), None, "nothing declared");

        // A generic extension is not platform-specific, so it is accepted
        // whichever backend is running.
        assert_eq!(
            bare("v.slang").current_platform_source(),
            Some("v.slang".to_string())
        );

        // A source for a different backend's language resolves to nothing.
        let other = ["metal", "hlsl", "glsl"]
            .into_iter()
            .find(|ext| *ext != platform_key())
            .expect("some other platform exists");
        assert_eq!(
            bare(&alloc::format!("v.{other}")).current_platform_source(),
            None,
            "a {other} source is not loadable here"
        );
    }

    #[test]
    fn the_platform_key_is_the_running_backends_own() {
        assert!(["metal", "hlsl", "glsl"].contains(&platform_key()));
        assert_eq!(platform_key(), crate::platform::Platform::current().key());
    }

    // Shader keeps a hand-written Component impl rather than the generated
    // one, so its identity and payload injection are its own code.
    #[test]
    fn a_shader_takes_its_identity_and_payload_on_load() {
        use crate::ecs::Component;
        use crate::ecs::asset_id::AssetId;

        let bytes = postcard::to_allocvec(&Shader::default()).expect("a shader encodes");
        let mut shader = <Shader as Component>::from_baked(&bytes).expect("it loads back");
        assert_eq!(Shader::NAME, "Shader");

        shader.inject_name(AssetId(4));
        assert_eq!(shader.asset_id, AssetId(4));

        let locator = PayloadLocator {
            blob_index: 1,
            offset: 8,
            len: 16,
        };
        shader.inject_locator(locator.clone());
        assert_eq!(shader.locator, Some(locator));
    }
}
