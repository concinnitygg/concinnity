//! The Shader asset: the authored schema (Shader, ShaderStage, and the
//! ShaderPrograms container the cook fills), and the `Component` impl. The
//! compile lives in concinnity-cook (`compile::shader`); which programs a
//! world shader compiles to is `render::slang_programs::surface`.

use crate::ecs::Component;
use crate::ecs::PayloadLocator;
use crate::ecs::asset_id::AssetId;
use alloc::string::String;
use alloc::vec::Vec;

use super::compiled_programs::CompiledProgram;

/// One of the two files a [Shader](#shader) declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShaderStage {
    /// The `vertex` file, defining `transform`.
    Vertex,
    /// The `fragment` file, defining `shade`.
    Fragment,
}

/// Replaces how surfaces are shaded, and optionally how vertices are placed,
/// with functions of your own. Written in Slang, one source for every
/// backend.
///
/// **A Shader is entirely optional.** The engine ships its own lighting and
/// projection and uses them for every draw a Shader does not claim, so a world
/// that wants standard lighting declares no Shader at all. The shadow pass and
/// the depth pre-pass are engine-internal and take no Shader stage; enable or
/// size shadows with `shadow_map_size` in [GraphicsConfig](#graphicsconfig).
///
/// ```rust
/// # use concinnity_core::components::Shader;
/// // Custom shading only; the engine still places every vertex.
/// let water = Shader {
///     fragment: "assets/shaders/water.slang".into(),
///     ..Default::default()
/// };
/// // Both hooks: a sway displacement, then the surface.
/// let reeds = Shader {
///     vertex: Some("assets/shaders/reeds_sway.slang".into()),
///     fragment: "assets/shaders/reeds.slang".into(),
///     ..Default::default()
/// };
/// assert!(water.vertex.is_none() && reeds.vertex.is_some());
/// ```
///
/// # The two hooks
///
/// A Shader file defines a function, not an entry point. The engine owns every
/// entry point, binding and pipeline on every backend, and calls the world's
/// functions from inside its own:
///
/// ```hlsl
/// // the `fragment` file, required
/// float4 shade(VertexOut in, GpuObjectData od);
///
/// // the `vertex` file, optional; without one the engine projects the vertex itself
/// VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
///                     float3 color, float2 uv);
/// ```
///
/// `shade` returns the surface's linear-light colour with alpha. `od` is the
/// surface's material record whichever path drew it: `tint_roughness`,
/// `emissive_metallic`, `albedo_index`, `normal_index`, `emissive_map_index`,
/// `orm_map_index` and `bb_max_alpha_cutoff.w` are the fields a surface
/// reads. `transform` receives the model matrix and the model-space
/// attributes, after skinning for a [SkinnedMesh](#skinnedmesh) and per
/// instance for an [InstancedProp](#instancedprop), and returns the projected
/// vertex; the engine's own is `project_vertex`, so a displacement is
/// `return project_vertex(model, pos + offset, normal, tangent, color, uv);`.
///
/// Both files are compiled inside the engine's own main-pass source, so they
/// see the same vocabulary the engine's shading uses and declare no layout,
/// binding, register, attribute or varying of their own:
///
/// - `shade_surface(in, od)`: the engine's PBR lighting, so
///   `return shade_surface(in, od) * tint;` starts from it.
/// - `project_vertex(model, pos, normal, tangent, color, uv)`: the engine's
///   projection.
/// - `pool_sample(index, uv)`: a texture from the world's pool by the record's
///   index.
/// - `decode_normal_map(rg)`: a tangent-space normal from a normal-map texel.
/// - `shadow_factor_cascaded(world_pos, view_depth, screen_xy)`: the sun's
///   cascaded shadow term.
/// - `environment_specular(world_pos, reflected, lod)`: the reflection
///   environment.
/// - `irradiance_sample(normal)`: the diffuse environment.
/// - `VIEW`: the view block, with `vp`, `view_mat`, `elapsed`, `cam_x` /
///   `cam_y` / `cam_z` and `sky_rot`.
/// - `LIGHTS`: the light block, with `dir[]`, `pt[]`, `num_dir`, `num_pt` and
///   `ambient_intensity`.
/// - `SKY_DIR(d)`: a world direction in the environment map's frame.
///
/// `VertexOut` is the engine's varying block: `position` (clip), `world_pos`,
/// `normal`, `tangent`, `bitangent`, `uv`, `view_depth` and `color`. A `shade`
/// must not read `in.object_id`; the record is `od`.
///
/// # More than one Shader
///
/// The first declared Shader is the world's default: everything renders with it
/// unless a [Material](#material) names another one through its `shader` field.
/// A world may declare up to 8 Shaders in total.
///
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
/// A Shader referenced only by materials belonging to one [Scene](#scene) is
/// owned by that scene: its pipeline is built when the scene loads (behind the
/// loading screen, alongside that scene's textures and meshes) and released when
/// the scene unloads. A Shader used across scenes, or by the world default,
/// loads at startup.
///
/// # Compilation
///
/// `cn build` compiles both files for the backend it cooks for and stores the
/// result in the world; a player needs no shader compiler. A file that fails
/// to compile, or omits its hook, fails the build naming the Shader and the
/// hook. Under `cn debug` a save to either file recompiles it and swaps the
/// live pipelines.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Shader {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the `.slang` file defining `shade`. Required.
    pub fragment: String,
    /// Path to the `.slang` file defining `transform`. Omit to keep the
    /// engine's own projection.
    #[serde(default)]
    pub vertex: Option<String>,
    /// Injected at load time from BlobAssetDef::payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Shader {
    /// The declared path for `stage`, if that file is present.
    pub fn stage(&self, stage: ShaderStage) -> Option<&str> {
        match stage {
            ShaderStage::Vertex => self.vertex.as_deref(),
            ShaderStage::Fragment => Some(&self.fragment),
        }
    }
}

/// The compiled payload a [`Shader`] carries in the blob: the authored files
/// and every program the cook compiled from them. Written by the cook, decoded
/// once by the renderer at load.
///
/// The sources ride along for the reason an `SdfVolume`'s field does: an
/// artifact is only loadable while the engine template it was built against
/// still matches, and the renderer proves that by reassembling and digesting.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ShaderPrograms {
    /// The Shader's asset name, for diagnostics.
    pub name: String,
    /// The `vertex` file's text, when the Shader declares one.
    pub vertex: Option<String>,
    /// The `fragment` file's text.
    pub fragment: String,
    /// Compiled entries, in the order the cook emitted them.
    pub programs: Vec<CompiledProgram>,
}

impl ShaderPrograms {
    /// Serialize the payload for the blob.
    pub fn encode(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Read a payload back out of the blob.
    pub fn decode(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    /// The artifact holding `entry`, if one was compiled from source matching
    /// `digest`. A mismatch is a stale artifact and reads as absent.
    pub fn artifact(&self, entry: &str, digest: u64) -> Option<&[u8]> {
        super::compiled_programs::artifact(&self.programs, entry, digest)
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn a_shader_parses_from_authored_args() {
        let s: Shader =
            serde_json::from_str(r#"{"fragment":"assets/shaders/water.slang"}"#).unwrap();
        assert_eq!(s.fragment, "assets/shaders/water.slang");
        assert!(s.vertex.is_none(), "the vertex file is optional");
        assert_eq!(s.stage(ShaderStage::Vertex), None);
        assert_eq!(
            s.stage(ShaderStage::Fragment),
            Some("assets/shaders/water.slang")
        );
        // The identity and payload locator are injected, never authored.
        assert_eq!(s.asset_id, AssetId::default());
        assert!(s.locator.is_none());

        let both: Shader =
            serde_json::from_str(r#"{"vertex":"v.slang","fragment":"f.slang"}"#).unwrap();
        assert_eq!(both.stage(ShaderStage::Vertex), Some("v.slang"));

        let bytes = postcard::to_allocvec(&both).unwrap();
        let back: Shader = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.vertex.as_deref(), Some("v.slang"));
        assert_eq!(back.fragment, "f.slang");
    }

    // The per-platform `sources` table is gone: a declaration still spelling it
    // is refused rather than read as a fragment-less Shader.
    #[test]
    fn the_old_per_platform_table_is_rejected() {
        let err = serde_json::from_str::<Shader>(
            r#"{"vertex":{"sources":{"metal":"a.metal"}},"fragment":{"source":"a.metal"}}"#,
        );
        assert!(err.is_err());
    }

    #[test]
    fn stages_parse_from_their_authored_spellings() {
        let stage = |s: &str| serde_json::from_str::<ShaderStage>(s).unwrap();
        assert_eq!(stage(r#""vertex""#), ShaderStage::Vertex);
        assert_eq!(stage(r#""fragment""#), ShaderStage::Fragment);
    }

    #[test]
    fn programs_round_trip_and_find_artifacts_by_entry_and_digest() {
        let payload = ShaderPrograms {
            name: "wall".to_string(),
            vertex: None,
            fragment: "float4 shade(VertexOut in, GpuObjectData od) { return 1.0; }".to_string(),
            programs: vec![CompiledProgram {
                entries: vec!["fragment_main".to_string()],
                source_digest: 3,
                artifact: vec![1, 2, 3],
            }],
        };
        let bytes = payload.encode().expect("encode");
        let decoded = ShaderPrograms::decode(&bytes).expect("decode");
        assert_eq!(decoded, payload);
        assert_eq!(decoded.artifact("fragment_main", 3), Some(&[1u8, 2, 3][..]));
        assert_eq!(decoded.artifact("fragment_main", 4), None, "stale");
        assert_eq!(decoded.artifact("vertex_main", 3), None);
    }

    #[test]
    fn an_empty_payload_holds_no_programs() {
        let payload = ShaderPrograms::default();
        assert!(payload.programs.is_empty());
        assert_eq!(
            ShaderPrograms::decode(&payload.encode().unwrap()),
            Ok(payload)
        );
    }

    #[test]
    fn decoding_garbage_is_an_error_not_a_panic() {
        assert!(ShaderPrograms::decode(&[0xff, 0xff, 0xff]).is_err());
    }

    // Shader keeps a hand-written Component impl rather than the generated
    // one, so its identity and payload injection are its own code.
    #[test]
    fn a_shader_takes_its_identity_and_payload_on_load() {
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
