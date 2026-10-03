//! The Shader asset: the authored schema (Shader, ShaderStage, and the
//! ShaderPrograms container the cook fills). The compile lives in
//! concinnity-cook (`compile::shader`); which programs a world shader compiles
//! to is `render::shader_programs::surface`.

use crate::ecs::PayloadLocator;
use alloc::string::String;
use alloc::vec::Vec;

use super::compiled_programs::CompiledProgram;
use crate::render::shader_programs::surface::Sources;
use crate::render::shader_source::SourceFile;

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
/// with functions of your own. Written in HLSL, one source for every
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
///     fragment: "assets/shaders/water.hlsl".into(),
///     ..Default::default()
/// };
/// // Both hooks: a sway displacement, then the surface.
/// let reeds = Shader {
///     vertex: Some("assets/shaders/reeds_sway.hlsl".into()),
///     fragment: "assets/shaders/reeds.hlsl".into(),
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
/// float4 shade(VertexOut v, GpuObjectData od);
///
/// // the `vertex` file, optional; without one the engine projects the vertex itself
/// VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
///                     float3 color, float2 uv);
/// ```
///
/// `shade` returns the surface's linear-light color with alpha. `od` is the
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
/// - `shade_surface(v, od)`: the engine's PBR lighting, so
///   `return shade_surface(v, od) * tint;` starts from it.
/// - `project_vertex(model, pos, normal, tangent, color, uv)`: the engine's
///   projection.
/// - `material_param(index)`: parameter `index`, 0 to 7, of the `params` the
///   surface's [Material](#material) sets, from either hook; 0 for a surface
///   drawn without a material.
/// - `pool_sample(index, uv)`: a texture from the world's pool by the record's
///   index.
/// - `decode_normal_map(rg)`: a tangent-space normal from a normal-map texel.
/// - `shadow_factor_cascaded(world_pos, view_depth, screen_xy)`: the sun's
///   cascaded shadow term.
/// - `environment_specular(probe_mask_all(), world_pos, reflected, roughness,
///   radiance)`: the reflection environment for a surface of `roughness` into
///   `radiance`, false where the world has neither a reflection probe nor an
///   environment map.
/// - `irradiance_sample(normal)`: the diffuse environment.
/// - `VIEW`: the view block, with `vp`, `view_mat`, `elapsed`, `cam_x` /
///   `cam_y` / `cam_z` and `sky_rot`.
/// - `LIGHTS`: the light block, with `dir[]`, `pt[]`, `num_dir`, `num_pt` and
///   `ambient_intensity`.
/// - `SKY_DIR(d)`: a world direction in the environment map's frame.
///
/// `VertexOut` is the engine's varying block: `position` (clip), `world_pos`,
/// `normal`, `tangent`, `bitangent`, `uv`, `view_depth` and `color`. A `shade`
/// must not read `v.object_id`; the record is `od`.
///
/// # Parameters from the Material
///
/// A Shader declares no inputs of its own. Each [Material](#material) instead
/// carries eight numbers, its `params`, which `material_param(0)` through
/// `material_param(7)` read for the surface being drawn, so one Shader can be
/// set up differently by every material that uses it. What each parameter
/// means is the Shader's to decide, and worth a comment at the top of its
/// file:
///
/// ```hlsl
/// // material_param(0): glow strength, material_param(1): pulses per second
/// float4 shade(VertexOut v, GpuObjectData od)
/// {
///     float pulse = 0.5 + 0.5 * sin(VIEW.elapsed * 6.2831 * material_param(1));
///     return shade_surface(v, od) + float4(od.tint_roughness.rgb * material_param(0) * pulse, 0.0);
/// }
/// ```
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
/// hook, with each compiler error reported at its file and line; a compiler
/// warning is logged the same way and the build goes on.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
pub struct Shader {
    /// Path to the `.hlsl` file defining `shade`. Required.
    #[asset(owned_file)]
    pub fragment: String,
    /// Path to the `.hlsl` file defining `transform`. Omit to keep the
    /// engine's own projection.
    #[serde(default)]
    #[asset(owned_file)]
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
#[derive(
    Debug,
    Clone,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
)]
pub struct ShaderPrograms {
    /// The Shader's asset name, for diagnostics.
    pub name: String,
    /// The `vertex` file, when the Shader declares one.
    pub vertex: Option<ShaderSource>,
    /// The `fragment` file.
    pub fragment: ShaderSource,
    /// Compiled entries, in the order the cook emitted them.
    pub programs: Vec<CompiledProgram>,
}

/// An authored file as it was compiled (one of a Shader's files, or an
/// `SdfVolume`'s field): the path it was compiled under and its text.
#[derive(
    Debug,
    Clone,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
)]
pub struct ShaderSource {
    /// The path compiler diagnostics name the file by.
    pub path: String,
    /// The file's text.
    pub text: String,
}

impl ShaderSource {
    /// The file, borrowed.
    pub fn as_file(&self) -> SourceFile<'_> {
        SourceFile {
            path: &self.path,
            text: &self.text,
        }
    }
}

impl ShaderPrograms {
    /// The files, as the assembly splices them.
    pub fn sources(&self) -> Sources<'_> {
        Sources {
            vertex: self.vertex.as_ref().map(ShaderSource::as_file),
            fragment: self.fragment.as_file(),
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::Component;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn a_stage_reads_its_declared_file() {
        let fragment_only = Shader {
            fragment: "f.hlsl".to_string(),
            ..Shader::default()
        };
        assert_eq!(fragment_only.stage(ShaderStage::Vertex), None);
        assert_eq!(fragment_only.stage(ShaderStage::Fragment), Some("f.hlsl"));

        let both = Shader {
            vertex: Some("v.hlsl".to_string()),
            ..fragment_only
        };
        assert_eq!(both.stage(ShaderStage::Vertex), Some("v.hlsl"));
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
    fn programs_find_artifacts_by_entry_and_digest() {
        let payload = ShaderPrograms {
            name: "wall".to_string(),
            vertex: None,
            fragment: ShaderSource::default(),
            programs: vec![CompiledProgram {
                entry: "fragment_main".to_string(),
                source_digest: 3,
                artifact: vec![1, 2, 3],
            }],
        };
        assert_eq!(payload.artifact("fragment_main", 3), Some(&[1u8, 2, 3][..]));
        assert_eq!(payload.artifact("fragment_main", 4), None, "stale");
        assert_eq!(payload.artifact("vertex_main", 3), None);
    }

    #[test]
    fn decoding_garbage_is_an_error_not_a_panic() {
        assert!(ShaderPrograms::decode(&[0xff, 0xff, 0xff]).is_err());
    }

    // The generated impl of a `compiled` type stores its payload locator.
    #[test]
    fn a_shader_takes_its_payload_on_load() {
        let bytes = postcard::to_allocvec(&Shader::default()).expect("a shader encodes");
        let mut shader = <Shader as Component>::from_baked(&bytes).expect("it loads back");
        assert_eq!(Shader::NAME, "Shader");

        let locator = PayloadLocator {
            blob_index: 1,
            offset: 8,
            len: 16,
        };
        shader.inject_locator(locator.clone());
        assert_eq!(shader.locator, Some(locator));
    }
}
