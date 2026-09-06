// Surface-material schema.

use crate::ecs::ShaderHandle;
use crate::ecs::TextureHandle;
use crate::ecs::asset_id::AssetId;
use crate::ecs::de_opt_shader_handle;
use crate::ecs::de_opt_texture_handle;

/// A Material bundles the surface parameters that control how a [Prop](#prop) is
/// lit and shaded.
///
/// Reference it from a [Prop](#prop)'s `material` field. The `material` field takes
/// precedence over the older `texture` field.
///
/// ```rust
/// # use concinnity_core::components::Material;
/// Material {
///     roughness: 0.85,
///     metallic: 0.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Material {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [Texture](#texture) asset used as the base colour (albedo) map.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub albedo: Option<TextureHandle>,
    /// The [Texture](#texture) asset used as a tangent-space normal map.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub normal_map: Option<TextureHandle>,
    /// The [Texture](#texture) asset used as an emissive map. Multiplied by
    /// `emissive_factor` to drive the glow; when omitted, only the scalar
    /// `emissive_factor` is used. Pair a textured emissive with an
    /// `emissive_factor` above 1 to make the bright parts bloom.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub emissive_map: Option<TextureHandle>,
    /// The [Texture](#texture) asset used as a packed surface map: green =
    /// roughness, blue = metalness. When present it overrides the scalar
    /// `roughness` and `metallic` per-texel; when omitted those scalars are
    /// used. The red channel is reserved and not read as ambient occlusion:
    /// packed maps in the wild (glTF metallic-roughness, FBX specular maps)
    /// leave red empty, so treating it as occlusion would darken indirect
    /// light to black. Ambient occlusion comes from the screen-space pass.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub orm_map: Option<TextureHandle>,
    /// Perceptual roughness in [0, 1]. 0 = mirror, 1 = fully diffuse.
    /// Controls the width of the specular highlight.
    pub roughness: f32,
    /// Metallic factor in [0, 1]. 0 = dielectric (plastic/stone), 1 = metal.
    /// Metallic surfaces tint their reflections with the albedo colour and show
    /// almost no diffuse; dielectrics keep a neutral, dim reflection.
    pub metallic: f32,
    /// Linear-space RGB multiplier applied to the albedo sample. Useful for
    /// tinting a shared texture without a separate asset (e.g. coloured brick).
    pub tint: [f32; 3],
    /// Additive emission colour in linear space. Non-zero values make the
    /// surface appear to glow independently of the scene lighting.
    pub emissive_factor: [f32; 3],
    /// Alpha-cutout threshold in [0, 1]. When non-zero, a texel whose `albedo`
    /// alpha falls below it is discarded outright, punching a hole in the
    /// surface: this is how foliage, chain-link, and decal cards are drawn as
    /// one opaque quad. 0 (the default) disables the test and keeps every texel.
    /// Cutout is not glass: the surface still renders in the opaque pass, so
    /// leave `transparent` and `see_through` off.
    pub alpha_cutoff: f32,
    /// Surface opacity in [0, 1]. 1 = fully opaque (the default). Only
    /// meaningful when `transparent` is set: it drives how much of the scene
    /// behind the surface shows through the glass.
    pub opacity: f32,
    /// When true, the surface is a translucent dielectric (glass): it renders
    /// in the engine's transparent pass instead of the opaque pass, refracting
    /// and reflecting the scene rather than writing solid colour + depth. The
    /// importer sets this for materials it detects as glass; authored materials
    /// can opt in directly. Defaults to false (opaque).
    pub transparent: bool,
    /// When true, the glass is rendered as genuinely see-through: the scene
    /// behind it shows through with a sharp per-pixel reflection (requires a
    /// ray-tracing-capable GPU). When false (the default), a `transparent`
    /// surface still renders as low-roughness reflective glass that hides
    /// whatever is behind it. See-through only looks right when the space behind
    /// the glass is actually modelled, so it is opt-in per material. Setting it
    /// implies `transparent`.
    pub see_through: bool,
    /// The [Shader](#shader) asset that shades surfaces using this material.
    /// When omitted, the world's default shader is used. Referencing a shader
    /// from a material ties that shader's lifetime to the material's: a shader
    /// referenced only by scene-exclusive materials loads and unloads with the
    /// scene.
    #[serde(deserialize_with = "de_opt_shader_handle")]
    pub shader: Option<ShaderHandle>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            albedo: None,
            normal_map: None,
            emissive_map: None,
            orm_map: None,
            roughness: 0.8,
            metallic: 0.0,
            tint: [1.0, 1.0, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            alpha_cutoff: 0.0,
            opacity: 1.0,
            transparent: false,
            see_through: false,
            shader: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_material_is_an_opaque_untextured_dielectric() {
        let m = Material::default();
        assert_eq!(m.roughness, 0.8);
        assert_eq!(m.metallic, 0.0);
        assert_eq!(m.tint, [1.0, 1.0, 1.0]);
        assert_eq!(m.emissive_factor, [0.0, 0.0, 0.0]);
        assert_eq!(m.opacity, 1.0);
        assert!(!m.transparent);
        assert!(!m.see_through);
        // Zero alpha cutoff means "no cutout", not "discard everything".
        assert_eq!(m.alpha_cutoff, 0.0);
        for map in [&m.albedo, &m.normal_map, &m.emissive_map, &m.orm_map] {
            assert!(map.is_none());
        }
        // No shader means the engine's own main-pass program draws it.
        assert!(m.shader.is_none());
    }

    #[test]
    fn every_texture_slot_resolves_through_its_own_reference() {
        crate::test_support::install_resolvers();
        let m: Material = serde_json::from_str(
            r#"{"albedo":"tex_a","normal_map":"tex_nm","emissive_map":"tex_em",
                "orm_map":"tex_orm","shader":"water_shader"}"#,
        )
        .unwrap();
        assert_eq!(m.albedo, Some(TextureHandle(5)));
        assert_eq!(m.normal_map, Some(TextureHandle(6)));
        assert_eq!(m.emissive_map, Some(TextureHandle(6)));
        assert_eq!(m.orm_map, Some(TextureHandle(7)));
        assert_eq!(m.shader, Some(ShaderHandle(12)));
    }

    #[test]
    fn a_glass_material_round_trips_through_postcard() {
        let m: Material = serde_json::from_str(
            r#"{"roughness":0.05,"metallic":1,"tint":[0.8,0.9,1],"emissive_factor":[2,2,2],
                "alpha_cutoff":0.5,"opacity":0.3,"transparent":true,"see_through":true}"#,
        )
        .unwrap();
        let bytes = postcard::to_allocvec(&m).unwrap();
        let back: Material = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.roughness, 0.05);
        assert_eq!(back.metallic, 1.0);
        assert_eq!(back.tint, [0.8, 0.9, 1.0]);
        assert_eq!(back.emissive_factor, [2.0, 2.0, 2.0]);
        assert_eq!(back.alpha_cutoff, 0.5);
        assert_eq!(back.opacity, 0.3);
        assert!(back.transparent);
        assert!(back.see_through);
        assert_eq!(back.asset_id, AssetId::default());
    }
}
