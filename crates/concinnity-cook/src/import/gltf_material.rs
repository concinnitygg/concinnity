// glTF material -> Concinnity Material args. Split out of the scene expansion
// so the mapping is exercised on its own; see the module header of `super` for
// the field-by-field correspondence.

use serde_json::{Map, Value};

// glTF's default `alphaCutoff` when `alphaMode` is MASK and the field is absent.
const DEFAULT_ALPHA_CUTOFF: f32 = 0.5;

// One glTF material mapped onto Concinnity's `Material` args, plus the UV sets
// its consumed texture references asked for. The mesh readers only fill UV set
// 0, so any other set names UVs that were never imported.
pub(super) struct MappedMaterial {
    pub args: Map<String, Value>,
    pub extra_uv_sets: Vec<u32>,
}

// Bind one texture reference to a Material field, naming the Texture entry the
// expansion emits per glTF image, and record a UV set the meshes don't carry.
fn bind(
    mapped: &mut MappedMaterial,
    field: &str,
    prefix: &str,
    tex_coord: u32,
    image_index: usize,
) {
    if tex_coord != 0 && !mapped.extra_uv_sets.contains(&tex_coord) {
        mapped.extra_uv_sets.push(tex_coord);
    }
    mapped.args.insert(
        field.to_string(),
        Value::String(format!("{prefix}_tex_{image_index}")),
    );
}

pub(super) fn map_material(prefix: &str, mat: &gltf::Material) -> MappedMaterial {
    let pbr = mat.pbr_metallic_roughness();
    let mut mapped = MappedMaterial {
        args: Map::new(),
        extra_uv_sets: Vec::new(),
    };

    if let Some(info) = pbr.base_color_texture() {
        let image = info.texture().source().index();
        bind(&mut mapped, "albedo", prefix, info.tex_coord(), image);
    }
    if let Some(info) = mat.normal_texture() {
        let image = info.texture().source().index();
        bind(&mut mapped, "normal_map", prefix, info.tex_coord(), image);
    }
    // glTF packs metallic-roughness as G = roughness, B = metalness, exactly the
    // channels `Material::orm_map` reads.
    if let Some(info) = pbr.metallic_roughness_texture() {
        let image = info.texture().source().index();
        bind(&mut mapped, "orm_map", prefix, info.tex_coord(), image);
    }
    if let Some(info) = mat.emissive_texture() {
        let image = info.texture().source().index();
        bind(&mut mapped, "emissive_map", prefix, info.tex_coord(), image);
    }

    let base_color = pbr.base_color_factor();
    mapped.args.insert(
        "tint".to_string(),
        serde_json::json!([base_color[0], base_color[1], base_color[2]]),
    );
    mapped.args.insert(
        "metallic".to_string(),
        serde_json::json!(pbr.metallic_factor()),
    );
    mapped.args.insert(
        "roughness".to_string(),
        serde_json::json!(pbr.roughness_factor()),
    );
    let e = mat.emissive_factor();
    mapped.args.insert(
        "emissive_factor".to_string(),
        serde_json::json!([e[0], e[1], e[2]]),
    );

    // MASK is the cutout mode the opaque pass can honour with a discard. BLEND
    // stays unmapped: Concinnity's `transparent` means refracting glass, which
    // is not what a blended leaf card or decal wants.
    if mat.alpha_mode() == gltf::material::AlphaMode::Mask {
        mapped.args.insert(
            "alpha_cutoff".to_string(),
            serde_json::json!(mat.alpha_cutoff().unwrap_or(DEFAULT_ALPHA_CUTOFF)),
        );
    }

    mapped
}

#[cfg(test)]
mod tests {
    use super::*;

    // Four images so every consumed slot can bind a distinct one and the
    // assertions can tell them apart.
    fn material_doc(materials: &str) -> gltf::Gltf {
        let json = format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "materials": [{materials}],
  "textures": [{{"source": 0}}, {{"source": 1}}, {{"source": 2}}, {{"source": 3}}],
  "images": [{{"uri": "a.png"}}, {{"uri": "b.png"}}, {{"uri": "c.png"}}, {{"uri": "d.png"}}]
}}"#
        );
        gltf::Gltf::from_slice(json.as_bytes()).expect("parse gltf")
    }

    fn map_first(materials: &str) -> MappedMaterial {
        let doc = material_doc(materials);
        let mat = doc.document.materials().next().expect("one material");
        map_material("scn", &mat)
    }

    #[test]
    fn packed_metallic_roughness_and_emissive_textures_bind_to_their_material_fields() {
        let mapped = map_first(
            r#"{
              "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicRoughnessTexture": {"index": 1}
              },
              "normalTexture": {"index": 2},
              "emissiveTexture": {"index": 3},
              "emissiveFactor": [1.0, 1.0, 1.0]
            }"#,
        );
        assert_eq!(mapped.args["albedo"], "scn_tex_0");
        assert_eq!(mapped.args["orm_map"], "scn_tex_1");
        assert_eq!(mapped.args["normal_map"], "scn_tex_2");
        assert_eq!(mapped.args["emissive_map"], "scn_tex_3");
        assert_eq!(
            mapped.args["emissive_factor"],
            serde_json::json!([1.0, 1.0, 1.0])
        );
        assert!(mapped.extra_uv_sets.is_empty());
    }

    #[test]
    fn an_occlusion_texture_is_dropped_rather_than_packed_into_the_orm_red_channel() {
        let mapped = map_first(
            r#"{
              "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 1}},
              "occlusionTexture": {"index": 2}
            }"#,
        );
        // The ORM map is the metallic-roughness image; occlusion never binds,
        // because ambient occlusion comes from the screen-space pass.
        assert_eq!(mapped.args["orm_map"], "scn_tex_1");
        assert!(!mapped.args.contains_key("occlusion_map"));
        assert!(
            mapped
                .args
                .values()
                .all(|v| v.as_str() != Some("scn_tex_2")),
            "the occlusion image must not be bound to any field: {:?}",
            mapped.args
        );
    }

    #[test]
    fn occlusion_sharing_the_metallic_roughness_image_still_binds_only_that_image_once() {
        let mapped = map_first(
            r#"{
              "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 1}},
              "occlusionTexture": {"index": 1}
            }"#,
        );
        assert_eq!(mapped.args["orm_map"], "scn_tex_1");
        assert_eq!(
            mapped
                .args
                .values()
                .filter(|v| v.as_str() == Some("scn_tex_1"))
                .count(),
            1
        );
    }

    #[test]
    fn alpha_mode_mask_carries_its_cutoff_and_falls_back_to_the_gltf_default() {
        let explicit = map_first(r#"{"alphaMode": "MASK", "alphaCutoff": 0.25}"#);
        assert_eq!(explicit.args["alpha_cutoff"], serde_json::json!(0.25));

        let defaulted = map_first(r#"{"alphaMode": "MASK"}"#);
        assert_eq!(defaulted.args["alpha_cutoff"], serde_json::json!(0.5));
    }

    #[test]
    fn opaque_and_blend_materials_leave_the_cutoff_unset_so_it_stays_disabled() {
        // No `alpha_cutoff` arg means the Material default of 0.0 applies, which
        // disables the test. BLEND is deliberately not mapped onto `transparent`:
        // that field means refracting glass.
        for materials in [
            r#"{"alphaMode": "OPAQUE"}"#,
            r#"{"alphaMode": "BLEND"}"#,
            r#"{}"#,
        ] {
            let mapped = map_first(materials);
            assert!(
                !mapped.args.contains_key("alpha_cutoff"),
                "{materials} must not set a cutoff"
            );
            assert!(!mapped.args.contains_key("transparent"));
            assert!(!mapped.args.contains_key("see_through"));
        }
    }

    #[test]
    fn texture_references_on_a_uv_set_the_meshes_never_carry_are_reported_once_each() {
        let mapped = map_first(
            r#"{
              "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0, "texCoord": 1},
                "metallicRoughnessTexture": {"index": 1, "texCoord": 1}
              },
              "emissiveTexture": {"index": 2, "texCoord": 2},
              "normalTexture": {"index": 3}
            }"#,
        );
        assert_eq!(mapped.extra_uv_sets, vec![1, 2]);
        // The binding still happens; the warning is the only consequence.
        assert_eq!(mapped.args["albedo"], "scn_tex_0");
    }

    #[test]
    fn scalar_factors_map_onto_the_pbr_subset() {
        let mapped = map_first(
            r#"{
              "pbrMetallicRoughness": {
                "baseColorFactor": [0.5, 0.25, 0.125, 1.0],
                "metallicFactor": 0.75,
                "roughnessFactor": 0.375
              },
              "emissiveFactor": [0.25, 0.5, 0.75]
            }"#,
        );
        assert_eq!(mapped.args["tint"], serde_json::json!([0.5, 0.25, 0.125]));
        assert_eq!(mapped.args["metallic"], serde_json::json!(0.75));
        assert_eq!(mapped.args["roughness"], serde_json::json!(0.375));
        assert_eq!(
            mapped.args["emissive_factor"],
            serde_json::json!([0.25, 0.5, 0.75])
        );
    }
}
