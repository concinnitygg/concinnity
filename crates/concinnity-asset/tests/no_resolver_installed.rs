// What a named reference does when no resolver seam is installed.
//
// The seams are process-global and install-once, so this has to be its own test
// binary: any unit test that installs a stand-in would make the unset case
// unreachable. A real build always installs the resolvers before deserializing,
// so the behaviour pinned here is what an out-of-engine tool reading authoring
// JSON sees.

use concinnity_asset::{
    AssetId, AssetRef, AudioClipHandle, FontHandle, MaterialHandle, MeshHandle, ShaderHandle,
    SkinnedMeshHandle, TextureHandle, de_audio_clip_handle_vec, de_opt_asset_ref,
    de_opt_asset_ref_typed, de_opt_audio_clip_handle, de_opt_font_handle, de_opt_material_handle,
    de_opt_mesh_handle, de_opt_shader_handle, de_opt_skinned_mesh_handle, de_opt_texture_handle,
    de_texture_handle,
};

struct Texture;

#[derive(Debug, serde::Deserialize)]
struct Bare {
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    r: Option<AssetId>,
}

#[derive(serde::Deserialize)]
struct Typed {
    #[serde(default, deserialize_with = "de_opt_asset_ref_typed")]
    r: Option<AssetRef<Texture>>,
}

#[derive(serde::Deserialize)]
struct Handles {
    #[serde(default, deserialize_with = "de_opt_texture_handle")]
    tex: Option<TextureHandle>,
    #[serde(default, deserialize_with = "de_opt_mesh_handle")]
    mesh: Option<MeshHandle>,
    #[serde(default, deserialize_with = "de_opt_material_handle")]
    material: Option<MaterialHandle>,
    #[serde(default, deserialize_with = "de_opt_shader_handle")]
    shader: Option<ShaderHandle>,
    #[serde(default, deserialize_with = "de_opt_skinned_mesh_handle")]
    target: Option<SkinnedMeshHandle>,
    #[serde(default, deserialize_with = "de_opt_font_handle")]
    font: Option<FontHandle>,
    #[serde(default, deserialize_with = "de_opt_audio_clip_handle")]
    clip: Option<AudioClipHandle>,
    #[serde(default, deserialize_with = "de_audio_clip_handle_vec")]
    sounds: Vec<AudioClipHandle>,
}

#[derive(Debug, serde::Deserialize)]
struct Stage {
    #[serde(deserialize_with = "de_texture_handle")]
    texture: TextureHandle,
}

fn handles_error(json: &str) -> String {
    serde_json::from_str::<Handles>(json)
        .err()
        .expect("a name with no resolver is an error")
        .to_string()
}

#[test]
fn an_asset_id_name_is_a_deserialization_error() {
    let err = serde_json::from_str::<AssetId>("\"floor\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no asset-name resolver installed"), "{err}");
    let err = serde_json::from_str::<Bare>("{\"r\":\"floor\"}")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no asset-name resolver installed"), "{err}");
}

#[test]
fn every_handle_kind_names_itself_in_the_error() {
    // The kind is in the message because each has its own seam: knowing which
    // one is missing is the whole diagnostic.
    for (json, kind) in [
        (r#"{"tex":"floor"}"#, "texture"),
        (r#"{"mesh":"wall"}"#, "mesh"),
        (r#"{"material":"stone"}"#, "material"),
        (r#"{"shader":"water"}"#, "shader"),
        (r#"{"target":"hero"}"#, "skinned-mesh"),
        (r#"{"font":"body"}"#, "font"),
        (r#"{"clip":"theme"}"#, "audio-clip"),
        (r#"{"sounds":["door"]}"#, "audio-clip"),
    ] {
        let err = handles_error(json);
        let expected = format!("no {kind}-handle resolver installed");
        assert!(err.contains(&expected), "{json}: {err}");
    }

    let err = serde_json::from_str::<Stage>(r#"{"texture":"bg"}"#)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("no texture-handle resolver installed"),
        "{err}"
    );
}

#[test]
fn a_typed_reference_keeps_the_name_instead_of_failing() {
    // Unlike a bare AssetId, an AssetRef has somewhere to put an unresolved
    // name, so an authoring tool can read and rewrite a world it cannot resolve.
    let r: AssetRef<Texture> = serde_json::from_str("\"floor\"").unwrap();
    assert_eq!(r.name(), "floor");
    assert_eq!(r.id(), None);
    assert!(!r.is_resolved());

    let holder: Typed = serde_json::from_str("{\"r\":\"wall\"}").unwrap();
    let r = holder.r.expect("a name is still a reference");
    assert_eq!(r.name(), "wall");
    assert!(!r.is_resolved());
}

#[test]
fn already_resolved_integers_still_parse() {
    // The compiled-args and baked forms never consult a resolver, so they read
    // the same with the seam unset.
    assert_eq!(serde_json::from_str::<AssetId>("5").unwrap(), AssetId(5));
    assert_eq!(
        serde_json::from_str::<Bare>("{\"r\":5}").unwrap().r,
        Some(AssetId(5))
    );
    let h: Handles = serde_json::from_str(
        r#"{"tex":3,"mesh":4,"material":5,"shader":6,"target":7,"font":8,"clip":9,
            "sounds":[1,2]}"#,
    )
    .unwrap();
    assert_eq!(h.tex, Some(TextureHandle(3)));
    assert_eq!(h.mesh, Some(MeshHandle(4)));
    assert_eq!(h.material, Some(MaterialHandle(5)));
    assert_eq!(h.shader, Some(ShaderHandle(6)));
    assert_eq!(h.target, Some(SkinnedMeshHandle(7)));
    assert_eq!(h.font, Some(FontHandle(8)));
    assert_eq!(h.clip, Some(AudioClipHandle(9)));
    assert_eq!(h.sounds, vec![AudioClipHandle(1), AudioClipHandle(2)]);
    assert_eq!(
        serde_json::from_str::<Stage>(r#"{"texture":2}"#)
            .unwrap()
            .texture,
        TextureHandle(2)
    );

    let bytes = postcard::to_allocvec(&MaterialHandle(9)).unwrap();
    assert_eq!(
        postcard::from_bytes::<MaterialHandle>(&bytes).unwrap(),
        MaterialHandle(9)
    );
}
