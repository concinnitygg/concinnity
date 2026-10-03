//! What a named reference does when no resolver seam is installed.
//!
//! The seams are process-global and install-once, so this has to be its own test
//! binary: any unit test that installs a stand-in would make the unset case
//! unreachable. A real build always installs the resolvers before deserializing,
//! so the behavior pinned here is what an out-of-engine tool reading authoring
//! JSON sees.

use concinnity_core::components::Texture;
use concinnity_core::ecs::Ref;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{
    AudioClipHandle, FontHandle, MaterialHandle, MeshHandle, ShaderHandle, SkinnedMeshHandle,
    TextureHandle,
};

#[derive(Debug, serde::Deserialize)]
struct Optional {
    #[serde(default)]
    r: Option<Ref<Texture>>,
}

#[derive(serde::Deserialize)]
struct Handles {
    #[serde(default)]
    tex: Option<TextureHandle>,
    #[serde(default)]
    mesh: Option<MeshHandle>,
    #[serde(default)]
    material: Option<MaterialHandle>,
    #[serde(default)]
    shader: Option<ShaderHandle>,
    #[serde(default)]
    target: Option<SkinnedMeshHandle>,
    #[serde(default)]
    font: Option<FontHandle>,
    #[serde(default)]
    clip: Option<AudioClipHandle>,
    #[serde(default)]
    sounds: Vec<AudioClipHandle>,
}

fn handles_error(json: &str) -> String {
    serde_json::from_str::<Handles>(json)
        .err()
        .expect("a name with no resolver is an error")
        .to_string()
}

#[test]
fn a_reference_name_is_a_deserialization_error() {
    let err = serde_json::from_str::<AssetId>("\"floor\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no asset-name resolver installed"), "{err}");
    let err = serde_json::from_str::<Ref<Texture>>("\"floor\"")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no asset-name resolver installed"), "{err}");
    let err = serde_json::from_str::<Optional>("{\"r\":\"floor\"}")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no asset-name resolver installed"), "{err}");
}

#[test]
fn every_handle_space_names_itself_in_the_error() {
    // The space is in the message because each has its own seam: knowing which
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
}

#[test]
fn already_resolved_integers_still_parse() {
    // The compiled-args and baked forms never consult a resolver, so they read
    // the same with the seam unset.
    assert_eq!(serde_json::from_str::<AssetId>("5").unwrap(), AssetId(5));
    assert_eq!(
        serde_json::from_str::<Optional>("{\"r\":5}").unwrap().r,
        Some(Ref::new(AssetId(5)))
    );
    let h: Handles = serde_json::from_str(
        r#"{"tex":3,"mesh":4,"material":5,"shader":6,"target":7,"font":8,"clip":9,
            "sounds":[1,2]}"#,
    )
    .unwrap();
    assert_eq!(h.tex, Some(TextureHandle::new(3)));
    assert_eq!(h.mesh, Some(MeshHandle::new(4)));
    assert_eq!(h.material, Some(MaterialHandle::new(5)));
    assert_eq!(h.shader, Some(ShaderHandle::new(6)));
    assert_eq!(h.target, Some(SkinnedMeshHandle::new(7)));
    assert_eq!(h.font, Some(FontHandle::new(8)));
    assert_eq!(h.clip, Some(AudioClipHandle::new(9)));
    assert_eq!(
        h.sounds,
        vec![AudioClipHandle::new(1), AudioClipHandle::new(2)]
    );

    let bytes = postcard::to_allocvec(&MaterialHandle::new(9)).unwrap();
    assert_eq!(
        postcard::from_bytes::<MaterialHandle>(&bytes).unwrap(),
        MaterialHandle::new(9)
    );
}
