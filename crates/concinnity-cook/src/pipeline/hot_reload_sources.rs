//! The file source behind each texture and mesh handle, so a hot-reload
//! watcher can map a saved file back to the handle it feeds.

use concinnity_host::thread::asset_id;

use super::partition::ResourceJob;
use super::result::{MeshSourceInfo, TextureSourceInfo};
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

// Both tables are indexed by handle. A procedural texture or an inline-authored
// mesh leaves an empty source, since there is nothing to watch.
pub(super) fn hot_reload_sources(
    assets: &[WorldJsonlAsset],
    resource_jobs: &[ResourceJob],
) -> (Vec<TextureSourceInfo>, Vec<MeshSourceInfo>) {
    (
        by_handle(
            assets,
            resource_jobs,
            RegisteredType::Texture,
            texture_source,
        ),
        by_handle(assets, resource_jobs, RegisteredType::Mesh, mesh_source),
    )
}

// One slot per handle of `rt`, up to the highest assigned, each filled from
// the asset that holds it.
fn by_handle<T: Clone + Default>(
    assets: &[WorldJsonlAsset],
    resource_jobs: &[ResourceJob],
    rt: RegisteredType,
    fill: fn(&WorldJsonlAsset) -> T,
) -> Vec<T> {
    let jobs = || resource_jobs.iter().filter(move |(_, t, _)| *t == rt);
    let len = jobs()
        .map(|(_, _, handle)| *handle as usize + 1)
        .max()
        .unwrap_or(0);
    let mut table = vec![T::default(); len];
    for (asset_idx, _, handle) in jobs() {
        table[*handle as usize] = fill(&assets[*asset_idx]);
    }
    table
}

fn str_arg(args: &serde_json::Value, key: &str) -> String {
    args.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn u32_arg(args: &serde_json::Value, key: &str, default: u32) -> u32 {
    args.get(key)
        .and_then(|v| v.as_u64())
        .unwrap_or(default as u64) as u32
}

fn texture_source(asset: &WorldJsonlAsset) -> TextureSourceInfo {
    let generated = !str_arg(&asset.args, "generator").is_empty();
    let (source, image_index) = if generated {
        (String::new(), 0)
    } else {
        (
            str_arg(&asset.args, "source"),
            u32_arg(&asset.args, "image_index", 0),
        )
    };
    TextureSourceInfo {
        name_id: asset_id::intern(&asset.name).0,
        source,
        image_index,
    }
}

fn mesh_source(asset: &WorldJsonlAsset) -> MeshSourceInfo {
    let args = &asset.args;
    MeshSourceInfo {
        source: str_arg(args, "source"),
        primitive_index: u32_arg(args, "primitive_index", 0),
        lod_levels: u32_arg(args, "lod_levels", 1),
        lod_distances: args
            .get("lod_distances")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|d| d.as_f64())
                    .map(|d| d as f32)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::fixtures::wja;

    #[test]
    fn a_generator_texture_has_nothing_to_watch() {
        asset_id::reset_interner();
        let assets = vec![wja(
            "noise",
            "Texture",
            serde_json::json!({"generator": "checker", "source": "ignored.png", "image_index": 3}),
        )];
        let (textures, meshes) = hot_reload_sources(&assets, &[(0, RegisteredType::Texture, 0)]);
        assert_eq!(
            textures,
            vec![TextureSourceInfo {
                name_id: asset_id::intern("noise").0,
                source: String::new(),
                image_index: 0,
            }]
        );
        assert!(meshes.is_empty());
    }

    #[test]
    fn a_file_texture_fills_the_slot_at_its_handle() {
        asset_id::reset_interner();
        let assets = vec![
            wja("clip", "AudioClip", serde_json::json!({"source": "a.ogg"})),
            wja(
                "atlas",
                "Texture",
                serde_json::json!({"source": "atlas.ktx2", "image_index": 2}),
            ),
        ];
        let jobs = [
            (0, RegisteredType::AudioClip, 0),
            (1, RegisteredType::Texture, 2),
        ];
        let (textures, _) = hot_reload_sources(&assets, &jobs);
        assert_eq!(textures.len(), 3);
        assert_eq!(textures[0], TextureSourceInfo::default());
        assert_eq!(textures[1], TextureSourceInfo::default());
        assert_eq!(textures[2].name_id, asset_id::intern("atlas").0);
        assert_eq!(textures[2].source, "atlas.ktx2");
        assert_eq!(textures[2].image_index, 2);
    }

    #[test]
    fn mesh_sources_default_to_one_lod_level() {
        asset_id::reset_interner();
        let assets = vec![
            wja("plain", "Mesh", serde_json::json!({"source": "a.glb"})),
            wja(
                "lod",
                "Mesh",
                serde_json::json!({
                    "source": "b.glb",
                    "primitive_index": 4,
                    "lod_levels": 3,
                    "lod_distances": [10.0, 20.0],
                }),
            ),
        ];
        let jobs = [(0, RegisteredType::Mesh, 0), (1, RegisteredType::Mesh, 1)];
        let (textures, meshes) = hot_reload_sources(&assets, &jobs);
        assert!(textures.is_empty());
        assert_eq!(
            meshes,
            vec![
                MeshSourceInfo {
                    source: "a.glb".to_string(),
                    primitive_index: 0,
                    lod_levels: 1,
                    lod_distances: Vec::new(),
                },
                MeshSourceInfo {
                    source: "b.glb".to_string(),
                    primitive_index: 4,
                    lod_levels: 3,
                    lod_distances: vec![10.0, 20.0],
                },
            ]
        );
    }
}
