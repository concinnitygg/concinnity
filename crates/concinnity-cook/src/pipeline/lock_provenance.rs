//! Lock-file provenance for the resource stream: one `LockedResource` per
//! compiled resource.

use concinnity_core::ecs::ResourceRecord;
use concinnity_host::thread::asset_id;

use super::partition::ResourceJob;
use super::result::{MeshSourceInfo, TextureSourceInfo};
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use crate::blob::{LockedMeshSource, LockedResource, LockedTextureSource};

// `resources` is emitted in `resource_jobs` order, so the two zip
// index-aligned. Texture and Mesh records also carry their hot-reload source
// so a blob boot can rebuild those catalogs without the authored args.
pub(super) fn lock_provenance(
    assets: &[WorldJsonlAsset],
    resource_jobs: &[ResourceJob],
    resources: &[ResourceRecord],
    texture_sources: &[TextureSourceInfo],
    mesh_sources: &[MeshSourceInfo],
) -> Vec<LockedResource> {
    resource_jobs
        .iter()
        .zip(resources)
        .map(|((asset_idx, rt, handle), record)| {
            let asset = &assets[*asset_idx];
            LockedResource {
                name: asset.id.clone(),
                // Already interned by the build, so this looks up its id.
                id: Some(asset_id::intern(&asset.id).0),
                kind: rt.as_str().to_string(),
                handle: *handle,
                args_hash: crate::blob::checksum(asset.args.to_string().as_bytes()),
                payload_blob: record.payload.as_ref().map(|p| p.blob_index),
                texture_source: (*rt == RegisteredType::Texture).then(|| {
                    let t = &texture_sources[*handle as usize];
                    LockedTextureSource {
                        source: t.source.clone(),
                        image_index: t.image_index,
                    }
                }),
                mesh_source: (*rt == RegisteredType::Mesh).then(|| {
                    let m = &mesh_sources[*handle as usize];
                    LockedMeshSource {
                        source: m.source.clone(),
                        primitive_index: m.primitive_index,
                        lod_levels: m.lod_levels,
                        lod_distances: m.lod_distances.clone(),
                    }
                }),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::fixtures::wja;
    use concinnity_core::blob::ResourceKind;
    use concinnity_core::ecs::PayloadLocator;

    fn record(kind: ResourceKind, handle: u32, blob_index: Option<u32>) -> ResourceRecord {
        ResourceRecord {
            resource_kind: kind,
            handle,
            payload: blob_index.map(|blob_index| PayloadLocator {
                blob_index,
                offset: 0,
                len: 1,
            }),
            data_bytes: Vec::new(),
        }
    }

    #[test]
    fn sources_ride_only_their_own_kind_and_payload_blob_follows_the_record() {
        asset_id::reset_interner();
        let assets = vec![
            wja(
                "tex",
                RegisteredType::Texture,
                serde_json::json!({"source": "t.png"}),
            ),
            wja(
                "mesh",
                RegisteredType::Mesh,
                serde_json::json!({"source": "m.glb"}),
            ),
            wja(
                "clip",
                RegisteredType::AudioClip,
                serde_json::json!({"source": "c.ogg"}),
            ),
        ];
        let names: Vec<&str> = assets.iter().map(|a| a.id.as_str()).collect();
        asset_id::intern_all(&names);
        let jobs = [
            (0, RegisteredType::Texture, 0),
            (1, RegisteredType::Mesh, 0),
            (2, RegisteredType::AudioClip, 0),
        ];
        let resources = [
            record(ResourceKind::Texture, 0, Some(1)),
            record(ResourceKind::Mesh, 0, Some(2)),
            record(ResourceKind::AudioClip, 0, None),
        ];
        let textures = [TextureSourceInfo {
            name_id: 0,
            source: "t.png".to_string(),
            image_index: 5,
        }];
        let meshes = [MeshSourceInfo {
            source: "m.glb".to_string(),
            primitive_index: 1,
            lod_levels: 2,
            lod_distances: vec![8.0],
        }];

        let locks = lock_provenance(&assets, &jobs, &resources, &textures, &meshes);
        assert_eq!(locks.len(), 3);

        let tex = &locks[0];
        assert_eq!(
            (tex.name.as_str(), tex.id, tex.kind.as_str()),
            ("tex", Some(0), "Texture")
        );
        assert_eq!(tex.payload_blob, Some(1));
        let ts = tex
            .texture_source
            .as_ref()
            .expect("texture carries its source");
        assert_eq!((ts.source.as_str(), ts.image_index), ("t.png", 5));
        assert!(tex.mesh_source.is_none());

        let mesh = &locks[1];
        assert_eq!(mesh.payload_blob, Some(2));
        assert!(mesh.texture_source.is_none());
        let ms = mesh.mesh_source.as_ref().expect("mesh carries its source");
        assert_eq!(ms.source, "m.glb");
        assert_eq!((ms.primitive_index, ms.lod_levels), (1, 2));
        assert_eq!(ms.lod_distances, vec![8.0]);

        let clip = &locks[2];
        assert_eq!((clip.id, clip.handle), (Some(2), 0));
        assert_eq!(clip.payload_blob, None);
        assert!(clip.texture_source.is_none() && clip.mesh_source.is_none());
        assert_eq!(
            clip.args_hash,
            crate::blob::checksum(assets[2].args.to_string().as_bytes())
        );
    }
}
