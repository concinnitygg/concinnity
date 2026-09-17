//! Splits a world into component assets, each resolved to a `BlobAssetDef`,
//! and resource assets, each queued for the resource stream.

use concinnity_core::ecs::BlobAssetDef;
use concinnity_core::resource::ResourceHandles;
use concinnity_host::thread::asset_id;

use crate::asset_api::{self, AssetRequest};
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

// A resource asset queued for compile: its index in the world's asset list,
// its type, and its assigned per-kind handle.
pub(super) type ResourceJob = (usize, RegisteredType, u32);

pub(super) struct Partitioned {
    // Component defs in declaration order, keyed by asset name.
    pub(super) named: Vec<(String, BlobAssetDef)>,
    // The source asset index of each entry in `named`, which is not 1:1 with
    // the asset list once resources are split out.
    pub(super) named_src: Vec<usize>,
    pub(super) resource_jobs: Vec<ResourceJob>,
}

// A resource asset has left the component registry, so it never goes through
// `create_asset_def`; it takes the handle assigned in `handles` instead.
pub(super) fn partition_components(
    assets: &[WorldJsonlAsset],
    handles: &ResourceHandles,
) -> std::io::Result<Partitioned> {
    let mut out = Partitioned {
        named: Vec::new(),
        named_src: Vec::new(),
        resource_jobs: Vec::new(),
    };
    for (i, asset) in assets.iter().enumerate() {
        if let Some(kind) = asset.asset_type.resource_kind() {
            let id = asset_id::intern(&asset.name);
            let handle = handles
                .get(kind, id)
                .expect("resource asset was assigned a handle");
            out.resource_jobs.push((i, asset.asset_type, handle));
            continue;
        }
        let req = AssetRequest {
            asset_type: asset.asset_type.as_str().to_string(),
            args: Some(asset.args.clone()),
        };
        let mut def = asset_api::create_asset_def(&req).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': {}", asset.name, e),
            )
        })?;
        def.name = Some(asset_id::intern(&asset.name));
        out.named.push((asset.name.clone(), def));
        out.named_src.push(i);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::fixtures::wja;
    use concinnity_core::blob::ResourceKind;

    fn handles_for(assets: &[WorldJsonlAsset]) -> ResourceHandles {
        asset_id::reset_interner();
        let names: Vec<&str> = assets.iter().map(|a| a.name.as_str()).collect();
        asset_id::intern_all(&names);
        ResourceHandles::from_assets(assets.iter().filter_map(|a| {
            crate::authoring::resource_type::asset_resource_kind(a.asset_type)
                .map(|kind| (asset_id::intern(&a.name), kind))
        }))
    }

    #[test]
    fn resources_become_jobs_and_components_become_defs() {
        let assets = vec![
            wja(
                "tex",
                RegisteredType::Texture,
                serde_json::json!({"generator": "checker"}),
            ),
            wja(
                "crate_mesh",
                RegisteredType::ProceduralMesh,
                serde_json::json!({}),
            ),
            wja(
                "clip",
                RegisteredType::AudioClip,
                serde_json::json!({"source": "a.ogg"}),
            ),
            wja(
                "tex2",
                RegisteredType::Texture,
                serde_json::json!({"source": "b.png"}),
            ),
        ];
        let handles = handles_for(&assets);
        let out = partition_components(&assets, &handles).expect("partitions");

        assert_eq!(
            out.resource_jobs,
            vec![
                (0, RegisteredType::Texture, 0),
                (2, RegisteredType::AudioClip, 0),
                (3, RegisteredType::Texture, 1),
            ]
        );
        assert_eq!(
            handles.get(ResourceKind::Texture, asset_id::intern("tex2")),
            Some(1)
        );
        assert_eq!(out.named.len(), 1);
        assert_eq!(out.named[0].0, "crate_mesh");
        assert_eq!(out.named[0].1.name, Some(asset_id::intern("crate_mesh")));
        assert_eq!(out.named_src, vec![1]);
    }

    #[test]
    fn a_prop_def_records_its_source_index() {
        let assets = vec![
            wja(
                "clip",
                RegisteredType::AudioClip,
                serde_json::json!({"source": "a.ogg"}),
            ),
            wja("box", RegisteredType::ProceduralMesh, serde_json::json!({})),
            wja(
                "crate",
                RegisteredType::Prop,
                serde_json::json!({"mesh": "box"}),
            ),
        ];
        let handles = handles_for(&assets);
        let out = partition_components(&assets, &handles).expect("partitions");

        let names: Vec<&str> = out.named.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["box", "crate"]);
        assert_eq!(out.named_src, vec![1, 2]);
        assert_eq!(
            Some(out.named[1].1.discriminant),
            RegisteredType::Prop.discriminant()
        );
    }

    #[test]
    fn bad_component_args_name_the_asset() {
        let assets = vec![wja(
            "broken_crate",
            RegisteredType::Prop,
            serde_json::json!({"position": "not a vector"}),
        )];
        let handles = handles_for(&assets);
        let err = partition_components(&assets, &handles)
            .err()
            .expect("bad args fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().starts_with("Asset 'broken_crate': "),
            "got: {err}"
        );
    }
}
