// The blob record schema: the component defs stream and the resource records
// stream, postcard-serialized together as the `BlobMeta` block the header's
// meta_len measures. Interpretation of the records (discriminant -> component
// type, resource_kind -> table) belongs to the runtime registry, not here --
// these are containers, not meaning.

use alloc::vec::Vec;
use concinnity_asset::{AssetId, PayloadLocator};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BlobAssetDef {
    // The asset's interned identity. `None` for unnamed runtime-only assets.
    // Injected into the component at load time via `Component::inject_name`.
    pub name: Option<AssetId>,
    pub kind: AssetKind,
    pub discriminant: u8,
    // The serialized runtime component (cook already ran the asset -> component
    // translation), loaded via `Component::from_baked`. Every record is baked;
    // the transitional authored-args record kind is retired.
    #[serde(with = "serde_bytes")]
    pub args_bytes: Vec<u8>,
    pub payload: Option<PayloadLocator>,
}

// The blob carries only components: every system is internal client code,
// constructed at runtime from world content, never serialized. This kind is
// kept as the single discriminator the blob format records per asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AssetKind {
    Component,
}

// The kinds of resource the runtime keeps in per-kind tables, one dense handle
// space per kind. The `#[repr(u8)]` discriminant is the resource stream's
// `resource_kind` tag (like `ComponentTag` for components); cook writes it and
// the runtime selects the table by it. Order is the assignment order cook uses.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Mesh,
    Texture,
    Material,
    Font,
    AudioClip,
    CubemapTexture,
    EnvironmentMap,
    ColorLut,
    SkinnedMesh,
}

// One entry in the blob's resource stream: a compiled resource addressed by its
// dense per-kind handle, carried alongside the component stream. `resource_kind`
// selects the per-kind table (`ResourceKind as u8`); `handle` is the dense index
// within that kind (== the record's position within its kind). A payload
// resource (mesh, texture, audio clip) carries a `PayloadLocator` into the blob
// payload section; a data resource (a baked Material) carries its runtime bytes
// in `data_bytes`. Both fields are present so either shape round-trips; a given
// kind uses one branch (AudioClip uses `payload`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResourceRecord {
    pub resource_kind: u8,
    pub handle: u32,
    pub payload: Option<PayloadLocator>,
    #[serde(with = "serde_bytes")]
    pub data_bytes: Vec<u8>,
}

// A verified summary of the blob's shape, produced by cook from the final
// record streams and carried alongside them in the metadata block. The runtime
// trusts it (debug builds re-derive and assert it matches): the per-type
// counts pre-size the ECS columns before the bulk component load, and
// `max_blob_index` names the overflow files without scanning either stream.
// Anything further (type presence, feature flags) is deliberately not
// duplicated here: it is a counts lookup away.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct WorldManifest {
    // (component discriminant, record count), ascending, nonzero counts only.
    pub component_counts: Vec<(u8, u32)>,
    // Highest blob index any payload locator references; 0 = no overflow.
    pub max_blob_index: u32,
}

impl WorldManifest {
    // Derive the manifest from the final record streams. Cook builds the
    // shipped manifest with this (so it is consistent by construction); the
    // runtime re-derives it in debug builds to assert the shipped copy
    // matches.
    pub fn from_records(defs: &[BlobAssetDef], resources: &[ResourceRecord]) -> Self {
        let mut counts = [0u32; 256];
        for def in defs {
            counts[def.discriminant as usize] += 1;
        }
        let component_counts = counts
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n > 0)
            .map(|(d, &n)| (d as u8, n))
            .collect();
        let max_blob_index = defs
            .iter()
            .filter_map(|d| d.payload.as_ref())
            .chain(resources.iter().filter_map(|r| r.payload.as_ref()))
            .map(|p| p.blob_index)
            .max()
            .unwrap_or(0);
        WorldManifest {
            component_counts,
            max_blob_index,
        }
    }
}

// The blob's metadata section: the component stream, the resource stream, and
// the manifest summarizing them, postcard-serialized together as the block the
// header's `meta_len` measures. Folding everything into one block keeps the
// 16-byte header and every payload-offset computation
// (`payload_section_start`, the lock's `payload_bytes`) unchanged; only the
// block's contents grew. Blob 0 carries the full metadata; overflow blobs
// carry an empty `BlobMeta` (whose default manifest is consistent with its
// empty streams).
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct BlobMeta {
    pub defs: Vec<BlobAssetDef>,
    pub resources: Vec<ResourceRecord>,
    pub manifest: WorldManifest,
    pub scene_groups: Vec<SceneGroup>,
}

// One scene's exclusively-owned blob content: the resource-stream entries and
// payload-carrying component defs reachable only from that scene's members.
// Content shared between scenes (or used outside any scene) belongs to no
// group and loads with the world. Groups are listed in scene declaration
// order; their payloads are packed into dedicated blobs after the global set.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct SceneGroup {
    pub scene: AssetId,
    // (resource_kind, handle) pairs from the resource stream.
    pub resources: Vec<(u8, u32)>,
    // Names of payload-carrying component defs.
    pub defs: Vec<AssetId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta() -> BlobMeta {
        let defs = vec![BlobAssetDef {
            name: Some(AssetId(3)),
            kind: AssetKind::Component,
            discriminant: 42,
            args_bytes: vec![9, 8, 7],
            payload: Some(PayloadLocator {
                blob_index: 1,
                offset: 16,
                len: 4,
            }),
        }];
        let resources = vec![ResourceRecord {
            resource_kind: ResourceKind::Material as u8,
            handle: 5,
            payload: None,
            data_bytes: vec![1, 2, 3, 4],
        }];
        let manifest = WorldManifest::from_records(&defs, &resources);
        BlobMeta {
            defs,
            resources,
            manifest,
            scene_groups: vec![SceneGroup {
                scene: AssetId(7),
                resources: vec![(ResourceKind::Material as u8, 5)],
                defs: vec![AssetId(3)],
            }],
        }
    }

    // The postcard encoding of the metadata block is the on-disk format; every
    // record type must survive a byte round-trip unchanged.
    #[test]
    fn blob_meta_round_trips_through_postcard() {
        let meta = sample_meta();
        let bytes = postcard::to_allocvec(&meta).expect("serialize");
        let back: BlobMeta = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(back, meta);
    }

    // Both ResourceRecord branches (payload locator vs inline data bytes)
    // round-trip: payload resources and data resources share one record shape.
    #[test]
    fn resource_record_round_trips_both_branches() {
        let payload_res = ResourceRecord {
            resource_kind: ResourceKind::AudioClip as u8,
            handle: 0,
            payload: Some(PayloadLocator {
                blob_index: 0,
                offset: 0,
                len: 7,
            }),
            data_bytes: Vec::new(),
        };
        let data_res = ResourceRecord {
            resource_kind: ResourceKind::Material as u8,
            handle: 1,
            payload: None,
            data_bytes: vec![0xAA, 0xBB],
        };
        for rec in [payload_res, data_res] {
            let bytes = postcard::to_allocvec(&rec).expect("serialize");
            let back: ResourceRecord = postcard::from_bytes(&bytes).expect("deserialize");
            assert_eq!(back, rec);
        }
    }

    // The manifest is a pure function of the record streams: per-type counts
    // ascending with zero-count types omitted, and the highest blob index any
    // payload locator (component or resource) references.
    #[test]
    fn manifest_derives_counts_and_max_blob_index() {
        let def = |disc: u8, blob_index: u32| BlobAssetDef {
            name: None,
            kind: AssetKind::Component,
            discriminant: disc,
            args_bytes: Vec::new(),
            payload: Some(PayloadLocator {
                blob_index,
                offset: 0,
                len: 1,
            }),
        };
        let defs = vec![def(7, 0), def(7, 2), def(3, 1)];
        let resources = vec![ResourceRecord {
            resource_kind: ResourceKind::Texture as u8,
            handle: 0,
            payload: Some(PayloadLocator {
                blob_index: 4,
                offset: 0,
                len: 1,
            }),
            data_bytes: Vec::new(),
        }];
        let manifest = WorldManifest::from_records(&defs, &resources);
        assert_eq!(manifest.component_counts, vec![(3, 1), (7, 2)]);
        assert_eq!(manifest.max_blob_index, 4, "resource payloads count too");

        let empty = WorldManifest::from_records(&[], &[]);
        assert_eq!(empty, WorldManifest::default());
    }

    // The resource stream tag is the enum discriminant; a reorder would silently
    // re-key every table, so pin the current assignment.
    #[test]
    fn resource_kind_discriminants_are_stable() {
        assert_eq!(ResourceKind::Mesh as u8, 0);
        assert_eq!(ResourceKind::Texture as u8, 1);
        assert_eq!(ResourceKind::Material as u8, 2);
        assert_eq!(ResourceKind::Font as u8, 3);
        assert_eq!(ResourceKind::AudioClip as u8, 4);
        assert_eq!(ResourceKind::CubemapTexture as u8, 5);
        assert_eq!(ResourceKind::EnvironmentMap as u8, 6);
        assert_eq!(ResourceKind::ColorLut as u8, 7);
        assert_eq!(ResourceKind::SkinnedMesh as u8, 8);
    }
}
