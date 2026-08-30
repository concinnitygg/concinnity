//! What a pipeline run hands back: the compiled defs and packed blobs, plus the
//! dev-only source catalogues a `cn debug` build reads back by handle.

use concinnity_core::blob::{MeshBoundsRecord, PhysicsBudgetRecord, ResourceKind, SceneGroup};

use crate::ecs::{BlobAssetDef, ResourceRecord};

/// A texture's identity + on-disk source, in `TextureHandle` order. Now that
/// Texture is a resource (no `source`/`asset_id` on a component the renderer
/// drains), this is how a dev build hands the `cn debug` tools what they need: the
/// hot-reload watcher maps `source` -> handle, and the runtime spawn-by-name path
/// maps `name_id` -> handle. `source` is empty for a procedural texture (nothing
/// to watch). `name_id` is the interned asset name (same interner the runtime
/// shares in-process under `cn debug`), so nothing is interned at runtime.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextureSourceInfo {
    /// The interned asset name.
    pub name_id: u32,
    /// Authored source path; empty for a procedural texture.
    pub source: String,
    /// Index of the image within the source document.
    pub image_index: u32,
}

/// A file-backed Mesh's re-import inputs, in `MeshHandle` order (the Mesh block
/// leads the shared mesh-source handle space, so Mesh handles are dense from 0).
/// Now that Mesh is a resource (no `source` on a component the renderer drains),
/// this is how a dev build hands the `cn debug` hot-reload watcher what it needs
/// to re-import a saved `.glb`/`.fbx`. `source` is empty for an inline-authored
/// mesh (nothing to watch).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshSourceInfo {
    /// Authored source path; empty for an inline-authored mesh.
    pub source: String,
    /// Index of the primitive within the source document.
    pub primitive_index: u32,
    /// How many LODs the mesh declares, including LOD0.
    pub lod_levels: u32,
    /// Camera distance at which each LOD past 0 takes over.
    pub lod_distances: Vec<f32>,
}

/// The in-memory result of a complete build pipeline run.
/// Defs have payload locators filled in; `payloads[i]` is the raw bytes for
/// blob i. This can be used directly without touching disk.
pub struct PipelineResult {
    /// The compiled component defs, with payload locators filled in.
    pub defs: Vec<BlobAssetDef>,
    /// Asset name of each def, index-aligned with `defs` (defs only carry the
    /// interned id; the lock file records the readable name).
    pub names: Vec<String>,
    /// The blob's resource stream: compiled resources addressed by their dense
    /// per-kind handle, carried alongside the component defs. Empty until a
    /// resource kind migrates off the component registry (AudioClip first).
    pub resources: Vec<ResourceRecord>,
    /// Per-scene exclusively-owned blob content, in scene declaration order.
    pub scene_groups: Vec<SceneGroup>,
    /// Baked AABB + counts per static mesh payload, by mesh-source handle.
    pub mesh_bounds: Vec<MeshBoundsRecord>,
    /// The world's physics reservation, or `None` when it runs no physics.
    pub physics_budget: Option<PhysicsBudgetRecord>,
    // Unified mesh-source handle -> asset name for mesh payloads compiled as
    // component defs (ProceduralMesh and friends). Resource-stream Mesh
    // handles lead the space and resolve through `resources`; these resolve
    // through `names`/`defs`. Consumed by the thumbnail baker to compose a
    // Model's sub-meshes.
    pub(crate) mesh_component_names: Vec<(u32, String)>,
    /// Raw bytes of each blob, indexed by blob number.
    pub payloads: Vec<Vec<u8>>,
    // Compiled-asset payloads served from the build cache this run.
    pub(crate) cache_hits: usize,
    // Compiled-asset payloads compiled fresh this run.
    pub(crate) cache_misses: usize,
    /// File-backed texture sources in `TextureHandle` order, for the `cn debug`
    /// hot-reload watcher. Dev-only info; not written to the shipped blob.
    pub texture_sources: Vec<TextureSourceInfo>,
    /// File-backed mesh sources in `MeshHandle` order (dense over the Mesh block
    /// of the shared mesh-source space), for the `cn debug` hot-reload watcher.
    /// Dev-only info; not written to the shipped blob.
    pub mesh_sources: Vec<MeshSourceInfo>,
    // Lock-file provenance for the resource stream, index-aligned with
    // `resources` (records only carry the kind tag + handle; the lock records
    // the readable name and args hash).
    pub(crate) resource_locks: Vec<crate::blob::LockedResource>,
}

impl PipelineResult {
    /// The interned asset name of every compiled resource of `kind`, dense by
    /// its per-kind handle: the identity a runtime that addresses resources by
    /// handle has no other way to recover (a resource record carries its kind
    /// and handle, not its name). 0 where the build recorded no id.
    pub fn resource_names(&self, kind: ResourceKind) -> Vec<u32> {
        let mut names = Vec::new();
        for (record, lock) in self.resources.iter().zip(self.resource_locks.iter()) {
            if record.resource_kind != kind as u8 {
                continue;
            }
            let slot = record.handle as usize;
            if names.len() <= slot {
                names.resize(slot + 1, 0);
            }
            names[slot] = lock.id.unwrap_or_default();
        }
        names
    }

    /// The compiled payload bytes of the resource of `kind` declared under
    /// `name`, sliced out of the in-memory blob sections. `None` when no such
    /// resource was compiled or it carries no payload. The editor's glTF
    /// export reads a SkinnedMesh's composed geometry through this.
    pub fn resource_payload(&self, kind: ResourceKind, name: &str) -> Option<&[u8]> {
        let record = self
            .resources
            .iter()
            .zip(self.resource_locks.iter())
            .find(|(r, l)| r.resource_kind == kind as u8 && l.name == name)?
            .0;
        let loc = record.payload.as_ref()?;
        let blob = self.payloads.get(loc.blob_index as usize)?;
        let start = usize::try_from(loc.offset).ok()?;
        let end = start.checked_add(usize::try_from(loc.len).ok()?)?;
        blob.get(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::build_pipeline_from_str;
    use crate::pipeline::fixtures::SHADER_BUILD_LOCK;

    #[test]
    fn resource_payload_slices_the_named_resource() {
        let _guard = SHADER_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::compile::shader::install_stub_toolchain();
        let world = concat!(
            r#"{"name":"prism","type":"SkinnedMesh","args":{"#,
            r#""vertices":[{"pos":[0,0,0]},{"pos":[1,0,0]},{"pos":[0,1,0]}],"#,
            r#""indices":[0,1,2],"skeleton":[{"name":"root","parent":-1}],"#,
            r#""scale":[1,1,1]}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(world, None, None).expect("build");
        let bytes = result
            .resource_payload(ResourceKind::SkinnedMesh, "prism")
            .expect("named payload");
        let payload =
            concinnity_core::gfx::mesh_payload::deserialise_skinned_with_lods(bytes).unwrap();
        assert_eq!(payload.vertices.len(), 3);
        assert_eq!(payload.joints[0].name, "root");
        // The wrong name or the wrong kind finds nothing.
        assert!(
            result
                .resource_payload(ResourceKind::SkinnedMesh, "ghost")
                .is_none()
        );
        assert!(
            result
                .resource_payload(ResourceKind::Texture, "prism")
                .is_none()
        );
    }
}
