// src/resource.rs
//
// Engine-side resource-table wiring. The per-kind, handle-indexed tables
// themselves (an audio clip today; meshes / textures / materials on the Windows
// follow-up) are renderer-free and live in concinnity-core; this module
// re-exports them under the historical `crate::resource::*` paths, and adds the
// engine-only glue: `install_resource_tables`, which builds every table from a
// compiled blob's resource stream and inserts it as a World resource, plus the
// dev-only source catalogues the hot-reload path captures.

use concinnity_core::ecs::ResourceRecord;

// The per-kind runtime tables + their shared entry type live in concinnity-core
// so the physics / audio subsystem crates can reach them; re-export them under
// the historical `crate::resource::*` paths for every reader (the graphics
// systems, the editor's in-memory build, the examples' `compile_world`).
pub use concinnity_core::resource::{
    AudioClipTable, ColorLutTable, EnvironmentMapTable, FontTable, MaterialTable, MeshTable,
    ResourceEntry, SkinnedMeshTable, TextureTable,
};

/// One texture's identity + source file, in `TextureHandle` order. A procedural
/// texture has an empty `source`. `name_id` is the interned asset name (the same
/// interner the runtime shares in-process under `cn debug`), used by the runtime
/// spawn-by-name path without interning at runtime.
#[derive(Debug, Clone, Default)]
pub struct TextureSource {
    /// The interned asset name.
    pub name_id: u32,
    /// Authored source path; empty for a procedural texture.
    pub source: String,
    /// Index of the image within the source document.
    pub image_index: u32,
}

/// Dev-only catalogue of texture source files, indexed by `TextureHandle`,
/// inserted as a world resource by the in-memory (`cn debug` / editor) build.
/// `GraphicsSystem::init` reads it to seed the hot-reload watcher now that Texture
/// is a resource without a drained `source` field. Absent in the shipped disk
/// runtime, which does not hot-reload; init simply captures no sources then.
#[derive(Debug, Clone, Default)]
pub struct TextureSources(pub Vec<TextureSource>);

/// Dev-only source catalogue for the singleton ColorLut, inserted by the in-memory
/// (`cn debug` / editor) build so `GraphicsSystem::init` can seed the hot-reload
/// watcher now that ColorLut is a resource without a drained `source` field. The
/// raw authored source path of the first declared ColorLut, or `None`. Absent in
/// the shipped disk runtime, which does not hot-reload.
#[derive(Debug, Clone, Default)]
pub struct ColorLutSources(pub Option<String>);

/// One file-backed EnvironmentMap's re-bake inputs, captured dev-only so the
/// hot-reload watcher can re-run the IBL convolution with the same dimensions the
/// build used (a size change would invalidate the shader's prefilter-mip
/// assumptions).
#[derive(Debug, Clone, Default)]
pub struct EnvironmentMapSourceInfo {
    /// Authored source path of the environment map.
    pub source: String,
    /// Prefilter cube edge in pixels.
    pub prefilter_face_size: u32,
    /// Irradiance cube edge in pixels.
    pub irradiance_face_size: u32,
    /// Samples per prefilter texel.
    pub prefilter_samples: u32,
    /// Radiance clamp applied while prefiltering, to suppress fireflies.
    pub prefilter_clamp: f32,
}

/// Dev-only source catalogue for the singleton EnvironmentMap. `Some` only for a
/// file-backed map (a procedural `generator` has nothing to watch). Mirrors
/// [`ColorLutSources`]; absent in the shipped disk runtime.
#[derive(Debug, Clone, Default)]
pub struct EnvironmentMapSources(pub Option<EnvironmentMapSourceInfo>);

/// One file-backed Mesh's re-import inputs, in `MeshHandle` order. Mirrors
/// cook's `MeshSourceInfo`; an inline-authored mesh has an empty `source`.
#[derive(Debug, Clone, Default)]
pub struct MeshSource {
    /// Authored source path; empty for an inline-authored mesh.
    pub source: String,
    /// Index of the primitive within the source document.
    pub primitive_index: u32,
    /// How many LODs the mesh declares, including LOD0.
    pub lod_levels: u32,
    /// Camera distance at which each LOD past 0 takes over.
    pub lod_distances: Vec<f32>,
}

/// Dev-only catalogue of mesh source files, indexed by `MeshHandle`, inserted as
/// a world resource by the in-memory (`cn debug` / editor) build so
/// `GraphicsSystem::init` can seed the hot-reload watcher now that Mesh is a
/// resource without a drained `source` field. Absent in the shipped disk runtime.
#[derive(Debug, Clone, Default)]
pub struct MeshSources(pub Vec<MeshSource>);

/// Dev-only catalogue of material identities, in `MaterialHandle` order: the
/// interned asset name of each compiled `Material`. A material record carries
/// its kind and handle, not its name, so this is what lets an editor resolve
/// the material a Prop edit names to the handle the running world loaded it at.
/// Inserted by the in-memory (`cn debug` / editor) build, and by the editor's
/// blob boot from world-lock.json, so a session has it either way it started.
/// Absent in the shipped disk runtime, which addresses every material by handle.
#[derive(Debug, Clone, Default)]
pub struct MaterialNames(pub Vec<u32>);

/// Install every per-kind resource table from a compiled blob's resource stream
/// into `world`. This is the single place the table set is enumerated: the
/// shipped runtime (`App::load_blob`), the editor's in-memory build, and the
/// examples' `compile_world` all call it, so a resource kind that migrates into
/// the stream gets wired into every host by adding one line here. Systems then
/// read their table by handle. Each builder MOVES its kind's data bytes out of
/// the records, so the caller's record vec is spent scaffolding afterwards.
/// Dev-only source catalogues (hot-reload) stay with the debug path that
/// captures them, not here.
pub fn install_resource_tables(world: &mut crate::ecs::World, records: &mut [ResourceRecord]) {
    log_resource_footprint(records);
    world.insert_resource(AudioClipTable::from_records(records));
    world.insert_resource(TextureTable::from_records(records));
    world.insert_resource(ColorLutTable::from_records(records));
    world.insert_resource(EnvironmentMapTable::from_records(records));
    world.insert_resource(FontTable::from_records(records));
    world.insert_resource(MaterialTable::from_records(records));
    world.insert_resource(MeshTable::from_records(records));
    world.insert_resource(SkinnedMeshTable::from_records(records));
}

// Log the compiled-resource footprint at load: the payload bytes each record
// references in the blob (resident once the blob's payload section is read) plus
// the data-resource bytes the tables hold directly. A coarse figure toward the
// memory budget (see `app::budget`), surfaced at start so the resource load's
// weight is visible.
fn log_resource_footprint(records: &[ResourceRecord]) {
    if records.is_empty() {
        return;
    }
    let total: u64 = records
        .iter()
        .map(|r| r.data_bytes.len() as u64 + r.payload.as_ref().map_or(0, |p| p.len))
        .sum();
    tracing::info!(
        "Resource tables: {} record(s), {} MiB compiled",
        records.len(),
        total / (1024 * 1024)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::{PayloadLocator, ResourceKind};

    // Each kind gets a distinct record count, so a table that picked up another
    // kind's records shows as a wrong length. Order matches `table_lens`.
    const KIND_COUNTS: [(ResourceKind, u32); 8] = [
        (ResourceKind::AudioClip, 1),
        (ResourceKind::Texture, 2),
        (ResourceKind::ColorLut, 3),
        (ResourceKind::EnvironmentMap, 4),
        (ResourceKind::Font, 5),
        (ResourceKind::Material, 6),
        (ResourceKind::Mesh, 7),
        (ResourceKind::SkinnedMesh, 8),
    ];

    // One record carrying both a payload locator and data bytes, so either
    // branch of the footprint sum has something to add.
    fn record(kind: ResourceKind, handle: u32) -> ResourceRecord {
        ResourceRecord {
            resource_kind: kind as u8,
            handle,
            payload: Some(PayloadLocator {
                blob_index: 0,
                offset: handle as u64 * 16,
                len: 16,
            }),
            data_bytes: vec![handle as u8; 4],
        }
    }

    fn records() -> Vec<ResourceRecord> {
        KIND_COUNTS
            .iter()
            .flat_map(|&(kind, count)| (0..count).map(move |h| record(kind, h)))
            .collect()
    }

    // Every table's length, in `KIND_COUNTS` order. The `expect`s are the
    // assertion that each kind's table was installed at all.
    fn table_lens(world: &crate::ecs::World) -> [usize; 8] {
        [
            world.resource::<AudioClipTable>().expect("audio").0.len(),
            world.resource::<TextureTable>().expect("texture").0.len(),
            world
                .resource::<ColorLutTable>()
                .expect("color lut")
                .0
                .len(),
            world
                .resource::<EnvironmentMapTable>()
                .expect("env map")
                .0
                .len(),
            world.resource::<FontTable>().expect("font").0.len(),
            world.resource::<MaterialTable>().expect("material").0.len(),
            world.resource::<MeshTable>().expect("mesh").0.len(),
            world
                .resource::<SkinnedMeshTable>()
                .expect("skinned")
                .0
                .len(),
        ]
    }

    // Every per-kind table is installed as a world resource, each holding only
    // its own kind's records, at their handles.
    #[test]
    fn install_wires_every_kind_table_into_the_world() {
        let mut world = crate::ecs::World::new();
        install_resource_tables(&mut world, &mut records());

        assert_eq!(table_lens(&world), [1, 2, 3, 4, 5, 6, 7, 8]);

        // A record lands at its own handle rather than its position in the stream.
        let mesh = world.resource::<MeshTable>().expect("mesh table installed");
        assert_eq!(mesh.0[6].data_bytes, vec![6; 4]);
        assert_eq!(mesh.0[6].payload.as_ref().expect("payload kept").offset, 96);
    }

    // The empty stream (a world with no compiled resources) still installs all
    // eight tables, each empty, so a system reading its table by handle finds
    // one rather than a missing resource.
    #[test]
    fn install_with_no_records_installs_empty_tables() {
        let mut world = crate::ecs::World::new();
        install_resource_tables(&mut world, &mut []);
        assert_eq!(table_lens(&world), [0; 8]);
    }
}
