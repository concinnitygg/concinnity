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

// One texture's identity + source file, in `TextureHandle` order. A procedural
// texture has an empty `source`. `name_id` is the interned asset name (the same
// interner the runtime shares in-process under `cn debug`), used by the runtime
// spawn-by-name path without interning at runtime.
#[derive(Debug, Clone, Default)]
pub struct TextureSource {
    pub name_id: u32,
    pub source: String,
    pub image_index: u32,
}

// Dev-only catalogue of texture source files, indexed by `TextureHandle`,
// inserted as a world resource by the in-memory (`cn debug` / editor) build.
// `GraphicsSystem::init` reads it to seed the hot-reload watcher now that Texture
// is a resource without a drained `source` field. Absent in the shipped disk
// runtime, which does not hot-reload; init simply captures no sources then.
#[derive(Debug, Clone, Default)]
pub struct TextureSources(pub Vec<TextureSource>);

// Dev-only source catalogue for the singleton ColorLut, inserted by the in-memory
// (`cn debug` / editor) build so `GraphicsSystem::init` can seed the hot-reload
// watcher now that ColorLut is a resource without a drained `source` field. The
// raw authored source path of the first declared ColorLut, or `None`. Absent in
// the shipped disk runtime, which does not hot-reload.
#[derive(Debug, Clone, Default)]
pub struct ColorLutSources(pub Option<String>);

// One file-backed EnvironmentMap's re-bake inputs, captured dev-only so the
// hot-reload watcher can re-run the IBL convolution with the same dimensions the
// build used (a size change would invalidate the shader's prefilter-mip
// assumptions).
#[derive(Debug, Clone, Default)]
pub struct EnvironmentMapSourceInfo {
    pub source: String,
    pub prefilter_face_size: u32,
    pub irradiance_face_size: u32,
    pub prefilter_samples: u32,
    pub prefilter_clamp: f32,
}

// Dev-only source catalogue for the singleton EnvironmentMap. `Some` only for a
// file-backed map (a procedural `generator` has nothing to watch). Mirrors
// [`ColorLutSources`]; absent in the shipped disk runtime.
#[derive(Debug, Clone, Default)]
pub struct EnvironmentMapSources(pub Option<EnvironmentMapSourceInfo>);

// One file-backed Mesh's re-import inputs, in `MeshHandle` order. Mirrors
// cook's `MeshSourceInfo`; an inline-authored mesh has an empty `source`.
#[derive(Debug, Clone, Default)]
pub struct MeshSource {
    pub source: String,
    pub primitive_index: u32,
    pub lod_levels: u32,
    pub lod_distances: Vec<f32>,
}

// Dev-only catalogue of mesh source files, indexed by `MeshHandle`, inserted as
// a world resource by the in-memory (`cn debug` / editor) build so
// `GraphicsSystem::init` can seed the hot-reload watcher now that Mesh is a
// resource without a drained `source` field. Absent in the shipped disk runtime.
#[derive(Debug, Clone, Default)]
pub struct MeshSources(pub Vec<MeshSource>);

// Install every per-kind resource table from a compiled blob's resource stream
// into `world`. This is the single place the table set is enumerated: the
// shipped runtime (`App::load_blob`), the editor's in-memory build, and the
// examples' `compile_world` all call it, so a resource kind that migrates into
// the stream gets wired into every host by adding one line here. Systems then
// read their table by handle. Dev-only source catalogues (hot-reload) stay with
// the debug path that captures them, not here.
pub fn install_resource_tables(world: &mut crate::ecs::World, records: &[ResourceRecord]) {
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
