//! The source catalogs a development build inserts beside the resource tables:
//! where each compiled resource came from, so the hot-reload watcher and the
//! editor can map a handle back to its file or its asset name. Absent in the
//! shipped disk runtime, which addresses every resource by handle alone.

use alloc::string::String;
use alloc::vec::Vec;

/// One texture's identity + source file, in `TextureHandle` order. A procedural
/// texture has an empty `source`. `name_id` is the interned asset name, used by
/// the runtime spawn-by-name path without interning at runtime.
#[derive(Debug, Clone, Default)]
pub struct TextureSource {
    /// The interned asset name.
    pub name_id: u32,
    /// Authored source path; empty for a procedural texture.
    pub source: String,
    /// Index of the image within the source document.
    pub image_index: u32,
}

/// Catalog of texture source files, indexed by `TextureHandle`. Graphics init
/// reads it to seed the hot-reload watcher.
#[derive(Debug, Clone, Default)]
pub struct TextureSources(pub Vec<TextureSource>);

/// Source catalog for the singleton `ColorLut`: the raw authored source path of
/// the first declared `ColorLut`, or `None`.
#[derive(Debug, Clone, Default)]
pub struct ColorLutSources(pub Option<String>);

/// One file-backed `EnvironmentMap`'s re-bake inputs, so a re-bake runs the IBL
/// convolution with the same dimensions the build used (a size change would
/// invalidate the shader's prefilter-mip assumptions).
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

/// Source catalog for the singleton `EnvironmentMap`. `Some` only for a
/// file-backed map (a procedural `generator` has nothing to watch).
#[derive(Debug, Clone, Default)]
pub struct EnvironmentMapSources(pub Option<EnvironmentMapSourceInfo>);

/// One file-backed Mesh's re-import inputs, in `MeshHandle` order. An
/// inline-authored mesh has an empty `source`.
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

/// Catalog of mesh source files, indexed by `MeshHandle`.
#[derive(Debug, Clone, Default)]
pub struct MeshSources(pub Vec<MeshSource>);

/// Catalog of material identities, in `MaterialHandle` order: the interned
/// asset name of each compiled `Material`. A material record carries its kind
/// and handle, not its name, so this is what lets an editor resolve the
/// material a Prop edit names to the handle the running world loaded it at.
#[derive(Debug, Clone, Default)]
pub struct MaterialNames(pub Vec<u32>);
