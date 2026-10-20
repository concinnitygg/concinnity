//! The source catalogs graphics init captures when the launch arms the dev
//! loop, bundled as the one `HotReloadSources` world resource the dev drive
//! takes once. The catalogs themselves are core's
//! `render::world_load::source_maps`.

use concinnity_core::components::{Identity, ProceduralMesh};
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::backend::init::SdfVolumeSource;
use concinnity_core::render::world_load::source_maps::{
    ColorLutSource, EnvironmentMapSource, MeshSourceMap, ProceduralMeshSourceMap, SdfFieldMap,
    ShaderSourceMap, SkinnedMeshSourceMap, TextureSourceMap,
};
use concinnity_host::thread::asset_id;
use hashbrown::HashMap;
use std::path::Path;

use super::shader_overrides::ShaderOverrides;

/// Bundle of every captured source catalog, parked as a world resource by
/// `GraphicsSystem` init and taken once by concinnity-dev's hot-reload drive,
/// which builds the filesystem watcher + `AssetHotReloadState` from it. Never
/// parked when the launch does not arm the dev loop.
#[derive(Default)]
pub struct HotReloadSources {
    /// Reloadable textures.
    pub map: TextureSourceMap,
    /// The reloadable color LUT, when the world declares one.
    pub color_lut: Option<ColorLutSource>,
    /// The reloadable environment map, when the world declares one.
    pub environment_map: Option<EnvironmentMapSource>,
    /// Reloadable static meshes.
    pub meshes: MeshSourceMap,
    /// Reloadable skinned meshes.
    pub skinned_meshes: SkinnedMeshSourceMap,
    /// Reloadable procedural meshes.
    pub procedural_meshes: ProceduralMeshSourceMap,
    /// Reloadable world Shaders.
    pub shaders: ShaderSourceMap,
    /// Hot-reloaded Shader programs, shared with the streaming pump so a
    /// Shader whose scene loads later installs the edit.
    pub shader_overrides: ShaderOverrides,
    /// Reloadable `SdfVolume` fields.
    pub sdf_fields: SdfFieldMap,
}

impl HotReloadSources {
    /// Whether nothing reloadable was captured, so no watcher is worth starting.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
            && self.color_lut.is_none()
            && self.environment_map.is_none()
            && self.meshes.is_empty()
            && self.skinned_meshes.is_empty()
            && self.procedural_meshes.is_empty()
            && self.shaders.is_empty()
            && self.sdf_fields.is_empty()
    }
}

// Snapshot each ProceduralMesh, with its interned name, before the mesh load
// drains them. A world.jsonl reload diffs a freshly parsed entry against this
// and regenerates the mesh when they differ, logging it by name.
pub(crate) fn procedural_mesh_snapshot(
    ctx: &PipelineContext<'_>,
) -> HashMap<AssetId, (String, ProceduralMesh)> {
    ctx.join2::<ProceduralMesh, Identity>()
        .filter_map(|(_, pm, identity)| {
            let id = identity.id();
            Some((id, (asset_id::name_of(id)?, pm.clone())))
        })
        .collect()
}

// The SdfVolume field catalog, each field resolved the way the cook found the
// file it compiled, under the build's asset root `assets_dir`.
pub(crate) fn resolve_sdf_fields(
    volumes: &[SdfVolumeSource],
    assets_dir: Option<&Path>,
) -> SdfFieldMap {
    SdfFieldMap::build(volumes, |raw| {
        concinnity_host::store::source::find_existing(raw, assets_dir)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::render_types::{DrawIndex, SkinnedIndex};
    use concinnity_core::render::world_load::source_maps::{
        MeshSourceEntry, ProceduralMeshSourceEntry, ShaderSourceEntry, SkinnedMeshSourceEntry,
    };

    fn mesh_entry(source: &str) -> MeshSourceEntry {
        MeshSourceEntry {
            source: source.to_string(),
            primitive_index: 0,
            lod_levels: 1,
            lod_distances: Vec::new(),
            draw_indices: vec![DrawIndex(0)],
        }
    }

    fn skinned_entry(source: &str) -> SkinnedMeshSourceEntry {
        SkinnedMeshSourceEntry {
            source: source.to_string(),
            skin_index: 0,
            skinned_index: SkinnedIndex(0),
            vertex_base: 0,
            vertex_count: 3,
            index_count: 3,
            joint_count: 1,
        }
    }

    // The handoff bundle defaults to nothing captured, the shape of a launch
    // without the dev loop: no watcher, no reloadable sources.
    #[test]
    fn bundle_defaults_to_nothing_captured() {
        let sources = HotReloadSources::default();
        assert!(sources.map.is_empty());
        assert!(sources.meshes.is_empty());
        assert!(sources.skinned_meshes.is_empty());
        assert!(sources.procedural_meshes.is_empty());
        assert!(sources.shaders.is_empty());
        assert!(sources.color_lut.is_none());
        assert!(sources.environment_map.is_none());
        assert!(sources.is_empty());
    }

    // Any one captured catalog is enough to make the bundle worth parking.
    #[test]
    fn bundle_with_any_single_capture_is_not_empty() {
        let populated: [fn(&mut HotReloadSources); 7] = [
            |s| s.map.push_texture("assets/wall.png".to_string(), 0, 0),
            |s| {
                s.color_lut = Some(ColorLutSource {
                    resolved_path: "assets/grade.cube".to_string(),
                });
            },
            |s| {
                s.environment_map = Some(EnvironmentMapSource {
                    resolved_path: "assets/sky.hdr".to_string(),
                    prefilter_face_size: 64,
                    irradiance_face_size: 16,
                    prefilter_samples: 32,
                    prefilter_clamp: 10.0,
                });
            },
            |s| s.meshes.entries.push(mesh_entry("assets/prop.glb")),
            |s| {
                s.skinned_meshes
                    .entries
                    .push(skinned_entry("assets/fox.glb"));
            },
            |s| {
                s.procedural_meshes.entries.push(ProceduralMeshSourceEntry {
                    name: "ground".to_string(),
                    args: ProceduralMesh::default(),
                    draw_indices: vec![DrawIndex(0)],
                });
            },
            |s| {
                s.shaders.entries.push(ShaderSourceEntry {
                    id: AssetId(1),
                    name: "scene".to_string(),
                    bucket: 0,
                    files: Vec::new(),
                });
            },
        ];
        for (i, populate) in populated.iter().enumerate() {
            let mut sources = HotReloadSources::default();
            populate(&mut sources);
            assert!(!sources.is_empty(), "field {i} alone is a capture");
        }
    }

    // A bare field name is found anywhere under the asset root, as the cook
    // finds it, so a field the build compiled from a nested directory is
    // watched there.
    #[test]
    fn a_field_resolves_where_the_cook_found_it() {
        use concinnity_core::components::SdfVolume;
        let source = |label: &str, field: &str| SdfVolumeSource {
            volume: SdfVolume {
                fragment_shader: field.to_string(),
                ..SdfVolume::default()
            },
            fragment_source: Vec::new(),
            label: label.to_string(),
        };
        let assets = tempfile::tempdir().unwrap();
        let fields = assets.path().join("fields");
        std::fs::create_dir_all(&fields).unwrap();
        std::fs::write(fields.join("blob.hlsl"), "").unwrap();
        let map = resolve_sdf_fields(
            &[
                source("blob", "blob.hlsl"),
                source("lost", "cn_no_such_field.hlsl"),
            ],
            Some(assets.path()),
        );
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get("blob").unwrap().resolved_path,
            fields.join("blob.hlsl").to_string_lossy()
        );
    }
}
