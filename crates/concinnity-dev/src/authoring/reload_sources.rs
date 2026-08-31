// src/authoring/reload_sources.rs
//
// Dev-only resource catalogues for a blob boot. The in-memory build installs
// them as world resources while it compiles -- the source paths
// `GraphicsSystem::init` seeds the hot-reload watcher from, and the material
// identities the live draw seam resolves an edit's reference through -- and a
// process that only LOADS prebuilt blobs would start without them, leaving
// file-backed assets invisible to the watcher and every material edit
// rebuilding. The world-lock records each resource's handle, name, and (for
// Texture / Mesh) source info, so those catalogues read straight off the lock,
// SceneImport products included. The ColorLut / EnvironmentMap singletons are
// components, not lock resources, so their sources still come from the
// authored entries.

use concinnity_cook::blob::BlobLock;

use crate::ecs::World;
use crate::resource::{
    ColorLutSources, EnvironmentMapSourceInfo, EnvironmentMapSources, MaterialNames, MeshSource,
    MeshSources, TextureSource, TextureSources,
};
use concinnity_cook::authoring::registry::RegisteredType;

// Reconstruct the source catalogues from the lock file at `path` plus the
// parsed authored entries, and install them as world resources. Best effort: a
// missing or old-format lock installs nothing (the pre-existing degraded
// behavior). Returns how many file-backed texture + mesh sources were
// installed.
pub(crate) fn install_from_lock(
    world: &mut World,
    entries: &[serde_json::Value],
    path: &std::path::Path,
) -> std::io::Result<usize> {
    let content = std::fs::read_to_string(path)?;
    let lock: BlobLock = serde_json::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(install(world, entries, &lock))
}

fn install(world: &mut World, entries: &[serde_json::Value], lock: &BlobLock) -> usize {
    let mut textures: Vec<TextureSource> = Vec::new();
    let mut meshes: Vec<MeshSource> = Vec::new();
    let mut materials: Vec<u32> = Vec::new();
    let mut installed = 0usize;
    for res in &lock.resources {
        if res.kind == RegisteredType::Material.as_str() {
            place(&mut materials, res.handle as usize, name_id(res));
        }
        if let Some(tex) = &res.texture_source {
            if !tex.source.is_empty() {
                installed += 1;
            }
            place(
                &mut textures,
                res.handle as usize,
                TextureSource {
                    name_id: name_id(res),
                    source: tex.source.clone(),
                    image_index: tex.image_index,
                },
            );
        }
        if let Some(mesh) = &res.mesh_source {
            if !mesh.source.is_empty() {
                installed += 1;
            }
            place(
                &mut meshes,
                res.handle as usize,
                MeshSource {
                    source: mesh.source.clone(),
                    primitive_index: mesh.primitive_index,
                    lod_levels: mesh.lod_levels,
                    lod_distances: mesh.lod_distances.clone(),
                },
            );
        }
    }

    world.insert_resource(TextureSources(textures));
    world.insert_resource(MeshSources(meshes));
    world.insert_resource(MaterialNames(materials));
    // The ColorLut / EnvironmentMap singletons are components, not lock
    // resources; their sources come straight from the authored entries.
    world.insert_resource(ColorLutSources(scan_color_lut(entries)));
    world.insert_resource(EnvironmentMapSources(scan_environment_map(entries)));
    installed
}

// The interned id the build assigned this resource's name, re-interning it
// where an old-format lock recorded none.
fn name_id(res: &concinnity_cook::blob::LockedResource) -> u32 {
    res.id
        .unwrap_or_else(|| crate::ecs::asset_id::intern(&res.name).0)
}

// Grow `vec` with defaults so `slot` is addressable, then write the entry.
// Handles are dense from 0 but can arrive out of order.
fn place<T: Default>(vec: &mut Vec<T>, slot: usize, entry: T) {
    if vec.len() <= slot {
        vec.resize_with(slot + 1, T::default);
    }
    vec[slot] = entry;
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

// Normalized type match (lowercase, underscores stripped), the convention the
// cook world passes use.
fn entry_is(entry: &serde_json::Value, norm_type: &str) -> bool {
    entry
        .get("type")
        .and_then(|v| v.as_str())
        .is_some_and(|t| t.to_lowercase().replace('_', "") == norm_type)
}

// The first declared ColorLut's authored `source` path (non-empty), or `None`.
fn scan_color_lut(entries: &[serde_json::Value]) -> Option<String> {
    entries
        .iter()
        .find(|e| entry_is(e, "colorlut"))
        .and_then(|e| e.get("args"))
        .map(|args| str_arg(args, "source"))
        .filter(|s| !s.is_empty())
}

// The first declared file-backed EnvironmentMap's re-bake inputs, or `None`
// (a procedural `generator` has no file to watch). Defaults mirror the
// EnvironmentMap schema.
fn scan_environment_map(entries: &[serde_json::Value]) -> Option<EnvironmentMapSourceInfo> {
    let args = entries
        .iter()
        .find(|e| entry_is(e, "environmentmap"))?
        .get("args")?;
    let source = str_arg(args, "source");
    if source.is_empty() || !str_arg(args, "generator").is_empty() {
        return None;
    }
    Some(EnvironmentMapSourceInfo {
        source,
        prefilter_face_size: u32_arg(args, "prefilter_face_size", 512),
        irradiance_face_size: u32_arg(args, "irradiance_face_size", 8),
        prefilter_samples: u32_arg(args, "prefilter_samples", 1024),
        prefilter_clamp: args
            .get("prefilter_clamp")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(12.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_cook::blob::{LockedMeshSource, LockedResource, LockedTextureSource};

    fn lock_with(resources: Vec<LockedResource>) -> BlobLock {
        serde_json::from_value(serde_json::json!({
            "engine_version": "0",
            "built_at": "",
            "blobs": [],
            "assets": [],
            "resources": serde_json::to_value(&resources).unwrap(),
            "injected": [],
            "shadowed": [],
        }))
        .unwrap()
    }

    fn tex_resource(name: &str, handle: u32, source: &str, image_index: u32) -> LockedResource {
        LockedResource {
            name: name.to_string(),
            id: Some(handle + 100),
            kind: "Texture".to_string(),
            handle,
            texture_source: Some(LockedTextureSource {
                source: source.to_string(),
                image_index,
            }),
            ..Default::default()
        }
    }

    fn entry(name: &str, ty: &str, args: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"name": name, "type": ty, "args": args})
    }

    #[test]
    fn texture_sources_reconstruct_by_handle_from_the_lock_alone() {
        // Out-of-order handles, and no authored entries at all: a SceneImport
        // product's source rides in the lock, not in world.jsonl.
        let lock = lock_with(vec![
            tex_resource("tex_b", 1, "pack.glb", 3),
            tex_resource("tex_a", 0, "a.png", 0),
        ]);
        let mut world = World::new();
        let installed = install(&mut world, &[], &lock);
        assert_eq!(installed, 2);
        let tex = world.resource::<TextureSources>().unwrap();
        assert_eq!(tex.0.len(), 2);
        assert_eq!(tex.0[0].source, "a.png");
        assert_eq!(tex.0[0].name_id, 100);
        assert_eq!(tex.0[1].source, "pack.glb");
        assert_eq!(tex.0[1].image_index, 3);
    }

    #[test]
    fn resources_without_source_info_are_skipped() {
        // An old-format lock (or a non-source kind) carries no source info;
        // its catalogue slot stays empty rather than guessing.
        let lock = lock_with(vec![LockedResource {
            name: "old_tex".to_string(),
            kind: "Texture".to_string(),
            ..Default::default()
        }]);
        let mut world = World::new();
        let installed = install(&mut world, &[], &lock);
        assert_eq!(installed, 0);
        assert!(world.resource::<TextureSources>().unwrap().0.is_empty());
        assert!(world.resource::<MeshSources>().unwrap().0.is_empty());
    }

    #[test]
    fn procedural_texture_leaves_an_empty_source() {
        let lock = lock_with(vec![tex_resource("noise", 0, "", 0)]);
        let mut world = World::new();
        let installed = install(&mut world, &[], &lock);
        assert_eq!(installed, 0);
        assert_eq!(world.resource::<TextureSources>().unwrap().0[0].source, "");
    }

    #[test]
    fn mesh_sources_carry_lod_shape() {
        let lock = lock_with(vec![LockedResource {
            name: "rock".to_string(),
            kind: "Mesh".to_string(),
            handle: 0,
            mesh_source: Some(LockedMeshSource {
                source: "rock.glb".to_string(),
                primitive_index: 2,
                lod_levels: 3,
                lod_distances: vec![10.0, 30.0],
            }),
            ..Default::default()
        }]);
        let mut world = World::new();
        let installed = install(&mut world, &[], &lock);
        assert_eq!(installed, 1);
        let meshes = world.resource::<MeshSources>().unwrap();
        assert_eq!(meshes.0[0].source, "rock.glb");
        assert_eq!(meshes.0[0].primitive_index, 2);
        assert_eq!(meshes.0[0].lod_levels, 3);
        assert_eq!(meshes.0[0].lod_distances, vec![10.0, 30.0]);
    }

    // A material's identity is what the live draw seam resolves an edit's
    // reference through; it rides the lock as a name + handle, in whatever
    // order the resources were recorded.
    #[test]
    fn material_names_reconstruct_by_handle() {
        let material = |name: &str, handle: u32| LockedResource {
            name: name.to_string(),
            id: Some(handle + 100),
            kind: "Material".to_string(),
            handle,
            ..Default::default()
        };
        let lock = lock_with(vec![
            material("glass", 1),
            tex_resource("tex_a", 0, "a.png", 0),
            material("steel", 0),
        ]);
        let mut world = World::new();
        install(&mut world, &[], &lock);
        assert_eq!(world.resource::<MaterialNames>().unwrap().0, vec![100, 101]);
    }

    #[test]
    fn texture_without_a_lock_id_interns_its_name() {
        let mut resource = tex_resource("late_tex", 0, "late.png", 0);
        resource.id = None;
        let lock = lock_with(vec![resource]);
        let mut world = World::new();
        install(&mut world, &[], &lock);
        let tex = world.resource::<TextureSources>().unwrap();
        assert_eq!(tex.0[0].name_id, crate::ecs::asset_id::intern("late_tex").0);
    }

    #[test]
    fn singleton_scans_read_the_authored_entries() {
        let lock = lock_with(Vec::new());
        let entries = vec![
            entry("grade", "ColorLut", serde_json::json!({"source": "g.cube"})),
            entry(
                "sky",
                "EnvironmentMap",
                serde_json::json!({"source": "sky.hdr", "prefilter_face_size": 128}),
            ),
        ];
        let mut world = World::new();
        install(&mut world, &entries, &lock);
        assert_eq!(
            world.resource::<ColorLutSources>().unwrap().0.as_deref(),
            Some("g.cube")
        );
        let env = world.resource::<EnvironmentMapSources>().unwrap();
        let info = env.0.as_ref().unwrap();
        assert_eq!(info.source, "sky.hdr");
        assert_eq!(info.prefilter_face_size, 128);
        assert_eq!(info.prefilter_samples, 1024);
    }

    #[test]
    fn procedural_environment_map_is_not_watchable() {
        let lock = lock_with(Vec::new());
        let entries = vec![entry(
            "sky",
            "EnvironmentMap",
            serde_json::json!({"source": "sky.hdr", "generator": "gradient"}),
        )];
        let mut world = World::new();
        install(&mut world, &entries, &lock);
        assert!(
            world
                .resource::<EnvironmentMapSources>()
                .unwrap()
                .0
                .is_none()
        );
    }
}
