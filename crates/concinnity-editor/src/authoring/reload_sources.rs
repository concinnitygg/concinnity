// src/authoring/reload_sources.rs
//
// Hot-reload source catalogues for a blob boot. The in-memory build installs
// dev-only source catalogues (texture / mesh / ColorLut / EnvironmentMap ->
// file path) as world resources so `GraphicsSystem::init` can seed the
// hot-reload watcher; a process that only LOADS prebuilt blobs would start
// without them, leaving file-backed assets invisible to the watcher. The
// world-lock records each resource's name, kind, and handle, and the args
// live in the authored entries plus the lock's injected list, so the
// catalogues can be reconstructed by a name join. Assets whose args are not
// recorded (a SceneImport's generated products carry no args in the lock)
// are skipped; they join the catalogue on the next in-memory rebuild.

use std::collections::HashMap;

use concinnity_cook::blob::BlobLock;

use crate::ecs::World;
use crate::resource::{
    ColorLutSources, EnvironmentMapSourceInfo, EnvironmentMapSources, MeshSource, MeshSources,
    TextureSource, TextureSources,
};

// Reconstruct the source catalogues from the working directory's
// world-lock.json plus the parsed authored entries, and install them as world
// resources. Best effort: a missing or old-format lock installs nothing (the
// pre-existing degraded behavior). Returns how many file-backed texture +
// mesh sources were installed.
pub(crate) fn install_from_lock(
    world: &mut World,
    entries: &[serde_json::Value],
) -> std::io::Result<usize> {
    let content = std::fs::read_to_string(concinnity_cook::blob::LOCK_PATH)?;
    let lock: BlobLock = serde_json::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(install(world, entries, &lock))
}

fn install(world: &mut World, entries: &[serde_json::Value], lock: &BlobLock) -> usize {
    let args_by_name = args_by_name(entries, lock);

    let mut textures: Vec<TextureSource> = Vec::new();
    let mut meshes: Vec<MeshSource> = Vec::new();
    let mut installed = 0usize;
    for res in &lock.resources {
        let Some(args) = args_by_name.get(res.name.as_str()) else {
            continue;
        };
        match res.kind.as_str() {
            "Texture" => {
                let entry = texture_source(res.id, &res.name, args);
                if !entry.source.is_empty() {
                    installed += 1;
                }
                place(&mut textures, res.handle as usize, entry);
            }
            "Mesh" => {
                let entry = mesh_source(args);
                if !entry.source.is_empty() {
                    installed += 1;
                }
                place(&mut meshes, res.handle as usize, entry);
            }
            _ => {}
        }
    }

    world.insert_resource(TextureSources(textures));
    world.insert_resource(MeshSources(meshes));
    // The ColorLut / EnvironmentMap singletons are components, not lock
    // resources; their sources come straight from the authored entries.
    world.insert_resource(ColorLutSources(scan_color_lut(entries)));
    world.insert_resource(EnvironmentMapSources(scan_environment_map(entries)));
    installed
}

// Args for every asset whose declaration the boot can see: the authored
// entries plus the lock's injected companions. An authored entry wins a name
// collision (it is the override the build honored).
fn args_by_name<'a>(
    entries: &'a [serde_json::Value],
    lock: &'a BlobLock,
) -> HashMap<&'a str, &'a serde_json::Value> {
    let mut map: HashMap<&str, &serde_json::Value> = HashMap::new();
    for inj in &lock.injected {
        map.insert(inj.name.as_str(), &inj.args);
    }
    for entry in entries {
        if let (Some(name), Some(args)) = (
            entry.get("name").and_then(|v| v.as_str()),
            entry.get("args"),
        ) {
            map.insert(name, args);
        }
    }
    map
}

// Grow `vec` with defaults so `slot` is addressable, then write the entry.
// Handles are dense from 0, but the args join can visit them out of order.
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

// A texture's watchable source: the authored `source` path unless a
// `generator` makes it procedural (nothing to watch). Mirrors the cook's
// catalogue pass over the same args.
fn texture_source(id: Option<u32>, name: &str, args: &serde_json::Value) -> TextureSource {
    if !str_arg(args, "generator").is_empty() {
        return TextureSource::default();
    }
    TextureSource {
        name_id: id.unwrap_or_else(|| crate::ecs::asset_id::intern(name).0),
        source: str_arg(args, "source"),
        image_index: u32_arg(args, "image_index", 0),
    }
}

fn mesh_source(args: &serde_json::Value) -> MeshSource {
    MeshSource {
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
    use concinnity_cook::blob::{LockedInjection, LockedResource};

    fn lock_with(resources: Vec<LockedResource>, injected: Vec<LockedInjection>) -> BlobLock {
        serde_json::from_value(serde_json::json!({
            "engine_version": "0",
            "built_at": "",
            "blobs": [],
            "assets": [],
            "resources": serde_json::to_value(&resources).unwrap(),
            "injected": serde_json::to_value(&injected).unwrap(),
            "shadowed": [],
        }))
        .unwrap()
    }

    fn resource(name: &str, kind: &str, handle: u32) -> LockedResource {
        LockedResource {
            name: name.to_string(),
            id: Some(handle + 100),
            kind: kind.to_string(),
            handle,
            args_hash: String::new(),
            payload_blob: None,
        }
    }

    fn entry(name: &str, ty: &str, args: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"name": name, "type": ty, "args": args})
    }

    #[test]
    fn authored_texture_sources_reconstruct_by_handle() {
        let lock = lock_with(
            vec![
                resource("tex_b", "Texture", 1),
                resource("tex_a", "Texture", 0),
            ],
            Vec::new(),
        );
        let entries = vec![
            entry("tex_a", "Texture", serde_json::json!({"source": "a.png"})),
            entry(
                "tex_b",
                "Texture",
                serde_json::json!({"source": "pack.glb", "image_index": 3}),
            ),
        ];
        let mut world = World::new_empty();
        let installed = install(&mut world, &entries, &lock);
        assert_eq!(installed, 2);
        let tex = world.resource::<TextureSources>().unwrap();
        assert_eq!(tex.0.len(), 2);
        assert_eq!(tex.0[0].source, "a.png");
        assert_eq!(tex.0[1].source, "pack.glb");
        assert_eq!(tex.0[1].image_index, 3);
    }

    #[test]
    fn generated_textures_without_recorded_args_are_skipped() {
        // A SceneImport product is in the lock's resources but nowhere carries
        // args; its catalogue slot stays empty rather than mis-joining.
        let lock = lock_with(vec![resource("import_tex", "Texture", 0)], Vec::new());
        let mut world = World::new_empty();
        let installed = install(&mut world, &[], &lock);
        assert_eq!(installed, 0);
        assert!(world.resource::<TextureSources>().unwrap().0.is_empty());
    }

    #[test]
    fn procedural_texture_leaves_an_empty_source() {
        let lock = lock_with(vec![resource("noise", "Texture", 0)], Vec::new());
        let entries = vec![entry(
            "noise",
            "Texture",
            serde_json::json!({"generator": "noise", "source": "ignored.png"}),
        )];
        let mut world = World::new_empty();
        let installed = install(&mut world, &entries, &lock);
        assert_eq!(installed, 0);
        assert_eq!(world.resource::<TextureSources>().unwrap().0[0].source, "");
    }

    #[test]
    fn mesh_sources_carry_lod_shape() {
        let lock = lock_with(vec![resource("rock", "Mesh", 0)], Vec::new());
        let entries = vec![entry(
            "rock",
            "Mesh",
            serde_json::json!({
                "source": "rock.glb", "primitive_index": 2,
                "lod_levels": 3, "lod_distances": [10.0, 30.0]
            }),
        )];
        let mut world = World::new_empty();
        install(&mut world, &entries, &lock);
        let meshes = world.resource::<MeshSources>().unwrap();
        assert_eq!(meshes.0[0].source, "rock.glb");
        assert_eq!(meshes.0[0].primitive_index, 2);
        assert_eq!(meshes.0[0].lod_levels, 3);
        assert_eq!(meshes.0[0].lod_distances, vec![10.0, 30.0]);
    }

    #[test]
    fn injected_args_join_when_no_authored_entry_exists() {
        let lock = lock_with(
            vec![resource("companion_tex", "Texture", 0)],
            vec![LockedInjection {
                name: "companion_tex".to_string(),
                asset_type: "Texture".to_string(),
                args: serde_json::json!({"source": "chip.png"}),
                injected_by: "companion".to_string(),
            }],
        );
        let mut world = World::new_empty();
        let installed = install(&mut world, &[], &lock);
        assert_eq!(installed, 1);
        assert_eq!(
            world.resource::<TextureSources>().unwrap().0[0].source,
            "chip.png"
        );
    }

    #[test]
    fn singleton_scans_read_the_authored_entries() {
        let lock = lock_with(Vec::new(), Vec::new());
        let entries = vec![
            entry("grade", "ColorLut", serde_json::json!({"source": "g.cube"})),
            entry(
                "sky",
                "EnvironmentMap",
                serde_json::json!({"source": "sky.hdr", "prefilter_face_size": 128}),
            ),
        ];
        let mut world = World::new_empty();
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
        let lock = lock_with(Vec::new(), Vec::new());
        let entries = vec![entry(
            "sky",
            "EnvironmentMap",
            serde_json::json!({"source": "sky.hdr", "generator": "gradient"}),
        )];
        let mut world = World::new_empty();
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
