// src/anim_reload.rs
//
// Animation clip hot-reload: re-import each captured file-backed Animation from
// its source .glb and push the rebuilt clip into the live AnimationSystem. The
// GLB decode lives here (editor side) because the runtime crate links no image
// decoders; the runtime crate exposes only the catalogue (`reload_entries`) and
// the setter (`apply_reloaded_clip`). Mirrors the path the desugar pass takes at
// build time, so a hot-reloaded clip is byte-identical to a fresh `cn build`.

use std::collections::HashMap;

use crate::gfx::animation::AnimationSystem;
use crate::gfx::skinning::{AnimationClip, JointTrack, Keyframe};

// Re-import every file-backed clip when an asset-source change is pending.
// Driven by the debug server's per-frame tick. No-op when no file-backed clips
// were captured or no source change is pending.
pub(crate) fn reload_clips_if_pending(anim: &mut AnimationSystem) {
    if anim.reload_entries().is_empty()
        || !concinnity_engine::app::dev_flags::take_pending_animations()
    {
        return;
    }
    reload_clips(anim);
}

fn reload_clips(anim: &mut AnimationSystem) {
    // Parse each unique source .glb once per reload; many clips can target the
    // same character file.
    let mut parsed_cache: HashMap<String, concinnity_cook::gltf_source::GltfDoc> = HashMap::new();
    let mut reloaded = 0usize;
    let mut failed = 0usize;

    // Snapshot the catalogue so the &mut setter can run while we iterate.
    let entries = anim.reload_entries().to_vec();
    for entry in &entries {
        let imported = if entry.source.to_lowercase().ends_with(".fbx") {
            // FBX parses per entry: the importer owns its tree walk end to
            // end, and rigs are small at hot-reload scale.
            concinnity_cook::fbx::import_fbx_animation(
                &entry.source,
                entry.animation_index,
                &entry.animation_name,
                entry.sample_rate,
                entry.skin_index,
            )
        } else {
            match parsed_cache.get(&entry.source) {
                Some(d) => Ok(d),
                None => match concinnity_cook::glb::parse_glb(&entry.source) {
                    Ok(d) => Ok(&*parsed_cache.entry(entry.source.clone()).or_insert(d)),
                    Err(e) => Err(e),
                },
            }
            .and_then(|doc| {
                concinnity_cook::glb::import_glb_animation_from_doc(
                    doc,
                    &entry.source,
                    entry.skin_index,
                    entry.animation_index,
                    &entry.animation_name,
                )
            })
        };
        let imported = match imported {
            Ok(a) => a,
            Err(e) => {
                tracing::error!(
                    "animation hot-reload: failed to import animation from '{}': {} \
                     (clip slot {:?}:{} kept its old keyframes)",
                    entry.source,
                    e,
                    entry.target,
                    entry.clip_index
                );
                failed += 1;
                continue;
            }
        };
        let clip = imported_to_clip(&imported, entry.looping);
        if anim.apply_reloaded_clip(entry.target, entry.clip_index, clip, entry.weight) {
            reloaded += 1;
        } else {
            tracing::error!(
                "animation hot-reload: target {:?} clip {} no longer present (skipped)",
                entry.target,
                entry.clip_index
            );
            failed += 1;
        }
    }
    tracing::info!(
        "animation hot-reload: reloaded {} clip(s) ({} failed)",
        reloaded,
        failed
    );
}

// Convert a build-side ImportedAnimation into the runtime AnimationClip form.
fn imported_to_clip(
    imported: &concinnity_cook::glb::ImportedAnimation,
    looping: bool,
) -> AnimationClip {
    AnimationClip {
        root: None,
        duration: imported.duration.max(1e-3),
        looping,
        tracks: imported
            .tracks
            .iter()
            .map(|t| JointTrack {
                joint: t.joint,
                keys: t
                    .keys
                    .iter()
                    .map(|k| Keyframe {
                        time: k.time,
                        pose: k.pose,
                    })
                    .collect(),
            })
            .collect(),
        morph_keys: imported
            .morph_track
            .iter()
            .map(|k| (k.time, k.weights.clone()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::intern;
    use crate::test_support;

    // Minimal in-memory GLB fixture: a one-triangle skinned mesh with a
    // two-joint skeleton and one animation named "wave" (two translation keys
    // on the tip joint). Assembled by hand so no binary file is checked in.

    fn f32s(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn u16s(vals: &[u16]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn make_glb(json: &serde_json::Value, bin: &[u8]) -> Vec<u8> {
        let mut json_bytes = serde_json::to_vec(json).expect("serialise glTF json");
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }
        let mut bin_bytes = bin.to_vec();
        while !bin_bytes.len().is_multiple_of(4) {
            bin_bytes.push(0);
        }
        let total = 12 + 8 + json_bytes.len() + 8 + bin_bytes.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&json_bytes);
        out.extend_from_slice(&(bin_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin_bytes);
        out
    }

    fn skinned_glb() -> Vec<u8> {
        let mut bin = f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]); // positions -> 36
        bin.extend(u16s(&[0, 1, 2])); // indices -> 42
        bin.extend([0u8; 2]); // pad -> 44
        bin.extend([0u8; 12]); // JOINTS_0 -> 56
        bin.extend(f32s(&[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ])); // WEIGHTS_0 -> 104
        bin.extend(f32s(&[0.0, 1.0])); // animation input times -> 112
        bin.extend(f32s(&[0.0, 0.0, 0.0, 0.0, 2.0, 0.0])); // translations -> 136

        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 136}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 6},
                {"buffer": 0, "byteOffset": 44, "byteLength": 12},
                {"buffer": 0, "byteOffset": 56, "byteLength": 48},
                {"buffer": 0, "byteOffset": 104, "byteLength": 8},
                {"buffer": 0, "byteOffset": 112, "byteLength": 24}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                 "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
                {"bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC4"},
                {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
                {"bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR",
                 "min": [0.0], "max": [1.0]},
                {"bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC3"}
            ],
            "meshes": [{"primitives": [
                {"attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, "indices": 1}
            ]}],
            "skins": [{"joints": [2, 1]}],
            "nodes": [
                {"mesh": 0, "skin": 0},
                {"name": "root", "children": [2], "translation": [0.0, 1.0, 0.0]},
                {"name": "tip", "translation": [0.0, 0.5, 0.0]}
            ],
            "scenes": [{"nodes": [0, 1]}],
            "scene": 0,
            "animations": [{
                "name": "wave",
                "channels": [
                    {"sampler": 0, "target": {"node": 2, "path": "translation"}}
                ],
                "samplers": [
                    {"input": 4, "output": 5, "interpolation": "LINEAR"}
                ]
            }]
        });
        make_glb(&json, &bin)
    }

    fn write_fixture(dir: &tempfile::TempDir) -> String {
        let path = dir.path().join("hero.glb");
        std::fs::write(&path, skinned_glb()).unwrap();
        path.to_string_lossy().into_owned()
    }

    // A file-backed Animation targeting a SkinnedMesh named "reload_hero".
    fn file_backed_animation(source: &str, animation_name: &str) -> crate::components::Animation {
        crate::components::Animation {
            asset_id: intern("reload_clip"),
            target: Some(crate::ecs::SkinnedMeshHandle(intern("reload_hero").0)),
            source: source.to_string(),
            animation_name: animation_name.to_string(),
            ..Default::default()
        }
    }

    // Build a world whose AnimationSystem captured one reload entry. Capture
    // only happens under the process-wide dev flag, so callers must hold the
    // shared test lock; the flag is restored before returning.
    fn world_with_reload_entry(source: &str, animation_name: &str) -> crate::ecs::World {
        concinnity_engine::app::dev_flags::set_enabled(true);
        let mut world = crate::ecs::World::new();
        world.add_component(file_backed_animation(source, animation_name));
        let started = world.start();
        concinnity_engine::app::dev_flags::set_enabled(false);
        started.unwrap();
        world
    }

    fn with_anim<R>(world: &mut crate::ecs::World, f: impl FnOnce(&mut AnimationSystem) -> R) -> R {
        for system in world.systems_mut() {
            if let crate::ecs::SystemAsset::AnimationSystem(anim) = system {
                return f(anim);
            }
        }
        panic!("AnimationSystem not constructed");
    }

    #[test]
    fn reload_rebuilds_a_clip_from_its_source() {
        let _guard = test_support::lock();
        let dir = tempfile::tempdir().unwrap();
        let source = write_fixture(&dir);
        let mut world = world_with_reload_entry(&source, "wave");
        with_anim(&mut world, |anim| {
            let entries = anim.reload_entries().to_vec();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].source, source);
            assert_eq!(entries[0].clip_index, 0);
            assert_eq!(entries[0].animation_name, "wave");
            // The re-import parses the source, selects "wave", and swaps the
            // rebuilt clip into the captured slot.
            reload_clips(anim);
            // The catalogue itself is untouched by a reload.
            assert_eq!(anim.reload_entries().len(), 1);
        });
    }

    #[test]
    fn reload_with_a_missing_source_keeps_the_old_clip() {
        let _guard = test_support::lock();
        let dir = tempfile::tempdir().unwrap();
        let source = dir
            .path()
            .join("missing.glb")
            .to_string_lossy()
            .into_owned();
        let mut world = world_with_reload_entry(&source, "wave");
        with_anim(&mut world, |anim| {
            assert_eq!(anim.reload_entries().len(), 1);
            // The parse failure is reported per clip; nothing panics and the
            // catalogue survives for the next attempt.
            reload_clips(anim);
            assert_eq!(anim.reload_entries().len(), 1);
        });
    }

    #[test]
    fn reload_with_an_unknown_animation_name_keeps_the_old_clip() {
        let _guard = test_support::lock();
        let dir = tempfile::tempdir().unwrap();
        let source = write_fixture(&dir);
        let mut world = world_with_reload_entry(&source, "sprint");
        with_anim(&mut world, |anim| {
            reload_clips(anim);
            assert_eq!(anim.reload_entries().len(), 1);
        });
    }

    #[test]
    fn reload_if_pending_gates_on_entries_and_the_flag() {
        use concinnity_engine::app::dev_flags;
        let _guard = test_support::lock();

        // No entries: the early-out fires before the flag is consumed.
        let mut empty = AnimationSystem::new();
        dev_flags::set_pending_animations();
        reload_clips_if_pending(&mut empty);
        assert!(
            dev_flags::take_pending_animations(),
            "empty catalogue must not consume the pending flag"
        );

        // Entries present but no pending flag: nothing to do.
        let dir = tempfile::tempdir().unwrap();
        let source = write_fixture(&dir);
        let mut world = world_with_reload_entry(&source, "wave");
        with_anim(&mut world, |anim| {
            reload_clips_if_pending(anim);
        });

        // Entries present and the flag raised: the reload consumes it.
        dev_flags::set_pending_animations();
        with_anim(&mut world, |anim| {
            reload_clips_if_pending(anim);
        });
        assert!(
            !dev_flags::take_pending_animations(),
            "a reload pass must consume the pending flag"
        );
    }

    #[test]
    fn imported_to_clip_maps_tracks_and_clamps_duration() {
        let doc =
            concinnity_cook::gltf_source::GltfDoc::from_slice(&skinned_glb(), None, "hero.glb")
                .expect("fixture parses");
        let imported =
            concinnity_cook::glb::import_glb_animation_from_doc(&doc, "hero.glb", 0, 0, "wave")
                .expect("fixture animation imports");
        let clip = imported_to_clip(&imported, true);
        assert!(clip.looping);
        assert!((clip.duration - 1.0).abs() < 1e-6);
        assert_eq!(clip.tracks.len(), 1);
        assert_eq!(clip.tracks[0].keys.len(), 2);
        assert_eq!(clip.tracks[0].keys[1].time, 1.0);

        // A degenerate zero-length import clamps to a positive duration.
        let zero = concinnity_cook::glb::ImportedAnimation {
            morph_track: Vec::new(),
            name: "flat".to_string(),
            duration: 0.0,
            tracks: Vec::new(),
        };
        let clip = imported_to_clip(&zero, false);
        assert!(!clip.looping);
        assert_eq!(clip.duration, 1e-3);
        assert!(clip.tracks.is_empty());
    }
}
