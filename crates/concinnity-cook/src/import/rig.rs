// src/import/rig.rs
//
// The rig half of a scene expansion: `SkinnedMesh` and `Animation` entries for
// a source file's skinned meshes and animation clips. Format-agnostic; the
// `.fbx` and `.glb` arms of `super` each discover their own parts and clips
// and hand them here so both containers generate identically shaped entries.
//
// Every skinned part of a file gets its own `SkinnedMesh` (a character is
// commonly body + hair + clothes bound to one skeleton) and its own copy of
// every clip: an `Animation` drives exactly one `SkinnedMesh`, so a single
// clip entry would leave the other parts standing in bind pose.

use super::sanitize_name;

// One skinned mesh a source file exposes.
pub(super) struct SkinnedPart {
    // Selector the generated asset carries, matching the importer's own scan
    // order over the file's skinned meshes.
    pub skin_index: usize,
    // Material asset name the part renders with.
    pub material: String,
}

// SkinnedMesh + Animation entries for one import. Empty when the file holds no
// skinned mesh, so a static scene expands exactly as it did before rigs were
// recognised; clips are likewise skipped when there is nothing to drive.
pub(super) fn rig_entries(
    prefix: &str,
    source: &str,
    parts: &[SkinnedPart],
    clips: &[String],
) -> Vec<serde_json::Value> {
    let mut entries = Vec::with_capacity(parts.len() * (1 + clips.len()));
    for part in parts {
        let mesh_name = skinned_mesh_name(prefix, part.skin_index);
        entries.push(serde_json::json!({
            "name": mesh_name,
            "type": "SkinnedMesh",
            "args": {
                "source": source,
                "skin_index": part.skin_index,
                "material": part.material,
                "position": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        }));
        for (clip_index, clip) in clips.iter().enumerate() {
            entries.push(serde_json::json!({
                "name": animation_name(prefix, part.skin_index, clip, clip_index),
                "type": "Animation",
                "args": {
                    "target": mesh_name,
                    "source": source,
                    "animation_index": clip_index,
                }
            }));
        }
    }
    entries
}

// Asset name for one skinned part, mirroring the `{prefix}_prim_{i}` shape the
// static meshes use.
fn skinned_mesh_name(prefix: &str, skin_index: usize) -> String {
    format!("{prefix}_skin_{skin_index}")
}

// Asset name for one clip on one part. Both indices are in the name: the clip
// index because sanitized clip names can collide, and the part index because
// every part carries its own copy of the clip.
fn animation_name(prefix: &str, skin_index: usize, clip: &str, clip_index: usize) -> String {
    let label = if clip.is_empty() {
        "clip".to_string()
    } else {
        sanitize_name(clip)
    };
    format!("{prefix}_anim_{skin_index}_{label}_{clip_index}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(skin_index: usize, material: &str) -> SkinnedPart {
        SkinnedPart {
            skin_index,
            material: material.to_string(),
        }
    }

    #[test]
    fn a_file_without_rigs_generates_nothing() {
        assert!(rig_entries("hero", "hero.glb", &[], &["walk".to_string()]).is_empty());
    }

    #[test]
    fn a_rig_without_clips_generates_only_the_mesh() {
        let entries = rig_entries("hero", "hero.glb", &[part(0, "hero_mat_0")], &[]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["name"], "hero_skin_0");
        assert_eq!(entries[0]["type"], "SkinnedMesh");
        assert_eq!(entries[0]["args"]["source"], "hero.glb");
        assert_eq!(entries[0]["args"]["skin_index"], serde_json::json!(0));
        assert_eq!(entries[0]["args"]["material"], "hero_mat_0");
        // An omitted scale would bake to zero and collapse the mesh.
        assert_eq!(
            entries[0]["args"]["scale"],
            serde_json::json!([1.0, 1.0, 1.0])
        );
    }

    #[test]
    fn every_part_gets_its_own_copy_of_every_clip() {
        let entries = rig_entries(
            "hero",
            "hero.glb",
            &[part(0, "hero_mat_0"), part(1, "hero_mat_1")],
            &["Walk".to_string(), "Run".to_string()],
        );
        let names: Vec<&str> = entries
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec![
                "hero_skin_0",
                "hero_anim_0_walk_0",
                "hero_anim_0_run_1",
                "hero_skin_1",
                "hero_anim_1_walk_0",
                "hero_anim_1_run_1",
            ]
        );

        // Each clip targets the part it was generated for, and selects itself
        // by index so duplicate clip names stay unambiguous.
        let run_on_hair = &entries[5];
        assert_eq!(run_on_hair["type"], "Animation");
        assert_eq!(run_on_hair["args"]["target"], "hero_skin_1");
        assert_eq!(run_on_hair["args"]["animation_index"], serde_json::json!(1));
        assert_eq!(run_on_hair["args"]["source"], "hero.glb");
    }

    #[test]
    fn an_unnamed_clip_falls_back_to_its_index() {
        let entries = rig_entries(
            "hero",
            "hero.fbx",
            &[part(0, "hero_mat_default")],
            &[String::new()],
        );
        assert_eq!(entries[1]["name"], "hero_anim_0_clip_0");
    }

    #[test]
    fn clip_names_are_sanitized_into_identifiers() {
        let entries = rig_entries(
            "hero",
            "hero.glb",
            &[part(0, "m")],
            &["Idle-Loop.02".to_string()],
        );
        assert_eq!(entries[1]["name"], "hero_anim_0_idle_loop_02_0");
    }
}
