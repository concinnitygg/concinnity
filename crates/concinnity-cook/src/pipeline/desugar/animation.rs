//! Animation clips imported from a source file, and the root-motion bake that
//! lifts a clip's root track into the asset's args.

use std::path::Path;

use serde::Deserialize;

use crate::authoring::world::WorldJsonlAsset;

use super::super::SKINNED_MESH_TYPE;
use super::skin_index_arg;

// Skin selector per SkinnedMesh asset name. An Animation resolves its channels
// against its target's skeleton, so it inherits the target's selector rather
// than carrying its own: the two must agree or joint indices bind to a
// different skeleton and the clip silently mis-poses.
fn skin_index_by_target(assets: &[WorldJsonlAsset]) -> std::collections::HashMap<String, u32> {
    assets
        .iter()
        .filter(|a| a.asset_type == SKINNED_MESH_TYPE)
        .map(|a| (a.name.clone(), skin_index_arg(a)))
        .collect()
}

// Expand file-sourced `Animation` assets in place, dispatching on the source
// extension: `.fbx` clips bake through the FBX importer (at the asset's
// `sample_rate`), everything else parses as glTF. The clip is picked by
// `animation_name` (preferred) or `animation_index` and the asset's
// `duration` + `tracks` are replaced with the imported data. An Animation
// with no `source` is left untouched, so inline-authored clips are
// byte-for-byte unchanged. Channels targeting non-joint nodes are dropped
// silently by the importers.
pub(in crate::pipeline) fn desugar_animation_imports(
    assets: &mut [WorldJsonlAsset],
    assets_dir: Option<&Path>,
) -> std::io::Result<()> {
    use crate::components::Animation;
    use crate::ecs::Component;

    let skin_by_target = skin_index_by_target(assets);

    for asset in assets.iter_mut() {
        if asset.asset_type != Animation::NAME {
            continue;
        }
        let source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if source.is_empty() {
            continue;
        }
        let animation_name = asset
            .args
            .get("animation_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let animation_index = asset
            .args
            .get("animation_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let skin_index = asset
            .args
            .get("target")
            .and_then(|v| v.as_str())
            .and_then(|t| skin_by_target.get(t))
            .copied()
            .unwrap_or(0);

        let imported = if source.to_lowercase().ends_with(".fbx") {
            let sample_rate = asset
                .args
                .get("sample_rate")
                .and_then(|v| v.as_f64())
                .unwrap_or(30.0) as f32;
            crate::import::fbx::import_fbx_animation(
                &source,
                animation_index as u32,
                &animation_name,
                sample_rate,
                skin_index,
            )
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': FBX import failed: {}", asset.name, e),
                )
            })?
        } else {
            // Look up by name when authored; fall back to the numeric index.
            let resolved_index = if !animation_name.is_empty() {
                let names =
                    crate::import::gltf::glb_animation_names(&source, assets_dir).map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Asset '{}': glTF import failed: {}", asset.name, e),
                        )
                    })?;
                names
                    .iter()
                    .position(|n| n == &animation_name)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!(
                                "Asset '{}': glTF '{}' has no animation named '{}' \
                                 (file contains: {:?})",
                                asset.name, source, animation_name, names
                            ),
                        )
                    })?
            } else {
                animation_index
            };

            crate::import::gltf::import_glb_animation(
                &source,
                resolved_index,
                skin_index,
                assets_dir,
            )
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': glTF import failed: {}", asset.name, e),
                )
            })?
        };

        // Convert ImportedAnimation -> the asset's serialised track shape.
        let tracks_json: Vec<serde_json::Value> = imported
            .tracks
            .iter()
            .map(|track| {
                let keyframes: Vec<serde_json::Value> = track
                    .keys
                    .iter()
                    .map(|k| {
                        serde_json::json!({
                            "time": k.time,
                            "translation": k.pose.translation,
                            "rotation_deg": k.pose.rotation_deg,
                            "scale": k.pose.scale,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "joint": track.joint,
                    "keyframes": keyframes,
                })
            })
            .collect();

        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        obj.insert("duration".to_string(), serde_json::json!(imported.duration));
        obj.insert("tracks".to_string(), serde_json::Value::Array(tracks_json));
        if !imported.morph_track.is_empty() {
            let morph_json: Vec<serde_json::Value> = imported
                .morph_track
                .iter()
                .map(|k| serde_json::json!({"time": k.time, "weights": k.weights}))
                .collect();
            obj.insert(
                "morph_track".to_string(),
                serde_json::Value::Array(morph_json),
            );
        }
        tracing::info!(
            "Asset '{}': imported '{}' animation '{}': {:.3} s, {} track(s), {} morph key(s)",
            asset.name,
            source,
            imported.name,
            imported.duration,
            imported.tracks.len(),
            imported.morph_track.len(),
        );
    }
    Ok(())
}

// Bake root motion on every Animation that opted in: strip the root joint's
// travel out of the pose tracks into the asset's `root_track` (see
// `root_motion::bake_root_motion`). Runs after the glTF pass so imported
// tracks are already inline; an Animation without `root_motion` is
// untouched. A root-motion clip whose root joint has no track produces an
// empty curve, which would silently never move a character, so it warns.
pub(in crate::pipeline) fn desugar_root_motion(
    assets: &mut [WorldJsonlAsset],
) -> std::io::Result<()> {
    use crate::components::Animation;
    use crate::ecs::Component;

    // This deserializes each flagged clip (whose `target` is a name reference),
    // so the name resolver must be installed. The full pipeline resets the
    // interner before reaching here; installing it again is a cheap no-op and
    // keeps this pass correct when called on its own.
    crate::ecs::asset_id::ensure_name_resolver();

    for asset in assets.iter_mut() {
        if asset.asset_type != Animation::NAME
            || asset.args.get("root_motion").and_then(|v| v.as_bool()) != Some(true)
        {
            continue;
        }
        let mut anim: Animation = Deserialize::deserialize(&asset.args).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Asset '{}': root-motion bake failed to parse args: {}",
                    asset.name, e
                ),
            )
        })?;
        crate::compile::root_motion::bake_root_motion(&mut anim);
        if anim.root_track.is_empty() {
            tracing::warn!(
                "Asset '{}': root_motion is set but the clip has no track on the root \
                 joint; the character will not move",
                asset.name
            );
        }
        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        obj.insert(
            "tracks".to_string(),
            serde_json::to_value(&anim.tracks).expect("serialize animation tracks"),
        );
        obj.insert(
            "root_track".to_string(),
            serde_json::to_value(&anim.root_track).expect("serialize root track"),
        );
        tracing::info!(
            "Asset '{}': baked root motion ({} key(s){})",
            asset.name,
            anim.root_track.len(),
            if anim.root_motion_y { ", incl. Y" } else { "" },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::desugar::fixtures::{morphing_skinned_glb, skinned_fbx};
    use crate::pipeline::fixtures::{wja, write_fixture};

    // Animation with no `source` is left byte-for-byte unchanged: the
    // inline-authored path must not regress.
    #[test]
    fn desugar_animation_imports_skips_inline_clips() {
        let original = serde_json::json!({
            "target": "flag",
            "duration": 2.0,
            "tracks": [{"joint": 0, "keyframes": [{"time": 0.0, "rotation_deg": [0,0,0]}]}],
        });
        let mut assets = vec![crate::authoring::world::WorldJsonlAsset {
            name: "wave".to_string(),
            asset_type: "Animation".to_string(),
            args: original.clone(),
        }];
        desugar_animation_imports(&mut assets, None).expect("desugar succeeds");
        assert_eq!(assets[0].args, original);
    }

    // Opting into root motion strips the root joint's X/Z travel into
    // `root_track` and anchors the pose; a clip without the flag is
    // untouched, and a second pass over already-baked args is a no-op.
    #[test]
    fn desugar_root_motion_bakes_the_root_track() {
        let walk = serde_json::json!({
            "target": "hero",
            "duration": 1.0,
            "root_motion": true,
            "tracks": [{"joint": 0, "keyframes": [
                {"time": 0.0, "translation": [0.0, 1.0, 0.0]},
                {"time": 1.0, "translation": [2.0, 1.0, 0.0]}
            ]}],
        });
        let plain = serde_json::json!({
            "target": "hero",
            "duration": 1.0,
            "tracks": [{"joint": 0, "keyframes": [
                {"time": 1.0, "translation": [2.0, 1.0, 0.0]}
            ]}],
        });
        let mut assets = vec![
            crate::authoring::world::WorldJsonlAsset {
                name: "walk".to_string(),
                asset_type: "Animation".to_string(),
                args: walk,
            },
            crate::authoring::world::WorldJsonlAsset {
                name: "plain".to_string(),
                asset_type: "Animation".to_string(),
                args: plain.clone(),
            },
        ];
        desugar_root_motion(&mut assets).expect("desugar succeeds");

        let baked = &assets[0].args;
        assert_eq!(baked["root_track"][1]["translation"][0], 2.0);
        assert_eq!(baked["root_track"][1]["translation"][1], 0.0);
        // The pose keeps Y but stays anchored on X.
        assert_eq!(baked["tracks"][0]["keyframes"][1]["translation"][0], 0.0);
        assert_eq!(baked["tracks"][0]["keyframes"][1]["translation"][1], 1.0);
        assert_eq!(assets[1].args, plain, "flag-less clip untouched");

        let after_first = assets[0].args.clone();
        desugar_root_motion(&mut assets).expect("second pass succeeds");
        assert_eq!(assets[0].args, after_first, "re-bake is a no-op");
    }

    // A clip that animates morph weights carries a morph track beside its
    // joint tracks.
    #[test]
    fn desugar_animation_imports_inlines_a_morph_weight_track() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "hero.glb", &morphing_skinned_glb());
        let mut assets = vec![wja(
            "wave",
            "Animation",
            serde_json::json!({"source": src, "animation_index": 0}),
        )];
        desugar_animation_imports(&mut assets, None).expect("desugar");

        let morph = assets[0].args["morph_track"]
            .as_array()
            .expect("morph track inlined");
        assert_eq!(morph.len(), 2);
        assert_eq!(morph[0], serde_json::json!({"time": 0.0, "weights": [0.0]}));
        assert_eq!(morph[1], serde_json::json!({"time": 1.0, "weights": [1.0]}));
    }

    #[test]
    fn desugar_animation_imports_inherit_the_targets_skin() {
        use crate::import::glb::test_fixtures::two_skin_glb;

        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "hero.glb", &two_skin_glb());
        let mut assets = vec![
            wja(
                "hair",
                SKINNED_MESH_TYPE,
                serde_json::json!({"source": src, "skin_index": 1}),
            ),
            wja(
                "hair_wave",
                "Animation",
                serde_json::json!({"target": "hair", "source": src}),
            ),
        ];
        // The clip carries no selector of its own; resolving it against the
        // target's skin is what keeps the joint indices in the same space.
        assert_eq!(skin_index_by_target(&assets).get("hair"), Some(&1));
        desugar_animation_imports(&mut assets, None).expect("desugar");
        assert!(
            !assets[1].args["tracks"]
                .as_array()
                .expect("tracks")
                .is_empty()
        );
    }

    #[test]
    fn an_animation_without_a_resolvable_target_falls_back_to_the_first_skin() {
        let assets = vec![
            wja(
                "body",
                SKINNED_MESH_TYPE,
                serde_json::json!({"skin_index": 2}),
            ),
            wja("orphan", "Animation", serde_json::json!({"target": "gone"})),
        ];
        let by_target = skin_index_by_target(&assets);
        assert_eq!(by_target.get("body"), Some(&2));
        assert!(!by_target.contains_key("gone"));
    }

    // An `.fbx` source routes to the FBX importer, which bakes the clip at the
    // asset's `sample_rate` rather than replaying authored keys.
    #[test]
    fn desugar_animation_imports_bakes_an_fbx_clip_at_the_sample_rate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "hero.fbx", &skinned_fbx(true));
        let mut assets = vec![wja(
            "wave",
            "Animation",
            serde_json::json!({"source": src, "sample_rate": 10.0}),
        )];
        desugar_animation_imports(&mut assets, None).expect("desugar");

        let args = &assets[0].args;
        assert!(
            (args["duration"].as_f64().expect("duration") - 1.0).abs() < 1e-3,
            "got: {}",
            args["duration"]
        );
        let tracks = args["tracks"].as_array().expect("tracks inlined");
        assert_eq!(tracks.len(), 1);
        let keys = tracks[0]["keyframes"].as_array().expect("keyframes");
        // One second at 10 samples per second, inclusive of both ends.
        assert_eq!(keys.len(), 11);
        assert_eq!(keys[0]["translation"][0], 0.0);
        assert_eq!(keys[10]["translation"][0], 2.0);
    }

    // A clip named by `animation_name` that the file does not contain fails
    // through the FBX importer too.
    #[test]
    fn desugar_animation_imports_reports_an_unknown_fbx_clip_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "hero.fbx", &skinned_fbx(true));
        let mut assets = vec![wja(
            "run",
            "Animation",
            serde_json::json!({"source": src, "animation_name": "sprint"}),
        )];
        let err = desugar_animation_imports(&mut assets, None).expect_err("no 'sprint' clip");
        let msg = err.to_string();
        assert!(msg.contains("Asset 'run'"), "got: {msg}");
        assert!(msg.contains("FBX import failed"), "got: {msg}");
    }

    #[test]
    fn desugar_animation_imports_missing_source_errors() {
        let mut assets = vec![wja(
            "walk",
            "Animation",
            serde_json::json!({"source": "/no/such/anim.glb"}),
        )];
        let err = desugar_animation_imports(&mut assets, None).expect_err("missing .glb");
        assert!(err.to_string().contains("Asset 'walk'"), "got: {err}");
    }

    #[test]
    fn desugar_animation_imports_missing_named_clip_errors() {
        // The by-name lookup also starts by reading the file, so a missing
        // source fails before the name search; the error still names the asset.
        let mut assets = vec![wja(
            "run",
            "Animation",
            serde_json::json!({"source": "/no/such/anim.glb", "animation_name": "Run"}),
        )];
        let err = desugar_animation_imports(&mut assets, None).expect_err("missing .glb");
        assert!(err.to_string().contains("Asset 'run'"), "got: {err}");
    }

    // A source-backed Animation is replaced by the imported clip's duration
    // and tracks. Channels targeting non-joint nodes are dropped, so the
    // fixture's two channels yield one track.
    #[test]
    fn desugar_animation_imports_inlines_the_indexed_clip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(
            &dir,
            "hero.glb",
            &crate::import::glb::test_fixtures::skinned_glb(),
        );
        let mut assets = vec![wja(
            "wave",
            "Animation",
            serde_json::json!({"source": src, "animation_index": 0}),
        )];
        desugar_animation_imports(&mut assets, None).expect("desugar");

        let args = &assets[0].args;
        assert_eq!(args["duration"], 1.0);
        let tracks = args["tracks"].as_array().expect("tracks inlined");
        assert_eq!(tracks.len(), 1, "the non-joint channel is dropped");
        let keys = tracks[0]["keyframes"].as_array().expect("keyframes");
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0]["time"], 0.0);
        assert_eq!(keys[1]["translation"], serde_json::json!([0.0, 2.0, 0.0]));
        // The fixture animates no morph weights, so no morph track appears.
        assert!(args.get("morph_track").is_none());
    }

    // `animation_name` picks the clip by name; a name the file does not carry
    // is a hard error that lists what it does contain.
    #[test]
    fn desugar_animation_imports_resolves_and_rejects_clip_names() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(
            &dir,
            "hero.glb",
            &crate::import::glb::test_fixtures::skinned_glb(),
        );
        let mut assets = vec![wja(
            "wave",
            "Animation",
            serde_json::json!({"source": src, "animation_name": "wave"}),
        )];
        desugar_animation_imports(&mut assets, None).expect("desugar");
        assert_eq!(assets[0].args["tracks"].as_array().unwrap().len(), 1);

        let mut missing = vec![wja(
            "run",
            "Animation",
            serde_json::json!({"source": src, "animation_name": "sprint"}),
        )];
        let err = desugar_animation_imports(&mut missing, None)
            .expect_err("the file has no 'sprint' clip");
        let msg = err.to_string();
        assert!(
            msg.contains("has no animation named 'sprint'"),
            "got: {msg}"
        );
        assert!(msg.contains("wave"), "the error lists the clips: {msg}");
    }

    #[test]
    fn desugar_root_motion_rejects_malformed_args() {
        let mut assets = vec![wja(
            "walk",
            "Animation",
            serde_json::json!({"root_motion": true, "duration": "long"}),
        )];
        let err = desugar_root_motion(&mut assets).expect_err("bad duration");
        assert!(
            err.to_string()
                .contains("root-motion bake failed to parse args"),
            "got: {err}"
        );
    }

    #[test]
    fn desugar_root_motion_tolerates_a_clip_with_no_root_track() {
        // Only joint 1 is animated: there is nothing to strip from the root,
        // so the bake warns and leaves an empty curve rather than failing.
        let mut assets = vec![wja(
            "wave",
            "Animation",
            serde_json::json!({
                "root_motion": true,
                "duration": 1.0,
                "tracks": [{"joint": 1, "keyframes": [
                    {"time": 0.0, "translation": [1.0, 0.0, 0.0]}
                ]}],
            }),
        )];
        desugar_root_motion(&mut assets).expect("bake succeeds");
        assert_eq!(assets[0].args["root_track"], serde_json::json!([]));
        // The non-root track is untouched.
        assert_eq!(
            assets[0].args["tracks"][0]["keyframes"][0]["translation"][0],
            1.0
        );
    }

    // Opting into vertical root motion keeps the Y travel in the root track
    // instead of anchoring it back into the pose.
    #[test]
    fn desugar_root_motion_keeps_y_travel_when_asked() {
        let clip = |root_motion_y: bool| {
            serde_json::json!({
                "target": "hero",
                "duration": 1.0,
                "root_motion": true,
                "root_motion_y": root_motion_y,
                "tracks": [{"joint": 0, "keyframes": [
                    {"time": 0.0, "translation": [0.0, 0.0, 0.0]},
                    {"time": 1.0, "translation": [0.0, 3.0, 0.0]}
                ]}],
            })
        };
        let mut assets = vec![
            wja("jump", "Animation", clip(true)),
            wja("walk", "Animation", clip(false)),
        ];
        desugar_root_motion(&mut assets).expect("bake succeeds");

        assert_eq!(assets[0].args["root_track"][1]["translation"][1], 3.0);
        assert_eq!(
            assets[0].args["tracks"][0]["keyframes"][1]["translation"][1],
            0.0
        );
        // Without the flag the rise stays in the pose and the root track is flat.
        assert_eq!(assets[1].args["root_track"][1]["translation"][1], 0.0);
        assert_eq!(
            assets[1].args["tracks"][0]["keyframes"][1]["translation"][1],
            3.0
        );
    }
}
