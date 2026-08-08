// src/assets/animation.rs

use alloc::string::String;
use alloc::vec::Vec;

use crate::ecs::asset_id::AssetId;
use crate::ecs::{SkinnedMeshHandle, de_opt_skinned_mesh_handle};
use crate::gfx::skeleton::{self as skinning, JointPose};

/// One keyframe in an animation track: a joint pose sampled at `time` seconds.
/// The pose fields (`translation`, `rotation_deg`, `scale`) are given directly
/// on the keyframe, each defaulting to the identity transform when omitted.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Time of this keyframe in seconds from the clip start.
    pub time: f32,
    /// The joint's transform at this keyframe.
    pub pose: JointPose,
}

// The authored JSON shape flattens the pose onto the keyframe object
// (`{"time":0,"translation":[..]}`), but `serde(flatten)` needs a
// self-describing format, which the baked postcard form is not. Serde impls
// branch on the format: human-readable keeps the flattened schema, binary
// nests the pose as a plain field.
#[derive(serde::Serialize, serde::Deserialize)]
struct KeyframeFlat {
    time: f32,
    #[serde(flatten)]
    pose: JointPose,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct KeyframePlain {
    time: f32,
    pose: JointPose,
}

impl serde::Serialize for Keyframe {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            KeyframeFlat {
                time: self.time,
                pose: self.pose,
            }
            .serialize(s)
        } else {
            KeyframePlain {
                time: self.time,
                pose: self.pose,
            }
            .serialize(s)
        }
    }
}

impl<'de> serde::Deserialize<'de> for Keyframe {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            let k = KeyframeFlat::deserialize(d)?;
            Ok(Self {
                time: k.time,
                pose: k.pose,
            })
        } else {
            let k = KeyframePlain::deserialize(d)?;
            Ok(Self {
                time: k.time,
                pose: k.pose,
            })
        }
    }
}

/// An animation channel: a time-ordered list of keyframes for one joint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnimationTrack {
    /// Index of the joint in the target skeleton this track drives.
    pub joint: usize,
    /// Keyframes, expected in ascending time order.
    pub keyframes: Vec<Keyframe>,
}

/// A skeletal animation clip that animates one [SkinnedMesh](#skinnedmesh).
///
/// The clip plays every frame, sampling each track and deforming the target
/// mesh's skeleton. Joints with no track hold their bind pose.
///
/// Several `Animation` assets may target the same [SkinnedMesh](#skinnedmesh);
/// they are then blended into one pose, weighted by each clip's `weight` (a
/// normalised weighted average). A single clip plays at full strength
/// regardless of its `weight`.
///
/// **File import.** A clip may be authored entirely by hand (`tracks` filled
/// out, `source` left empty) or imported from the same glTF (`.glb` /
/// `.gltf`) or `.fbx` file that backs the target [SkinnedMesh](#skinnedmesh).
/// Set `source` to the file path and the build imports `duration` + `tracks`
/// from it. `animation_index` picks one clip when the file contains several
/// (default 0); `animation_name` names it for matching against the file's
/// clip names: when set it takes precedence over the index. FBX curves are
/// baked at `sample_rate` keys per second. Channels whose target node is not
/// a joint of the file's first skinned node are dropped. The same file should
/// back the target [SkinnedMesh](#skinnedmesh) so the joint indices agree.
///
/// ```jsonl
/// // Inline:
/// {"name":"flag_wave","type":"Animation","args":{"target":"flag","duration":2.0,"tracks":[{"joint":1,"keyframes":[{"time":0.0,"rotation_deg":[0,0,0]},{"time":1.0,"rotation_deg":[0,30,0]},{"time":2.0,"rotation_deg":[0,0,0]}]}]}}
/// // From glTF:
/// {"name":"hero_walk","type":"Animation","args":{"target":"hero","source":"models/hero.glb","animation_name":"Walk","looping":true}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Animation {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [SkinnedMesh](#skinnedmesh) asset this clip animates.
    #[serde(deserialize_with = "de_opt_skinned_mesh_handle")]
    pub target: Option<SkinnedMeshHandle>,
    /// Optional path to a `.glb`, `.gltf`, or `.fbx` file. When set, the
    /// build imports `duration` + `tracks` from it; inline-authored clips
    /// leave this empty.
    pub source: String,
    /// Index of the animation to import when `source` is set and the file
    /// contains several. Ignored when `animation_name` is non-empty.
    pub animation_index: u32,
    /// Name of the animation to import. When set, the matching clip in the
    /// source file is looked up by name; takes precedence over
    /// `animation_index`.
    pub animation_name: String,
    /// Keys per second baked from sources whose curves need resampling at
    /// import (FBX). glTF keyframes pass through untouched. Default 30.
    pub sample_rate: f32,
    /// Clip length in seconds. Overridden by glTF import.
    pub duration: f32,
    /// When true, playback wraps after `duration`.
    pub looping: bool,
    /// Blend weight used when several clips target the same
    /// [SkinnedMesh](#skinnedmesh). Ignored when this is the only clip on its
    /// target.
    pub weight: f32,
    /// When non-zero, the clip's contribution ramps from 0 to its declared
    /// `weight` over this many seconds after the world starts. Zero (the
    /// default) plays the clip at full `weight` from the first frame.
    pub fade_in_secs: f32,
    /// When true, the build strips the root joint's travel out of the pose
    /// and bakes it into `root_track`: the pose stays anchored in place and
    /// the runtime moves the character by the curve's frame-to-frame delta
    /// instead (the [SkinnedMesh](#skinnedmesh) `capsule` is the usual
    /// consumer). X and Z travel is always stripped; Y only with
    /// `root_motion_y`.
    pub root_motion: bool,
    /// Also strip the root joint's vertical travel into `root_track`. Leave
    /// false (the default) so jumps and crouches stay authored in the pose.
    pub root_motion_y: bool,
    /// The displacement curve baked out of the root joint by the build when
    /// `root_motion` is set. Filled by the build; not usually authored by
    /// hand.
    pub root_track: Vec<crate::gfx::root_motion::RootKey>,
    /// Per-joint keyframe channels.
    pub tracks: Vec<AnimationTrack>,
    /// Morph-target weight keys for the target mesh, in time order. Each key
    /// holds one weight per morph target of the [SkinnedMesh](#skinnedmesh).
    /// Filled by the glTF import; empty when the clip animates no morph
    /// targets.
    pub morph_track: Vec<MorphKey>,
}

/// One morph-weight keyframe of an [Animation](#animation): per-target
/// weights at one sample time.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MorphKey {
    /// Sample time in seconds from clip start.
    pub time: f32,
    /// One weight per morph target, in target order.
    pub weights: Vec<f32>,
}

impl Default for Animation {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            target: None,
            source: String::new(),
            animation_index: 0,
            animation_name: String::new(),
            sample_rate: 30.0,
            duration: 1.0,
            looping: true,
            weight: 1.0,
            fade_in_secs: 0.0,
            root_motion: false,
            root_motion_y: false,
            root_track: Vec::new(),
            tracks: Vec::new(),
            morph_track: Vec::new(),
        }
    }
}

impl Animation {
    /// Convert this asset into the runtime `AnimationClip` consumed by the
    /// skinning math.
    pub fn to_clip(&self) -> skinning::AnimationClip {
        skinning::AnimationClip {
            duration: self.duration.max(1e-3),
            looping: self.looping,
            tracks: self
                .tracks
                .iter()
                .map(|t| skinning::JointTrack {
                    joint: t.joint,
                    keys: t
                        .keyframes
                        .iter()
                        .map(|k| skinning::Keyframe {
                            time: k.time,
                            pose: k.pose,
                        })
                        .collect(),
                })
                .collect(),
            morph_keys: self
                .morph_track
                .iter()
                .map(|k| (k.time, k.weights.clone()))
                .collect(),
            root: (!self.root_track.is_empty()).then(|| crate::gfx::root_motion::RootTrack {
                keys: self.root_track.clone(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_with_defaults() {
        let a: Animation = serde_json::from_str("{}").unwrap();
        assert_eq!(a.duration, 1.0);
        assert!(a.looping);
        assert_eq!(a.weight, 1.0);
        assert!(a.tracks.is_empty());
        assert_eq!(a.source, "");
        assert_eq!(a.animation_index, 0);
        assert_eq!(a.animation_name, "");
    }

    #[test]
    fn deserialises_glb_source_fields() {
        crate::test_support::reset_interner();
        let json = r#"{
            "target":"hero",
            "source":"models/hero.glb",
            "animation_index":2,
            "animation_name":"Walk",
            "looping":false
        }"#;
        let a: Animation = serde_json::from_str(json).unwrap();
        assert_eq!(a.source, "models/hero.glb");
        assert_eq!(a.animation_index, 2);
        assert_eq!(a.animation_name, "Walk");
        assert!(!a.looping);
    }

    #[test]
    fn deserialises_inline_tracks() {
        crate::test_support::reset_interner();
        let json = r#"{
            "target":"flag",
            "duration":2.0,
            "tracks":[{"joint":0,"keyframes":[{"time":0.0,"rotation_deg":[0,30,0]}]}]
        }"#;
        let a: Animation = serde_json::from_str(json).unwrap();
        assert_eq!(a.duration, 2.0);
        assert_eq!(a.tracks.len(), 1);
        assert_eq!(a.tracks[0].joint, 0);
    }

    #[test]
    fn to_clip_floors_duration_so_runtime_loop_does_not_divide_by_zero() {
        let a = Animation {
            duration: 0.0,
            ..Default::default()
        };
        let clip = a.to_clip();
        assert!(clip.duration >= 1e-3);
    }
}
