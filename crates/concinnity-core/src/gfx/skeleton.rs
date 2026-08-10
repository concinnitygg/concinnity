// src/gfx/skeleton.rs
//
// The skeletal-animation vocabulary: a joint hierarchy with its bind pose, the
// keyframe tracks a clip animates it with, and the sampling that turns a clip
// time into one local matrix per joint.
//
// Rotations are stored as YXZ Euler degrees (matching `Prop.rotation_deg`).
// Between keyframes, translation and scale interpolate linearly while rotation
// is converted to a quaternion and slerped (shortest-arc, constant angular
// velocity), so multi-axis joint rotation follows the correct path rather than
// the skewed one a component-wise Euler lerp would take.

use alloc::string::String;
use alloc::vec::Vec;

use crate::gfx::render_types::MAX_JOINTS;
use crate::gfx::root_motion::RootTrack;
use crate::gfx::transform::{
    IDENTITY, Mat4, compose, mat4_affine_inverse, mat4_mul, quat_from_mat3, quat_slerp,
    quat_to_mat3, rotation_mat3, trs_matrix,
};
use crate::math::rem_euclid;

/// A joint's local transform: translation, YXZ Euler rotation in degrees, and
/// per-axis scale. Used both for the bind pose and for animation keyframes.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct JointPose {
    pub translation: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],
}

impl Default for JointPose {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl JointPose {
    /// Column-major local matrix `T * R(YXZ) * S`.
    pub fn to_matrix(&self) -> Mat4 {
        trs_matrix(self.translation, self.rotation_deg, self.scale)
    }

    /// Interpolate two poses into a column-major local matrix. Translation and
    /// scale blend linearly; rotation is quaternion-slerped (shortest-arc,
    /// constant angular velocity) rather than Euler-lerped, so multi-axis
    /// joint rotation follows the correct path. `f` in `[0, 1]`.
    ///
    /// Slerps the poses' own Euler rotations rather than going through
    /// [`blend_matrices`], which would have to recover them from the composed
    /// matrices first.
    pub fn blend_matrix(&self, other: &JointPose, f: f32) -> Mat4 {
        let mix = |a: [f32; 3], b: [f32; 3]| {
            [
                a[0] + (b[0] - a[0]) * f,
                a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f,
            ]
        };
        let qa = quat_from_mat3(rotation_mat3(self.rotation_deg));
        let qb = quat_from_mat3(rotation_mat3(other.rotation_deg));
        let rotation = quat_to_mat3(quat_slerp(qa, qb, f));
        compose(
            rotation,
            mix(self.scale, other.scale),
            mix(self.translation, other.translation),
        )
    }
}

/// One joint in a skeleton: a parent link and a local bind transform.
#[derive(Debug, Clone)]
pub struct Joint {
    /// Authored joint name (empty when the source declared none). Resolved to
    /// an index at load time by consumers that reference joints by name
    /// (e.g. IK chains); never compared per frame.
    pub name: String,
    /// Index of the parent joint, or `None` for a root. Parents must appear
    /// before their children so a single forward pass resolves the hierarchy.
    pub parent: Option<usize>,
    /// Local bind-pose transform relative to the parent.
    pub bind: JointPose,
}

/// A joint hierarchy plus the bind pose. The inverse bind matrices are
/// precomputed once on construction.
#[derive(Debug, Clone)]
pub struct Skeleton {
    joints: Vec<Joint>,
    // Local bind matrix per joint, built once. Every pose sample starts from
    // these, so rebuilding them per frame would re-run the Euler trig for
    // every joint of every sampled clip.
    bind_locals: Vec<Mat4>,
    // World-space inverse bind matrix per joint.
    inverse_bind: Vec<Mat4>,
    // World-space bind position per joint. With a skinning matrix
    // `S = world * inverse_bind`, `S * bind_position` recovers the joint's
    // current mesh-space position without another hierarchy walk.
    bind_positions: Vec<[f32; 3]>,
}

impl Skeleton {
    /// Build a skeleton, resolving world bind matrices and inverting them.
    /// Joints referencing a parent that does not precede them are treated as
    /// roots (a forward pass cannot resolve them otherwise).
    pub fn new(joints: Vec<Joint>) -> Self {
        let bind_locals: Vec<Mat4> = joints.iter().map(|j| j.bind.to_matrix()).collect();
        let mut world_bind: Vec<Mat4> = Vec::with_capacity(joints.len());
        for (i, joint) in joints.iter().enumerate() {
            let local = bind_locals[i];
            let world = match joint.parent {
                Some(p) if p < i => mat4_mul(world_bind[p], local),
                _ => local,
            };
            world_bind.push(world);
        }
        let inverse_bind = world_bind.iter().map(|m| mat4_affine_inverse(*m)).collect();
        let bind_positions = world_bind
            .iter()
            .map(|m| [m[3][0], m[3][1], m[3][2]])
            .collect();
        Self {
            joints,
            bind_locals,
            inverse_bind,
            bind_positions,
        }
    }

    pub fn len(&self) -> usize {
        self.joints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    /// Index of the joint with the given authored name, or `None`. Load-time
    /// lookup for by-name joint references (IK chains); linear scan is fine.
    pub fn joint_index(&self, name: &str) -> Option<usize> {
        (!name.is_empty()).then(|| self.joints.iter().position(|j| j.name == name))?
    }

    /// World-space bind position of one joint.
    pub fn bind_position(&self, joint: usize) -> [f32; 3] {
        self.bind_positions.get(joint).copied().unwrap_or([0.0; 3])
    }

    /// Compose `local_poses` (one local matrix per joint) into mesh-space
    /// joint matrices with a single forward pass over the hierarchy, written
    /// into `out` (cleared first, so its capacity is reused). `local_poses`
    /// shorter than the skeleton has its missing tail filled from the bind
    /// pose.
    pub fn world_matrices_into(&self, local_poses: &[Mat4], out: &mut Vec<Mat4>) {
        out.clear();
        out.reserve(self.joints.len());
        for (i, joint) in self.joints.iter().enumerate() {
            let local = local_poses.get(i).copied().unwrap_or(self.bind_locals[i]);
            let world_mat = match joint.parent {
                Some(p) if p < i => mat4_mul(out[p], local),
                _ => local,
            };
            out.push(world_mat);
        }
    }

    /// Compose `local_poses` into world-space joint matrices, then multiply
    /// by the inverse bind matrices to produce the skinning matrices the
    /// vertex shader applies, written into `out` (cleared first, so its
    /// capacity is reused). `local_poses` must not alias `out`.
    ///
    /// The result is capped at `MAX_JOINTS` entries (the GPU joint buffer is
    /// fixed-size) and is always at least one matrix so the buffer is never
    /// empty.
    pub fn skinning_matrices_into(&self, local_poses: &[Mat4], out: &mut Vec<Mat4>) {
        self.world_matrices_into(local_poses, out);
        let n = out.len().min(self.inverse_bind.len()).min(MAX_JOINTS);
        for (i, ib) in self.inverse_bind[..n].iter().enumerate() {
            out[i] = mat4_mul(out[i], *ib);
        }
        out.truncate(n);
        if out.is_empty() {
            out.push(IDENTITY);
        }
    }

    /// Skinning matrices for the rest (bind) pose: every joint's local
    /// transform is its bind transform, so every skinning matrix is identity.
    /// Used to seed a `SkeletonPose` before the first animation tick.
    pub fn bind_skinning_matrices(&self) -> Vec<Mat4> {
        let mut out = Vec::new();
        self.skinning_matrices_into(&self.bind_locals, &mut out);
        out
    }

    /// The local bind matrix of every joint, in joint order. A pose sample
    /// seeds its output with these before applying the clip's tracks.
    pub fn bind_locals(&self) -> &[Mat4] {
        &self.bind_locals
    }
}

/// A single keyframe: a joint pose sampled at a point in time.
#[derive(Debug, Clone, Copy)]
pub struct Keyframe {
    pub time: f32,
    pub pose: JointPose,
}

/// An animation channel for one joint: a time-ordered list of keyframes.
#[derive(Debug, Clone)]
pub struct JointTrack {
    pub joint: usize,
    pub keys: Vec<Keyframe>,
}

impl JointTrack {
    // Sample this track at time `t` (seconds), returning the joint's local
    // matrix. Times outside the keyframe range clamp to the nearest end key;
    // between keys translation/scale lerp and rotation slerps.
    fn sample(&self, t: f32) -> Mat4 {
        match self.keys.as_slice() {
            [] => IDENTITY,
            [only] => only.pose.to_matrix(),
            keys => {
                if t <= keys[0].time {
                    return keys[0].pose.to_matrix();
                }
                let last = keys[keys.len() - 1];
                if t >= last.time {
                    return last.pose.to_matrix();
                }
                // Keys are time-ordered; imported clips are baked at the
                // sample rate, so tracks can carry dozens of keys.
                let i = keys.partition_point(|k| k.time < t);
                let (a, b) = (keys[i - 1], keys[i]);
                let span = (b.time - a.time).max(1e-6);
                let f = (t - a.time) / span;
                a.pose.blend_matrix(&b.pose, f)
            }
        }
    }
}

/// One animation clip: a fixed-length set of per-joint keyframe tracks.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    /// Total clip length in seconds.
    pub duration: f32,
    /// When true, sampling past `duration` wraps; otherwise it holds the end.
    pub looping: bool,
    pub tracks: Vec<JointTrack>,
    /// Morph-target weight keys in time order: (time, one weight per target).
    /// Empty for clips that animate no morph targets.
    pub morph_keys: Vec<(f32, Vec<f32>)>,
    /// The character-displacement curve stripped from the root joint at build
    /// time, when the clip opted into root motion. The pose tracks above keep
    /// the root anchored; the runtime turns this curve's frame delta into
    /// character movement instead.
    pub root: Option<RootTrack>,
}

impl AnimationClip {
    /// Sample the clip at time `t` against `skeleton`, writing one local
    /// matrix per joint into `out` (cleared first, so its capacity is
    /// reused). Joints with no track keep their bind transform.
    pub fn sample_into(&self, t: f32, skeleton: &Skeleton, out: &mut Vec<Mat4>) {
        self.sample_looped_into(t, self.looping, skeleton, out)
    }

    /// `sample_into` with the loop mode supplied by the caller instead of the
    /// clip's own flag. Lets a graph state override looping without cloning
    /// the clip.
    pub fn sample_looped_into(
        &self,
        t: f32,
        looping: bool,
        skeleton: &Skeleton,
        out: &mut Vec<Mat4>,
    ) {
        let local_t = self.clip_time(t, looping);
        out.clear();
        out.extend_from_slice(skeleton.bind_locals());
        for track in &self.tracks {
            if track.joint < out.len() {
                out[track.joint] = track.sample(local_t);
            }
        }
    }

    /// Sample the morph-weight track at time `t` into `out` (cleared first,
    /// so its capacity is reused), lerping between the surrounding keys with
    /// the same wrap/clamp semantics as pose sampling. `out` is left empty
    /// when the clip has no morph keys.
    pub fn sample_morph_weights_into(&self, t: f32, looping: bool, out: &mut Vec<f32>) {
        out.clear();
        if self.morph_keys.is_empty() {
            return;
        }
        let local_t = self.clip_time(t, looping);
        let first = &self.morph_keys[0];
        if local_t <= first.0 {
            out.extend_from_slice(&first.1);
            return;
        }
        for pair in self.morph_keys.windows(2) {
            if local_t <= pair[1].0 {
                let span = (pair[1].0 - pair[0].0).max(1e-6);
                let f = (local_t - pair[0].0) / span;
                let n = pair[0].1.len().max(pair[1].1.len());
                out.extend((0..n).map(|i| {
                    let a = pair[0].1.get(i).copied().unwrap_or(0.0);
                    let b = pair[1].1.get(i).copied().unwrap_or(0.0);
                    a + (b - a) * f
                }));
                return;
            }
        }
        out.extend_from_slice(&self.morph_keys[self.morph_keys.len() - 1].1);
    }

    // Clip-local time for a wall-clock `t`: wrapped when looping, otherwise
    // clamped into the clip's range.
    fn clip_time(&self, t: f32, looping: bool) -> f32 {
        if looping && self.duration > 1e-6 {
            rem_euclid(t, self.duration)
        } else {
            t.clamp(0.0, self.duration)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::transform::blend_matrices;
    use crate::math::atan2;
    use alloc::vec;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // A two-joint vertical chain: root at origin, child one unit up in y.
    fn chain() -> Skeleton {
        Skeleton::new(vec![
            Joint {
                name: String::new(),
                parent: None,
                bind: JointPose::default(),
            },
            Joint {
                name: String::new(),
                parent: Some(0),
                bind: JointPose {
                    translation: [0.0, 1.0, 0.0],
                    ..JointPose::default()
                },
            },
        ])
    }

    #[test]
    fn morph_weight_sampling_lerps_clamps_and_loops() {
        let clip = AnimationClip {
            duration: 1.0,
            looping: false,
            tracks: Vec::new(),
            morph_keys: vec![(0.0, vec![0.0, 1.0]), (1.0, vec![1.0, 0.0])],
            root: None,
        };
        let morph = |t: f32, looping: bool| {
            let mut out = Vec::new();
            clip.sample_morph_weights_into(t, looping, &mut out);
            out
        };
        assert!(morph(-1.0, false)[0].abs() < 1e-6);
        let mid = morph(0.5, false);
        assert!(approx(mid[0], 0.5) && approx(mid[1], 0.5));
        assert!(approx(morph(5.0, false)[0], 1.0), "clamps past the end");
        // Looping wraps: t = 1.25 samples like t = 0.25.
        let wrapped = morph(1.25, true);
        assert!(approx(wrapped[0], 0.25));

        let empty = AnimationClip {
            duration: 1.0,
            looping: true,
            tracks: Vec::new(),
            morph_keys: Vec::new(),
            root: None,
        };
        let mut out = vec![9.0];
        empty.sample_morph_weights_into(0.5, true, &mut out);
        assert!(out.is_empty(), "no morph keys clears the output");
    }

    #[test]
    fn bind_pose_skinning_matrices_are_identity() {
        let sk = chain();
        for m in sk.bind_skinning_matrices() {
            for col in 0..4 {
                for row in 0..4 {
                    assert!(approx(m[col][row], IDENTITY[col][row]));
                }
            }
        }
    }

    #[test]
    fn rotating_child_joint_moves_a_bound_point() {
        // Rotate the child joint 90 deg yaw. A point at the child's origin in
        // bind space (0,1,0) should be carried by the child's skinning matrix
        // but the joint origin itself is the rotation pivot, so it stays put.
        // A point offset +x from the child should swing to -z.
        let sk = chain();
        let mut locals: Vec<Mat4> = sk.joints().iter().map(|j| j.bind.to_matrix()).collect();
        locals[1] = JointPose {
            translation: [0.0, 1.0, 0.0],
            rotation_deg: [0.0, 90.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
        .to_matrix();
        let mut skin = Vec::new();
        sk.skinning_matrices_into(&locals, &mut skin);
        // Bind-space point one unit +x of the child joint origin: (1, 1, 0).
        let p = [1.0f32, 1.0, 0.0, 1.0];
        let m = skin[1];
        let out = [
            m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0] * p[3],
            m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1] * p[3],
            m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2] * p[3],
        ];
        // +x swings to -z under a +90 deg yaw; y unchanged.
        assert!(approx(out[0], 0.0), "x was {}", out[0]);
        assert!(approx(out[1], 1.0), "y was {}", out[1]);
        assert!(approx(out[2], -1.0), "z was {}", out[2]);
    }

    #[test]
    fn clip_sampling_interpolates_between_keys() {
        let sk = chain();
        let clip = AnimationClip {
            root: None,
            duration: 2.0,
            looping: true,
            tracks: vec![JointTrack {
                joint: 1,
                keys: vec![
                    Keyframe {
                        time: 0.0,
                        pose: JointPose {
                            translation: [0.0, 1.0, 0.0],
                            ..JointPose::default()
                        },
                    },
                    Keyframe {
                        time: 2.0,
                        pose: JointPose {
                            translation: [0.0, 1.0, 0.0],
                            rotation_deg: [0.0, 90.0, 0.0],
                            ..JointPose::default()
                        },
                    },
                ],
            }],
            morph_keys: Vec::new(),
        };
        // Halfway through: yaw should be 45 deg.
        let mut locals = Vec::new();
        clip.sample_into(1.0, &sk, &mut locals);
        // Recover yaw: for a pure yaw the first column is (cos, 0, -sin).
        let yaw = atan2(-locals[1][0][2], locals[1][0][0]).to_degrees();
        assert!(approx(yaw, 45.0), "yaw was {}", yaw);
    }

    #[test]
    fn many_key_track_samples_the_containing_segment() {
        // A densely baked track (like importer output): keys every 0.1s with
        // translation.x following the key time, so any sample time recovers
        // itself. Covers end clamps, exact key hits, and mid-segment lerps.
        let keys: Vec<Keyframe> = (0..=20)
            .map(|i| {
                let time = i as f32 * 0.1;
                Keyframe {
                    time,
                    pose: JointPose {
                        translation: [time, 0.0, 0.0],
                        ..JointPose::default()
                    },
                }
            })
            .collect();
        let track = JointTrack { joint: 0, keys };
        let x_at = |t: f32| track.sample(t)[3][0];
        assert!(approx(x_at(-0.5), 0.0), "clamps at the first key");
        assert!(approx(x_at(5.0), 2.0), "clamps at the last key");
        assert!(approx(x_at(0.7), 0.7), "exact key hit");
        assert!(approx(x_at(1.234), 1.234), "lerps inside a segment");
    }

    #[test]
    fn looping_clip_wraps_past_duration() {
        let sk = chain();
        let clip = AnimationClip {
            root: None,
            duration: 2.0,
            looping: true,
            tracks: vec![JointTrack {
                joint: 1,
                keys: vec![Keyframe {
                    time: 0.5,
                    pose: JointPose {
                        translation: [9.0, 1.0, 0.0],
                        ..JointPose::default()
                    },
                }],
            }],
            morph_keys: Vec::new(),
        };
        // t = 2.5 wraps to 0.5: identical sample, into reused capacity.
        let mut a = Vec::new();
        clip.sample_into(0.5, &sk, &mut a);
        let mut b = Vec::new();
        clip.sample_into(2.5, &sk, &mut b);
        assert_eq!(a[1], b[1]);
        // Resampling into a warm buffer does not reallocate it.
        let ptr = a.as_ptr();
        clip.sample_into(1.5, &sk, &mut a);
        assert_eq!(a.as_ptr(), ptr, "warm sample buffer is reused in place");
    }

    #[test]
    fn unparented_joint_is_treated_as_root() {
        // A joint whose parent index does not precede it must not panic and
        // must behave as a root.
        let sk = Skeleton::new(vec![Joint {
            name: String::new(),
            parent: Some(5),
            bind: JointPose::default(),
        }]);
        assert_eq!(sk.len(), 1);
        assert_eq!(sk.bind_skinning_matrices().len(), 1);
    }

    #[test]
    fn joint_index_resolves_names_and_refuses_the_empty_one() {
        let sk = Skeleton::new(vec![
            Joint {
                name: String::from("hips"),
                parent: None,
                bind: JointPose::default(),
            },
            Joint {
                name: String::new(),
                parent: Some(0),
                bind: JointPose::default(),
            },
        ]);
        assert_eq!(sk.joint_index("hips"), Some(0));
        assert_eq!(sk.joint_index("missing"), None);
        // An unnamed joint must not be reachable by the empty name.
        assert_eq!(sk.joint_index(""), None);
        assert!(!sk.is_empty());
    }

    #[test]
    fn blend_matrix_endpoints_match_keyframe_poses() {
        // At f=0 / f=1 the interpolated matrix must equal the keyframe pose's
        // own matrix, so a clip is continuous across keyframe boundaries.
        let a = JointPose {
            translation: [1.0, 2.0, 3.0],
            rotation_deg: [10.0, 20.0, 30.0],
            scale: [1.0, 1.5, 2.0],
        };
        let b = JointPose {
            translation: [-4.0, 0.0, 5.0],
            rotation_deg: [70.0, -40.0, 15.0],
            scale: [2.0, 1.0, 0.5],
        };
        let at0 = a.blend_matrix(&b, 0.0);
        let at1 = a.blend_matrix(&b, 1.0);
        let ma = a.to_matrix();
        let mb = b.to_matrix();
        for c in 0..4 {
            for row in 0..4 {
                assert!(approx(at0[c][row], ma[c][row]), "f=0 [{}][{}]", c, row);
                assert!(approx(at1[c][row], mb[c][row]), "f=1 [{}][{}]", c, row);
            }
        }
    }

    #[test]
    fn blend_matrix_lerps_translation_and_scale() {
        // Translation and scale stay linearly interpolated: only rotation
        // moved to the quaternion path.
        let a = JointPose {
            translation: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let b = JointPose {
            translation: [4.0, 8.0, -2.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [3.0, 3.0, 3.0],
        };
        let m = a.blend_matrix(&b, 0.25);
        assert!(approx(m[3][0], 1.0));
        assert!(approx(m[3][1], 2.0));
        assert!(approx(m[3][2], -0.5));
        // No rotation: the diagonal carries the lerped scale 1 + 0.25*2 = 1.5.
        assert!(approx(m[0][0], 1.5));
        assert!(approx(m[1][1], 1.5));
        assert!(approx(m[2][2], 1.5));
    }

    // The pose-space blend and the matrix-space one are the same operation
    // reached two ways, so they must not disagree where both apply.
    #[test]
    fn pose_blend_agrees_with_the_matrix_blend() {
        let a = JointPose {
            translation: [1.0, 2.0, 3.0],
            rotation_deg: [10.0, 20.0, 30.0],
            scale: [1.0, 1.5, 2.0],
        };
        let b = JointPose {
            translation: [-4.0, 0.0, 5.0],
            rotation_deg: [70.0, -40.0, 15.0],
            scale: [2.0, 1.0, 0.5],
        };
        for f in [0.0, 0.25, 0.5, 1.0] {
            let pose_space = a.blend_matrix(&b, f);
            let matrix_space = blend_matrices(a.to_matrix(), b.to_matrix(), f);
            for c in 0..4 {
                for row in 0..4 {
                    assert!(
                        approx(pose_space[c][row], matrix_space[c][row]),
                        "f={f} [{c}][{row}]"
                    );
                }
            }
        }
    }
}
