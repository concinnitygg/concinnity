//! Benchmarks over the CPU animation pipeline: clip sampling, weighted pose
//! blending, skinning-matrix resolution, the composed per-character cost, and
//! the two-bone IK solve. The skeleton is a 64-joint chain in the shape of a
//! humanoid rig; the realistic clip carries 61 keys per track (two seconds
//! baked at 30 Hz, what the FBX importer emits), and a 2-key variant isolates
//! the track scan cost from the interpolation itself. Sample times advance
//! every iteration so keyframe search never settles on one branch.
//!
//! Run with `cargo bench -p concinnity-bench --bench anim`.

use concinnity_bench::{Bench, Rng};
use concinnity_cpu::gfx::ik::{TwoBoneChain, apply_two_bone_ik};
use concinnity_cpu::gfx::skinning::{
    AnimationClip, Joint, JointPose, JointTrack, Keyframe, Mat4, PoseBlend, Skeleton,
};

const JOINTS: usize = 64;
const BAKED_KEYS: usize = 61;
const CLIP_SECONDS: f32 = 2.0;
const FRAME_DT: f32 = 1.0 / 60.0;

fn skeleton() -> Skeleton {
    let joints = (0..JOINTS)
        .map(|i| Joint {
            name: format!("joint{i}"),
            parent: (i > 0).then(|| i - 1),
            bind: JointPose {
                translation: [0.0, 0.1, 0.0],
                rotation_deg: [0.0, 0.0, if i % 4 == 0 { 5.0 } else { 0.0 }],
                scale: [1.0, 1.0, 1.0],
            },
        })
        .collect();
    Skeleton::new(joints)
}

// A clip with `keys` keyframes on every joint, deterministic joint motion in
// the ranges a walk cycle covers.
fn clip(keys: usize, seed: u64) -> AnimationClip {
    let mut rng = Rng::new(seed);
    let mut angle = move || rng.below(120) as f32 - 60.0;
    let tracks = (0..JOINTS)
        .map(|joint| JointTrack {
            joint,
            keys: (0..keys)
                .map(|k| Keyframe {
                    time: CLIP_SECONDS * k as f32 / (keys - 1).max(1) as f32,
                    pose: JointPose {
                        translation: [0.0, 0.1, 0.0],
                        rotation_deg: [angle() * 0.5, angle(), angle() * 0.25],
                        scale: [1.0, 1.0, 1.0],
                    },
                })
                .collect(),
        })
        .collect();
    AnimationClip {
        duration: CLIP_SECONDS,
        looping: true,
        tracks,
        morph_keys: Vec::new(),
        root: None,
    }
}

fn main() {
    let mut bench = Bench::from_env();

    let skeleton = skeleton();
    let baked = clip(BAKED_KEYS, 1);
    let sparse = clip(2, 2);
    let second = clip(BAKED_KEYS, 3);

    {
        let mut t = 0.0f32;
        let mut out: Vec<Mat4> = Vec::new();
        bench.run("anim/sample_clip_61key/64j", JOINTS as u64, || {
            t += FRAME_DT;
            baked.sample_looped_into(t, true, &skeleton, &mut out);
            out.len()
        });
    }

    {
        let mut t = 0.0f32;
        let mut out: Vec<Mat4> = Vec::new();
        bench.run("anim/sample_clip_2key/64j", JOINTS as u64, || {
            t += FRAME_DT;
            sparse.sample_looped_into(t, true, &skeleton, &mut out);
            out.len()
        });
    }

    {
        let mut pose_a: Vec<Mat4> = Vec::new();
        let mut pose_b: Vec<Mat4> = Vec::new();
        baked.sample_looped_into(0.3, true, &skeleton, &mut pose_a);
        second.sample_looped_into(0.7, true, &skeleton, &mut pose_b);
        let mut acc: Vec<Mat4> = Vec::new();
        bench.run("anim/blend_2_poses/64j", JOINTS as u64, || {
            let mut fold = PoseBlend::new(&mut acc);
            fold.add(&pose_a, 1.0);
            fold.add(&pose_b, 0.6);
            acc.len()
        });
    }

    {
        let mut locals: Vec<Mat4> = Vec::new();
        baked.sample_looped_into(0.5, true, &skeleton, &mut locals);
        let mut out: Vec<Mat4> = Vec::new();
        bench.run("anim/skin_matrices/64j", JOINTS as u64, || {
            skeleton.skinning_matrices_into(&locals, &mut out);
            out.len()
        });
    }

    // One animated character's frame: two clips sampled at unequal weights,
    // blended, and resolved to skinning matrices (the multi-clip arm of the
    // runtime path, matching the CPU stress world's skinned axis).
    {
        let mut t = 0.0f32;
        let mut acc: Vec<Mat4> = Vec::new();
        let mut sample: Vec<Mat4> = Vec::new();
        let mut out: Vec<Mat4> = Vec::new();
        bench.run("anim/character_pose_2clips/64j", 1, || {
            t += FRAME_DT;
            let mut fold = PoseBlend::new(&mut acc);
            baked.sample_looped_into(t, true, &skeleton, &mut sample);
            fold.add(&sample, 1.0);
            second.sample_looped_into(t * 0.9, true, &skeleton, &mut sample);
            fold.add(&sample, 0.6);
            skeleton.skinning_matrices_into(&acc, &mut out);
            out.len()
        });
    }

    {
        let chain = TwoBoneChain {
            root: 1,
            mid: 2,
            end: 3,
            pole: [0.0, 0.0, 1.0],
        };
        let mut t = 0.0f32;
        let mut pristine: Vec<Mat4> = Vec::new();
        baked.sample_looped_into(0.5, true, &skeleton, &mut pristine);
        let mut locals: Vec<Mat4> = Vec::new();
        let mut world: Vec<Mat4> = Vec::new();
        // Reset from a pristine pose each solve (a memcpy) rather than
        // resampling, so the number is the IK solve and not the clip sample.
        bench.run("anim/two_bone_ik/1", 1, || {
            t += FRAME_DT;
            locals.clone_from(&pristine);
            let target = [0.1, 0.25 + 0.05 * (t % 1.0), 0.05];
            apply_two_bone_ik(&skeleton, &mut locals, &chain, target, 1.0, &mut world);
            locals.len()
        });
    }

    bench.finish();
}
