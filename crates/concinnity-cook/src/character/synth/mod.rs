// Synthesized morph targets: pure generators over a source mesh's bind-pose
// geometry, skin weights, and bone frames, one per file. Each returns a
// position delta per vertex; `finish` turns that into dense `MorphDelta`s
// with normals recomputed from the displaced mesh, so at runtime a
// synthesized target is indistinguishable from a sculpted one.

pub(crate) mod blend_mask;
pub(crate) mod bulge;
pub(crate) mod girth;
pub(crate) mod mirror;
pub(crate) mod surface_offset;
pub(crate) mod taper;

use super::frame::BoneFrame;
use crate::components::{KeyPolarity, MorphDelta, SkinnedVertexData, SynthParams};
use concinnity_core::math::vec3;

// What every generator reads.
pub(crate) struct SynthInput<'a> {
    pub vertices: &'a [SkinnedVertexData],
    pub indices: &'a [u16],
    // Smooth bind-pose normals, one per vertex.
    pub normals: &'a [[f32; 3]],
    pub frames: &'a [BoneFrame],
    // Per-joint membership in the target's region.
    pub members: &'a [bool],
    // The region's first listed joint: the bone `surface_offset` windows on.
    pub primary: usize,
    pub params: &'a SynthParams,
}

impl SynthInput<'_> {
    // Each influence (joint, normalised weight) of `v` that is in the region.
    pub(crate) fn region_influences(&self, v: &SkinnedVertexData) -> Vec<(usize, f32)> {
        let sum: f32 = v.weights.iter().sum();
        if sum <= 1e-6 {
            return Vec::new();
        }
        (0..4)
            .map(|k| (v.joints[k] as usize, v.weights[k] / sum))
            .filter(|(j, w)| *w > 0.0 && self.members.get(*j).copied().unwrap_or(false))
            .collect()
    }
}

// Smooth vertex normals from positions and triangles, accumulated the way
// the payload compile does so a zero displacement gives a zero normal delta.
pub(crate) fn vertex_normals(positions: &[[f32; 3]], indices: &[u16]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0_f32; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if a >= positions.len() || b >= positions.len() || c >= positions.len() {
            continue;
        }
        let n = vec3::vec3_face_normal(positions[a], positions[b], positions[c]);
        vec3::vec3_add(&mut normals[a], n);
        vec3::vec3_add(&mut normals[b], n);
        vec3::vec3_add(&mut normals[c], n);
    }
    normals.iter().map(|n| vec3::vec3_normalise(*n)).collect()
}

// Dense deltas for a displacement: the positions as given, the normals as
// the difference between the displaced mesh's normals and `base_normals`.
pub(crate) fn finish(
    vertices: &[SkinnedVertexData],
    indices: &[u16],
    base_normals: &[[f32; 3]],
    displacement: &[[f32; 3]],
) -> Vec<MorphDelta> {
    let moved: Vec<[f32; 3]> = vertices
        .iter()
        .zip(displacement)
        .map(|(v, d)| vec3::add(v.pos, *d))
        .collect();
    let normals = vertex_normals(&moved, indices);
    displacement
        .iter()
        .zip(normals.iter().zip(base_normals))
        .map(|(d, (n, b))| MorphDelta {
            position: *d,
            normal: vec3::sub(*n, *b),
        })
        .collect()
}

// The targets a displacement yields under a polarity: the displacement
// itself as `name` (or `name+`), and its negation as `name-`.
pub(crate) fn polarised(
    name: &str,
    polarity: KeyPolarity,
    input: &SynthInput,
    displacement: Vec<[f32; 3]>,
) -> Vec<(String, Vec<MorphDelta>)> {
    let plus = finish(input.vertices, input.indices, input.normals, &displacement);
    match polarity {
        KeyPolarity::Unipolar => vec![(name.to_string(), plus)],
        KeyPolarity::Bipolar => {
            let negated: Vec<[f32; 3]> =
                displacement.iter().map(|d| vec3::scale(*d, -1.0)).collect();
            let minus = finish(input.vertices, input.indices, input.normals, &negated);
            vec![(format!("{name}+"), plus), (format!("{name}-"), minus)]
        }
    }
}

// Unit vector, or `None` for a zero vector.
pub(crate) fn unit(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = vec3::length(v);
    (len > 1e-6).then(|| vec3::scale(v, 1.0 / len))
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::character::frame::bone_frames;
    use crate::components::SkeletonJoint;

    pub(crate) fn joint(name: &str, parent: i32, translation: [f32; 3]) -> SkeletonJoint {
        SkeletonJoint {
            name: name.to_string(),
            parent,
            translation,
            ..Default::default()
        }
    }

    pub(crate) fn vertex(pos: [f32; 3], joint: u32) -> SkinnedVertexData {
        SkinnedVertexData {
            pos,
            color: [1.0; 3],
            uv: [0.0; 2],
            joints: [joint, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }

    // A closed cylinder of `radius` along +Y from 0 to `height`, `rings + 1`
    // rings of `segs` vertices plus two pole vertices, bound to joint 0.
    pub(crate) fn cylinder(
        segs: usize,
        rings: usize,
        radius: f32,
        height: f32,
    ) -> (Vec<SkinnedVertexData>, Vec<u16>, Vec<SkeletonJoint>) {
        let mut verts = Vec::new();
        for r in 0..=rings {
            let y = height * r as f32 / rings as f32;
            for s in 0..segs {
                let a = std::f32::consts::TAU * s as f32 / segs as f32;
                verts.push(vertex([radius * a.cos(), y, radius * a.sin()], 0));
            }
        }
        let bottom = verts.len() as u16;
        verts.push(vertex([0.0, 0.0, 0.0], 0));
        let top = verts.len() as u16;
        verts.push(vertex([0.0, height, 0.0], 0));
        let mut idx = Vec::new();
        for r in 0..rings {
            for s in 0..segs {
                let a = (r * segs + s) as u16;
                let b = (r * segs + (s + 1) % segs) as u16;
                let c = ((r + 1) * segs + (s + 1) % segs) as u16;
                let d = ((r + 1) * segs + s) as u16;
                idx.extend_from_slice(&[a, c, b, a, d, c]);
            }
        }
        for s in 0..segs {
            let a = s as u16;
            let b = ((s + 1) % segs) as u16;
            idx.extend_from_slice(&[bottom, a, b]);
            let a = (rings * segs + s) as u16;
            let b = (rings * segs + (s + 1) % segs) as u16;
            idx.extend_from_slice(&[top, b, a]);
        }
        let skeleton = vec![joint("bone", -1, [0.0, 0.0, 0.0])];
        (verts, idx, skeleton)
    }

    // A unit UV sphere centred at the origin bound to joint 0.
    pub(crate) fn sphere(segs: usize, rings: usize) -> (Vec<SkinnedVertexData>, Vec<u16>) {
        let mut verts = Vec::new();
        for r in 0..=rings {
            let phi = std::f32::consts::PI * r as f32 / rings as f32;
            for s in 0..segs {
                let th = std::f32::consts::TAU * s as f32 / segs as f32;
                verts.push(vertex(
                    [phi.sin() * th.cos(), phi.cos(), phi.sin() * th.sin()],
                    0,
                ));
            }
        }
        let mut idx = Vec::new();
        for r in 0..rings {
            for s in 0..segs {
                let a = (r * segs + s) as u16;
                let b = (r * segs + (s + 1) % segs) as u16;
                let c = ((r + 1) * segs + (s + 1) % segs) as u16;
                let d = ((r + 1) * segs + s) as u16;
                idx.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
        (verts, idx)
    }

    // Owned frame + normal data so a test can build a `SynthInput`.
    pub(crate) struct Fixture {
        pub vertices: Vec<SkinnedVertexData>,
        pub indices: Vec<u16>,
        pub normals: Vec<[f32; 3]>,
        pub frames: Vec<BoneFrame>,
        pub members: Vec<bool>,
        pub params: SynthParams,
    }

    impl Fixture {
        pub(crate) fn new(
            vertices: Vec<SkinnedVertexData>,
            indices: Vec<u16>,
            skeleton: &[SkeletonJoint],
        ) -> Self {
            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.pos).collect();
            let normals = vertex_normals(&positions, &indices);
            let frames = bone_frames(skeleton, &vertices);
            Self {
                vertices,
                indices,
                normals,
                frames,
                members: vec![true; skeleton.len()],
                params: SynthParams::default(),
            }
        }

        pub(crate) fn input(&self) -> SynthInput<'_> {
            SynthInput {
                vertices: &self.vertices,
                indices: &self.indices,
                normals: &self.normals,
                frames: &self.frames,
                members: &self.members,
                primary: self.members.iter().position(|m| *m).unwrap_or(0),
                params: &self.params,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn a_zero_displacement_leaves_normals_alone() {
        let (verts, idx, sk) = cylinder(8, 2, 1.0, 2.0);
        let fx = Fixture::new(verts, idx, &sk);
        let zero = vec![[0.0; 3]; fx.vertices.len()];
        let out = finish(&fx.vertices, &fx.indices, &fx.normals, &zero);
        assert!(
            out.iter()
                .all(|d| d.position == [0.0; 3] && d.normal == [0.0; 3])
        );
        // Normals point out: radially on the middle ring, down at the base.
        let side = fx.normals[8 + 1];
        assert!((side[1]).abs() < 1e-3 && side[0] > 0.0, "{side:?}");
        let bottom = fx.normals[fx.vertices.len() - 2];
        assert!(bottom[1] < -0.99, "{bottom:?}");
    }

    #[test]
    fn bipolar_targets_negate_the_displacement_and_recompute_normals() {
        let (verts, idx, sk) = cylinder(8, 2, 1.0, 2.0);
        let fx = Fixture::new(verts, idx, &sk);
        // Flatten the top pole downward.
        let mut disp = vec![[0.0; 3]; fx.vertices.len()];
        let top = fx.vertices.len() - 1;
        disp[top] = [0.0, -0.5, 0.0];
        let out = polarised("cap", KeyPolarity::Bipolar, &fx.input(), disp.clone());
        assert_eq!(out[0].0, "cap+");
        assert_eq!(out[1].0, "cap-");
        assert_eq!(out[0].1[top].position, [0.0, -0.5, 0.0]);
        assert_eq!(out[1].1[top].position, [0.0, 0.5, 0.0]);
        // Pushing the pole in tilts the top ring's normals inward; pulling
        // it out tilts them outward: the two normal deltas have opposite
        // radial sign rather than being mirror copies.
        let ring = 2 * 8;
        let n_plus = out[0].1[ring].normal;
        let n_minus = out[1].1[ring].normal;
        assert!(n_plus[0] < 0.0, "{n_plus:?}");
        assert!(n_minus[0] > 0.0, "{n_minus:?}");
        let uni = polarised("cap", KeyPolarity::Unipolar, &fx.input(), disp);
        assert_eq!(uni.len(), 1);
        assert_eq!(uni[0].0, "cap");
        assert!(unit([0.0; 3]).is_none());
        assert_eq!(unit([0.0, 3.0, 0.0]), Some([0.0, 1.0, 0.0]));
    }
}
