// Runs a schema's synthesized-target list over one imported source, appending
// the generated targets to its morph set. Targets run in schema order, so a
// `mirror` or `blend_mask` may name a target synthesized just before it.

use super::frame::{bone_frames, region_joints};
use super::synth::{self, SynthInput};
use concinnity_world::registry::build_only::CharacterSchema;
use concinnity_world::registry::build_only::KeyPolarity;

use crate::components::{MorphDelta, SkeletonJoint, SkinnedVertexData};

// A source's morph set: names and dense target-major deltas.
pub(crate) struct MorphSet {
    pub names: Vec<String>,
    pub deltas: Vec<MorphDelta>,
}

impl MorphSet {
    pub(crate) fn target(&self, name: &str, vertex_count: usize) -> Option<&[MorphDelta]> {
        let t = self.names.iter().position(|n| n == name)?;
        Some(&self.deltas[t * vertex_count..(t + 1) * vertex_count])
    }

    fn push(&mut self, name: String, deltas: Vec<MorphDelta>) {
        self.names.push(name);
        self.deltas.extend(deltas);
    }
}

// The source targets a generator with `source` reads under a polarity: the
// bare name, or the `+` / `-` pair.
fn source_names(source: &str, polarity: KeyPolarity) -> Vec<String> {
    match polarity {
        KeyPolarity::Unipolar => vec![source.to_string()],
        KeyPolarity::Bipolar => vec![format!("{source}+"), format!("{source}-")],
    }
}

fn output_names(name: &str, polarity: KeyPolarity) -> Vec<String> {
    source_names(name, polarity)
}

pub(crate) fn synthesize(
    schema: &CharacterSchema,
    skeleton: &[SkeletonJoint],
    vertices: &[SkinnedVertexData],
    indices: &[u16],
    morphs: &mut MorphSet,
) -> Result<(), String> {
    if schema.synthesized.is_empty() {
        return Ok(());
    }
    let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.pos).collect();
    let normals = synth::vertex_normals(&positions, indices);
    let frames = bone_frames(skeleton, vertices);
    for target in &schema.synthesized {
        let region = schema.region(&target.region).ok_or_else(|| {
            format!(
                "synthesized '{}': unknown region '{}'",
                target.name, target.region
            )
        })?;
        let members = region_joints(skeleton, &region.joints);
        let primary = region
            .joints
            .iter()
            .find_map(|n| skeleton.iter().position(|j| j.name == *n))
            .ok_or_else(|| {
                format!(
                    "synthesized '{}': region '{}' has no joint in the skeleton",
                    target.name, target.region
                )
            })?;
        let input = SynthInput {
            vertices,
            indices,
            normals: &normals,
            frames: &frames,
            members: &members,
            primary,
            params: &target.params,
        };
        let outputs = match target.generator.as_str() {
            "girth" => synth::polarised(
                &target.name,
                target.polarity,
                &input,
                synth::girth::displace(&input),
            ),
            "taper" => synth::polarised(
                &target.name,
                target.polarity,
                &input,
                synth::taper::displace(&input),
            ),
            "bulge" => synth::polarised(
                &target.name,
                target.polarity,
                &input,
                synth::bulge::displace(&input),
            ),
            "surface_offset" => synth::polarised(
                &target.name,
                target.polarity,
                &input,
                synth::surface_offset::displace(&input),
            ),
            "mirror" | "blend_mask" => {
                let mut out = Vec::new();
                for (src, dst) in source_names(&target.params.source, target.polarity)
                    .into_iter()
                    .zip(output_names(&target.name, target.polarity))
                {
                    let source = morphs
                        .target(&src, vertices.len())
                        .map(<[MorphDelta]>::to_vec)
                        .ok_or_else(|| {
                            format!(
                                "synthesized '{}': source target '{}' is not on the mesh",
                                target.name, src
                            )
                        })?;
                    let displacement = if target.generator == "mirror" {
                        synth::mirror::displace(&input, &source)
                    } else {
                        synth::blend_mask::displace(&input, &source)
                    };
                    out.push((
                        dst,
                        synth::finish(vertices, indices, &normals, &displacement),
                    ));
                }
                out
            }
            other => {
                return Err(format!(
                    "synthesized '{}': unknown generator '{}'",
                    target.name, other
                ));
            }
        };
        for (name, deltas) in outputs {
            if morphs.names.contains(&name) {
                return Err(format!(
                    "synthesized '{}': target '{}' is already on the mesh",
                    target.name, name
                ));
            }
            morphs.push(name, deltas);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::synth::test_support::cylinder;
    use super::*;
    use concinnity_world::registry::build_only::SchemaRegion;
    use concinnity_world::registry::build_only::SynthParams;
    use concinnity_world::registry::build_only::SynthesizedTarget;

    fn target(
        name: &str,
        generator: &str,
        polarity: KeyPolarity,
        params: SynthParams,
    ) -> SynthesizedTarget {
        SynthesizedTarget {
            name: name.into(),
            generator: generator.into(),
            region: "all".into(),
            polarity,
            params,
            ..Default::default()
        }
    }

    fn schema(targets: Vec<SynthesizedTarget>) -> CharacterSchema {
        CharacterSchema {
            regions: vec![SchemaRegion {
                name: "all".into(),
                joints: vec!["bone".into()],
            }],
            synthesized: targets,
            ..Default::default()
        }
    }

    #[test]
    fn targets_append_in_order_and_may_chain() {
        let (verts, idx, sk) = cylinder(8, 2, 0.5, 1.0);
        let mut morphs = MorphSet {
            names: vec!["authored".into()],
            deltas: vec![MorphDelta::default(); verts.len()],
        };
        let s = schema(vec![
            target(
                "girth",
                "girth",
                KeyPolarity::Bipolar,
                SynthParams::default(),
            ),
            target(
                "girth_masked",
                "blend_mask",
                KeyPolarity::Bipolar,
                SynthParams {
                    source: "girth".into(),
                    ..Default::default()
                },
            ),
            target(
                "girth_mirror",
                "mirror",
                KeyPolarity::Unipolar,
                SynthParams {
                    source: "girth+".into(),
                    ..Default::default()
                },
            ),
            target(
                "peak",
                "bulge",
                KeyPolarity::Unipolar,
                SynthParams::default(),
            ),
            target(
                "ramp",
                "taper",
                KeyPolarity::Unipolar,
                SynthParams::default(),
            ),
            target(
                "skin",
                "surface_offset",
                KeyPolarity::Unipolar,
                SynthParams::default(),
            ),
        ]);
        synthesize(&s, &sk, &verts, &idx, &mut morphs).expect("synthesize");
        assert_eq!(
            morphs.names,
            [
                "authored",
                "girth+",
                "girth-",
                "girth_masked+",
                "girth_masked-",
                "girth_mirror",
                "peak",
                "ramp",
                "skin"
            ]
        );
        assert_eq!(morphs.deltas.len(), 9 * verts.len());
        let plus = morphs.target("girth+", verts.len()).unwrap();
        let masked = morphs.target("girth_masked+", verts.len()).unwrap();
        assert_eq!(
            plus[1].position, masked[1].position,
            "full region weight keeps the source"
        );
        let mirrored = morphs.target("girth_mirror", verts.len()).unwrap();
        assert!(
            (mirrored[1].position[0] - plus[1].position[0]).abs() < 1e-6,
            "a symmetric push mirrors onto itself"
        );
    }

    #[test]
    fn errors_name_the_target_and_the_problem() {
        let (verts, idx, sk) = cylinder(4, 1, 0.5, 1.0);
        let run = |t: SynthesizedTarget| {
            let mut morphs = MorphSet {
                names: vec![],
                deltas: vec![],
            };
            synthesize(&schema(vec![t]), &sk, &verts, &idx, &mut morphs).unwrap_err()
        };
        let err = run(target(
            "x",
            "warp",
            KeyPolarity::Unipolar,
            SynthParams::default(),
        ));
        assert!(err.contains("unknown generator 'warp'"), "{err}");
        let err = run(target(
            "x",
            "mirror",
            KeyPolarity::Unipolar,
            SynthParams {
                source: "nope".into(),
                ..Default::default()
            },
        ));
        assert!(
            err.contains("source target 'nope' is not on the mesh"),
            "{err}"
        );
        let mut t = target("x", "girth", KeyPolarity::Unipolar, SynthParams::default());
        t.region = "wings".into();
        let err = run(t);
        assert!(err.contains("unknown region 'wings'"), "{err}");
        let mut morphs = MorphSet {
            names: vec!["x".into()],
            deltas: vec![MorphDelta::default(); verts.len()],
        };
        let err = synthesize(
            &schema(vec![target(
                "x",
                "girth",
                KeyPolarity::Unipolar,
                SynthParams::default(),
            )]),
            &sk,
            &verts,
            &idx,
            &mut morphs,
        )
        .unwrap_err();
        assert!(err.contains("already on the mesh"), "{err}");
    }
}
