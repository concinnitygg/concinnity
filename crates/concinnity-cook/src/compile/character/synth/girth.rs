// `girth`: push every vertex radially away from the bone axes of the region
// it is skinned to, each influence weighted by its skin weight. A positive
// amplitude thickens the limb; the bipolar pair gives thin as well.

use super::{SynthInput, unit};
use concinnity_core::math::vec3;

pub(crate) fn displace(input: &SynthInput) -> Vec<[f32; 3]> {
    input
        .vertices
        .iter()
        .map(|v| {
            let mut d = [0.0_f32; 3];
            for (j, w) in input.region_influences(v) {
                if let Some(r) = unit(input.frames[j].radial(v.pos)) {
                    vec3::vec3_add(&mut d, vec3::scale(r, w * input.params.amplitude));
                }
            }
            d
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn a_cylinder_grows_by_the_amplitude_everywhere_off_axis() {
        let (verts, idx, sk) = cylinder(12, 3, 0.5, 2.0);
        let mut fx = Fixture::new(verts, idx, &sk);
        fx.params.amplitude = 0.1;
        let d = displace(&fx.input());
        for (v, d) in fx.vertices.iter().zip(&d) {
            let radial = [v.pos[0], 0.0, v.pos[2]];
            if vec3::length(radial) < 1e-6 {
                assert_eq!(*d, [0.0; 3], "pole vertices sit on the axis and stay");
            } else {
                let expected = vec3::scale(vec3::vec3_normalise(radial), 0.1);
                assert!(
                    vec3::length(vec3::sub(*d, expected)) < 1e-5,
                    "{d:?} vs {expected:?}"
                );
            }
        }
        // A finished target carries no normal change on a uniform radial
        // push of a cylinder side (the surface stays a cylinder).
        let out = super::super::finish(&fx.vertices, &fx.indices, &fx.normals, &d);
        let side = out[12 + 1].normal;
        assert!(vec3::length(side) < 0.05, "{side:?}");
    }

    #[test]
    fn skin_weight_scales_the_push_and_joints_outside_the_region_give_nothing() {
        let (mut verts, idx, mut sk) = cylinder(8, 2, 0.5, 2.0);
        sk.push(joint("other", 0, [0.0, 0.0, 0.0]));
        verts[0].joints = [0, 1, 0, 0];
        verts[0].weights = [0.25, 0.75, 0.0, 0.0];
        let mut fx = Fixture::new(verts, idx, &sk);
        fx.params.amplitude = 1.0;
        fx.members = vec![true, false];
        let d = displace(&fx.input());
        assert!((vec3::length(d[0]) - 0.25).abs() < 1e-5, "{:?}", d[0]);
        assert!((vec3::length(d[1]) - 1.0).abs() < 1e-5);
    }
}
