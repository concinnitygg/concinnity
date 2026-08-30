// `taper`: a radial push that ramps along each region bone from nothing at
// the joint to the full amplitude at its child (or the reverse), so the
// distal end of a limb thickens or thins on its own.

use super::{SynthInput, unit};
use concinnity_core::math::vec3;

pub(crate) fn displace(input: &SynthInput) -> Vec<[f32; 3]> {
    input
        .vertices
        .iter()
        .map(|v| {
            let mut d = [0.0_f32; 3];
            for (j, w) in input.region_influences(v) {
                let frame = &input.frames[j];
                let mut t = frame.along(v.pos).clamp(0.0, 1.0);
                if input.params.reverse {
                    t = 1.0 - t;
                }
                if let Some(r) = unit(frame.radial(v.pos)) {
                    vec3::vec3_add(&mut d, vec3::scale(r, w * t * input.params.amplitude));
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
    fn the_push_ramps_from_the_joint_to_the_bone_end() {
        let (verts, idx, sk) = cylinder(8, 4, 0.5, 2.0);
        let mut fx = Fixture::new(verts, idx, &sk);
        fx.params.amplitude = 0.2;
        let d = displace(&fx.input());
        // Ring r sits at y = r/4 * 2: along = y / 2.
        for r in 0..=4 {
            let got = vec3::length(d[r * 8]);
            let expected = 0.2 * r as f32 / 4.0;
            assert!(
                (got - expected).abs() < 1e-5,
                "ring {r}: {got} vs {expected}"
            );
        }
        fx.params.reverse = true;
        let d = displace(&fx.input());
        assert!((vec3::length(d[0]) - 0.2).abs() < 1e-5);
        assert!(vec3::length(d[4 * 8]) < 1e-5);
    }
}
