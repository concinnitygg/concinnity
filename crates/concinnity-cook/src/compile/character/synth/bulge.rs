// `bulge`: a gaussian lobe centred at `along` on each region bone, `sigma`
// wide, pushing in `direction` (model space) on the side of the bone that
// faces it, or radially when no direction is given.

use super::{SynthInput, unit};
use concinnity_core::math::vec3;

pub(crate) fn displace(input: &SynthInput) -> Vec<[f32; 3]> {
    let p = input.params;
    let dir = unit(p.direction);
    let sigma = p.sigma.max(1e-3);
    input
        .vertices
        .iter()
        .map(|v| {
            let mut d = [0.0_f32; 3];
            for (j, w) in input.region_influences(v) {
                let frame = &input.frames[j];
                let t = frame.along(v.pos);
                let g = (-(t - p.along).powi(2) / (2.0 * sigma * sigma)).exp();
                let Some(radial) = unit(frame.radial(v.pos)) else {
                    continue;
                };
                let (push, facing) = match dir {
                    Some(dir) => (dir, vec3::dot(radial, dir).max(0.0)),
                    None => (radial, 1.0),
                };
                vec3::vec3_add(&mut d, vec3::scale(push, w * g * facing * p.amplitude));
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
    fn a_radial_lobe_peaks_at_along_and_fades_with_sigma() {
        let (verts, idx, sk) = cylinder(8, 10, 0.5, 1.0);
        let mut fx = Fixture::new(verts, idx, &sk);
        fx.params.amplitude = 0.1;
        fx.params.along = 0.5;
        fx.params.sigma = 0.1;
        let d = displace(&fx.input());
        // Ring r sits at along = r / 10.
        let at = |r: usize| vec3::length(d[r * 8]);
        assert!((at(5) - 0.1).abs() < 1e-5, "peak at the centre");
        let expected_4 = 0.1 * (-(0.1_f32).powi(2) / (2.0 * 0.01)).exp();
        assert!((at(4) - expected_4).abs() < 1e-5, "{}", at(4));
        assert!((at(4) - at(6)).abs() < 1e-5, "symmetric about the centre");
        assert!(at(0) < 1e-5 && at(10) < 1e-5, "nothing at the ends");
    }

    #[test]
    fn a_directed_lobe_only_raises_the_facing_side() {
        let (verts, idx, sk) = cylinder(8, 2, 0.5, 1.0);
        let mut fx = Fixture::new(verts, idx, &sk);
        fx.params.amplitude = 0.1;
        fx.params.direction = [1.0, 0.0, 0.0];
        fx.params.sigma = 10.0;
        let d = displace(&fx.input());
        // Middle ring: vertex 0 faces +X, vertex 4 faces -X, vertex 2 is +Z.
        assert!(
            (d[8][0] - 0.1).abs() < 1e-5 && d[8][2].abs() < 1e-6,
            "{:?}",
            d[8]
        );
        assert_eq!(d[8 + 4], [0.0; 3], "the far side is untouched");
        assert!(
            vec3::length(d[8 + 2]) < 1e-5,
            "the tangent side is untouched"
        );
        // The push is along the direction, not the radial.
        assert!(d[8 + 1][2].abs() < 1e-6 && d[8 + 1][0] > 0.0);
    }
}
