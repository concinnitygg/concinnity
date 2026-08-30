// `blend_mask`: an authored target restricted to a region by each vertex's
// skin weight to the region's joints, so one sculpted whole-body key yields
// a regional slider with a smooth boundary.

use super::SynthInput;
use crate::compile::character::frame::region_weight;
use crate::components::MorphDelta;
use concinnity_core::math::vec3;

pub(crate) fn displace(input: &SynthInput, source: &[MorphDelta]) -> Vec<[f32; 3]> {
    input
        .vertices
        .iter()
        .zip(source)
        .map(|(v, s)| vec3::scale(s.position, region_weight(v, input.members)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn the_source_is_scaled_by_the_region_weight() {
        let (mut verts, idx, mut sk) = cylinder(4, 1, 0.5, 1.0);
        sk.push(joint("other", 0, [0.0, 0.0, 0.0]));
        verts[0].joints = [0, 1, 0, 0];
        verts[0].weights = [0.3, 0.7, 0.0, 0.0];
        verts[1].joints = [1, 0, 0, 0];
        let mut fx = Fixture::new(verts, idx, &sk);
        fx.members = vec![true, false];
        let source: Vec<MorphDelta> = (0..fx.vertices.len())
            .map(|_| MorphDelta {
                position: [0.0, 1.0, 0.0],
                normal: [0.0; 3],
            })
            .collect();
        let d = displace(&fx.input(), &source);
        assert!((d[0][1] - 0.3).abs() < 1e-6, "{:?}", d[0]);
        assert_eq!(d[1], [0.0; 3], "fully outside the region");
        assert_eq!(d[2], [0.0, 1.0, 0.0], "fully inside keeps the source");
    }
}
