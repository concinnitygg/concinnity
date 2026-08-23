// `surface_offset`: push along the vertex normal, scaled by the skin weight
// to the region and a window along the region's first bone (`span`, with
// `falloff`-wide ramps), and limited to the side facing `direction` when one
// is given. Brow ridges, cheekbones, skin thickness.

use super::{SynthInput, unit};
use crate::character::frame::region_weight;
use concinnity_core::math::vec3;

// Smooth window weight of `t` inside `span`: 1 inside, ramping to 0 over
// `falloff` at each end.
pub(crate) fn window(t: f32, span: [f32; 2], falloff: f32) -> f32 {
    let f = falloff.max(1e-6);
    let rise = ((t - span[0]) / f).clamp(0.0, 1.0);
    let fall = ((span[1] - t) / f).clamp(0.0, 1.0);
    rise * fall
}

pub(crate) fn displace(input: &SynthInput) -> Vec<[f32; 3]> {
    let p = input.params;
    let dir = unit(p.direction);
    let frame = &input.frames[input.primary];
    input
        .vertices
        .iter()
        .zip(input.normals)
        .map(|(v, n)| {
            let mask = region_weight(v, input.members);
            let w = window(frame.along(v.pos), p.span, p.falloff);
            let facing = dir.map_or(1.0, |d| vec3::dot(*n, d).max(0.0));
            vec3::scale(*n, mask * w * facing * p.amplitude)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn the_window_ramps_in_and_out_over_the_falloff() {
        assert_eq!(window(0.5, [0.2, 0.8], 0.1), 1.0);
        assert_eq!(window(0.2, [0.2, 0.8], 0.1), 0.0);
        assert!((window(0.25, [0.2, 0.8], 0.1) - 0.5).abs() < 1e-6);
        assert!((window(0.75, [0.2, 0.8], 0.1) - 0.5).abs() < 1e-6);
        assert_eq!(window(0.9, [0.2, 0.8], 0.1), 0.0);
        assert_eq!(
            window(0.5, [0.0, 1.0], 0.0),
            1.0,
            "no falloff is a hard window"
        );
    }

    #[test]
    fn a_sphere_band_inflates_along_its_normals_on_the_facing_side() {
        let (verts, idx) = sphere(8, 8);
        let sk = vec![joint("bone", -1, [0.0, -1.0, 0.0])];
        let mut fx = Fixture::new(verts, idx, &sk);
        // The bone runs from the south pole up through the sphere (length =
        // reach = 2), so along = (y + 1) / 2: the equator is 0.5.
        fx.params.amplitude = 0.1;
        fx.params.span = [0.4, 0.6];
        fx.params.falloff = 0.05;
        fx.params.direction = [0.0, 0.0, 1.0];
        let d = displace(&fx.input());
        let equator = 4 * 8;
        let front = equator + 2; // +Z
        let back = equator + 6; // -Z
        let side = equator; // +X
        assert!(
            d[front][2] > 0.09 && d[front][2] <= 0.1 + 1e-6,
            "{:?}",
            d[front]
        );
        assert_eq!(d[back], [0.0; 3]);
        assert!(vec3::length(d[side]) < 1e-5, "tangent to the direction");
        // Off the band nothing moves: the pole rings are at along 0 / 1.
        assert_eq!(d[0], [0.0; 3]);
        assert_eq!(d[8 * 8], [0.0; 3]);
        // Without a direction the whole band inflates radially.
        fx.params.direction = [0.0; 3];
        let d = displace(&fx.input());
        assert!(vec3::length(d[back]) > 0.09);
        let dir = vec3::vec3_normalise(d[back]);
        assert!(dir[2] < -0.99, "outward along the normal: {dir:?}");
    }
}
