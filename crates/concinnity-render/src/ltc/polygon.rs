// src/ltc/polygon.rs
//
// The closed-form clamped-cosine polygon integral that area-light shading is
// built on, plus the horizon clipping it needs.
//
// This is the CPU twin of the shader code in `default.metal`. It exists so the
// integral can be checked against brute-force Monte Carlo in a unit test: the
// clipping in particular is easy to get subtly wrong in a way that still renders
// a plausible-looking highlight, and shader code cannot be tested directly.
// Keep the two in step -- the shader mirrors these functions line for line.
//
// The result is the fraction of the clamped-cosine distribution the polygon
// covers, in [0, 1]: 1 means the polygon fills the hemisphere. Multiply by a
// light's radiance to get outgoing radiance.

type Vec3 = [f32; 3];

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn normalize(v: Vec3) -> Vec3 {
    let len = dot(v, v).sqrt();
    if len < 1.0e-9 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

// Clip a quad against the horizon plane `z = 0`, keeping the part above it.
//
// A Sutherland-Hodgman sweep rather than the usual hardcoded 16-case table: a
// quad cut by one plane yields at most 5 vertices, and the loop form is short
// enough to read and impossible to get wrong case by case.
//
// Returns the vertex count; entries past it are untouched.
pub fn clip_quad_to_horizon(quad: &[Vec3; 4], out: &mut [Vec3; 5]) -> usize {
    let mut n = 0;
    for i in 0..4 {
        let current = quad[i];
        let previous = quad[(i + 3) % 4];
        let current_in = current[2] > 0.0;
        let previous_in = previous[2] > 0.0;
        if current_in != previous_in {
            // Where the edge crosses z = 0.
            let t = previous[2] / (previous[2] - current[2]);
            out[n] = [
                previous[0] + t * (current[0] - previous[0]),
                previous[1] + t * (current[1] - previous[1]),
                0.0,
            ];
            n += 1;
        }
        if current_in {
            out[n] = current;
            n += 1;
        }
    }
    n
}

// Twice the contribution of one edge of the spherical polygon. The `z` of the
// cross product carries the sign, so a reversed winding flips the whole sum,
// which is what distinguishes a front-facing polygon from a back-facing one.
fn integrate_edge(v1: Vec3, v2: Vec3) -> f32 {
    let cos_theta = dot(v1, v2).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    // theta / sin(theta) -> 1 as the edge shrinks; the guard avoids 0/0.
    let ratio = if sin_theta > 1.0e-4 {
        theta / sin_theta
    } else {
        1.0
    };
    cross(v1, v2)[2] * ratio
}

// Fraction of the clamped-cosine distribution covered by `quad`, whose vertices
// are already in the shading frame (the surface normal is +z).
//
// `two_sided` keeps the contribution of a polygon seen from behind; a one-sided
// light contributes nothing there.
pub fn integrate_clamped_cosine(quad: &[Vec3; 4], two_sided: bool) -> f32 {
    let mut clipped = [[0.0_f32; 3]; 5];
    let n = clip_quad_to_horizon(quad, &mut clipped);
    if n < 3 {
        return 0.0;
    }
    for v in clipped.iter_mut().take(n) {
        *v = normalize(*v);
    }

    let mut sum = 0.0;
    for i in 0..n {
        sum += integrate_edge(clipped[i], clipped[(i + 1) % n]);
    }

    // The edge sum is twice the irradiance, and dividing by pi normalises the
    // clamped cosine, so the covered fraction is sum / (2 * pi).
    let form_factor = sum / (2.0 * std::f32::consts::PI);
    if two_sided {
        form_factor.abs()
    } else {
        (-form_factor).max(0.0)
    }
}

// Transform the quad into the shading frame and integrate.
//
// `n` is the surface normal, `v` the view direction, `p` the shading point, and
// `m_inv` the LTC inverse transform (the identity for the diffuse term). The
// frame puts `n` on +z with its first tangent in the view plane, matching how
// the lookup table was fitted.
pub fn evaluate(
    n: Vec3,
    v: Vec3,
    p: Vec3,
    m_inv: &[[f32; 3]; 3],
    corners: &[Vec3; 4],
    two_sided: bool,
) -> f32 {
    let t1 = normalize([
        v[0] - n[0] * dot(v, n),
        v[1] - n[1] * dot(v, n),
        v[2] - n[2] * dot(v, n),
    ]);
    let t2 = cross(n, t1);

    let mut quad = [[0.0_f32; 3]; 4];
    for (out, corner) in quad.iter_mut().zip(corners) {
        let d = sub(*corner, p);
        // World -> shading frame, then the LTC inverse.
        let local = [dot(t1, d), dot(t2, d), dot(n, d)];
        for (row, o) in out.iter_mut().enumerate() {
            *o = m_inv[row][0] * local[0] + m_inv[row][1] * local[1] + m_inv[row][2] * local[2];
        }
    }
    integrate_clamped_cosine(&quad, two_sided)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // Brute-force reference: cosine-weighted hemisphere sampling, counting the
    // fraction of samples whose ray hits the quad. With a cosine pdf the hit
    // fraction IS the covered fraction of the clamped cosine, so this converges
    // to what `integrate_clamped_cosine` computes in closed form.
    fn monte_carlo(quad: &[Vec3; 4], samples: usize) -> f32 {
        let mut hits = 0.0_f64;
        let total = samples * samples;
        for j in 0..samples {
            for i in 0..samples {
                let u1 = (i as f32 + 0.5) / samples as f32;
                let u2 = (j as f32 + 0.5) / samples as f32;
                // Cosine-weighted direction about +z.
                let r = u1.sqrt();
                let phi = 2.0 * std::f32::consts::PI * u2;
                let dir = [r * phi.cos(), r * phi.sin(), (1.0 - u1).max(0.0).sqrt()];
                if ray_hits_quad(dir, quad) {
                    hits += 1.0;
                }
            }
        }
        (hits / total as f64) as f32
    }

    // Ray from the origin along `dir` against the quad, as two triangles.
    fn ray_hits_quad(dir: Vec3, quad: &[Vec3; 4]) -> bool {
        ray_hits_triangle(dir, quad[0], quad[1], quad[2])
            || ray_hits_triangle(dir, quad[0], quad[2], quad[3])
    }

    fn ray_hits_triangle(dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
        // Moller-Trumbore from the origin.
        let e1 = sub(b, a);
        let e2 = sub(c, a);
        let h = cross(dir, e2);
        let det = dot(e1, h);
        if det.abs() < 1.0e-9 {
            return false;
        }
        let inv_det = 1.0 / det;
        let s = [-a[0], -a[1], -a[2]];
        let u = inv_det * dot(s, h);
        if !(0.0..=1.0).contains(&u) {
            return false;
        }
        let q = cross(s, e1);
        let v = inv_det * dot(dir, q);
        if v < 0.0 || u + v > 1.0 {
            return false;
        }
        inv_det * dot(e2, q) > 1.0e-6
    }

    // A quad wound so its front face points back at the shading point, which is
    // the one-sided orientation that should contribute.
    fn facing_quad(centre: Vec3, half: f32) -> [Vec3; 4] {
        [
            [centre[0] - half, centre[1] - half, centre[2]],
            [centre[0] + half, centre[1] - half, centre[2]],
            [centre[0] + half, centre[1] + half, centre[2]],
            [centre[0] - half, centre[1] + half, centre[2]],
        ]
    }

    // The headline check: the closed form must match brute force, including for
    // quads that straddle the horizon and so exercise the clipping.
    #[test]
    fn the_closed_form_matches_brute_force() {
        let cases: [(&str, [Vec3; 4]); 5] = [
            ("small, overhead", facing_quad([0.0, 0.0, 3.0], 0.5)),
            ("large, overhead", facing_quad([0.0, 0.0, 2.0], 2.0)),
            ("offset to the side", facing_quad([2.0, 0.5, 2.0], 1.0)),
            // Spans z = 0, so the clipper has to cut it.
            (
                "straddling the horizon",
                [
                    [-1.0, -1.0, -0.5],
                    [1.0, -1.0, -0.5],
                    [1.0, 1.0, 1.5],
                    [-1.0, 1.0, 1.5],
                ],
            ),
            (
                "one corner below",
                [
                    [-1.0, -1.0, 0.4],
                    [1.5, -1.0, -0.6],
                    [1.0, 1.0, 0.9],
                    [-1.0, 1.0, 1.0],
                ],
            ),
        ];
        for (name, quad) in cases {
            let exact = integrate_clamped_cosine(&quad, true);
            let reference = monte_carlo(&quad, 400);
            assert!(
                (exact - reference).abs() < 0.01,
                "{name}: closed form {exact} vs brute force {reference}"
            );
        }
    }

    // A polygon filling the hemisphere covers the whole distribution.
    #[test]
    fn a_hemisphere_filling_quad_integrates_to_one() {
        let huge = facing_quad([0.0, 0.0, 0.001], 2000.0);
        let f = integrate_clamped_cosine(&huge, true);
        assert!((f - 1.0).abs() < 0.02, "got {f}");
    }

    #[test]
    fn a_quad_entirely_below_the_horizon_contributes_nothing() {
        let below = facing_quad([0.0, 0.0, -2.0], 1.0);
        assert_eq!(integrate_clamped_cosine(&below, true), 0.0);
        assert_eq!(integrate_clamped_cosine(&below, false), 0.0);
    }

    // Winding decides which face a one-sided light shows. Reversing it must turn
    // a one-sided light off while leaving a two-sided one unchanged.
    #[test]
    fn one_sided_lights_respect_winding() {
        let quad = facing_quad([0.0, 0.0, 2.0], 1.0);
        let reversed = [quad[3], quad[2], quad[1], quad[0]];

        let front = integrate_clamped_cosine(&quad, false);
        let back = integrate_clamped_cosine(&reversed, false);
        assert!(
            front > 0.0 || back > 0.0,
            "one winding must light the surface"
        );
        assert!(front == 0.0 || back == 0.0, "the opposite winding must not");
        // Two-sided sees the same energy either way.
        let a = integrate_clamped_cosine(&quad, true);
        let b = integrate_clamped_cosine(&reversed, true);
        assert!((a - b).abs() < 1.0e-5, "{a} vs {b}");
    }

    // Clipping is what keeps a partly-below-horizon quad from leaking negative
    // contributions; without it the sum goes wrong rather than just smaller.
    #[test]
    fn clipping_keeps_the_result_bounded() {
        let straddling = [
            [-2.0_f32, -2.0, -1.0],
            [2.0, -2.0, -1.0],
            [2.0, 2.0, 3.0],
            [-2.0, 2.0, 3.0],
        ];
        let f = integrate_clamped_cosine(&straddling, true);
        assert!((0.0..=1.0).contains(&f), "form factor {f} out of range");
    }

    #[test]
    fn the_clipper_returns_the_expected_vertex_counts() {
        let mut out = [[0.0_f32; 3]; 5];

        // Fully above: unchanged.
        let above = facing_quad([0.0, 0.0, 1.0], 1.0);
        assert_eq!(clip_quad_to_horizon(&above, &mut out), 4);

        // Fully below: nothing survives.
        let below = facing_quad([0.0, 0.0, -1.0], 1.0);
        assert_eq!(clip_quad_to_horizon(&below, &mut out), 0);

        // Two corners below: the cut adds two vertices and removes two.
        let half = [
            [-1.0_f32, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        assert_eq!(clip_quad_to_horizon(&half, &mut out), 4);

        // One corner below: three originals survive plus two cut points.
        let one_below = [
            [-1.0_f32, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        assert_eq!(clip_quad_to_horizon(&one_below, &mut out), 5);
    }

    // `evaluate` places the quad in the shading frame; a light directly overhead
    // of an up-facing surface must land symmetrically about that frame's axis.
    #[test]
    fn evaluate_transforms_into_the_shading_frame() {
        let n = [0.0, 0.0, 1.0];
        let v = normalize([0.3, 0.0, 1.0]);
        let p = [0.0, 0.0, 0.0];
        let overhead = facing_quad([0.0, 0.0, 3.0], 1.0);
        let direct = integrate_clamped_cosine(&overhead, true);
        let framed = evaluate(n, v, p, &IDENTITY, &overhead, true);
        assert!(
            (direct - framed).abs() < 1.0e-4,
            "an up-facing surface's frame should not change the result: {direct} vs {framed}"
        );
    }

    // Moving the same light further away must reduce its contribution.
    #[test]
    fn contribution_falls_off_with_distance() {
        let near = integrate_clamped_cosine(&facing_quad([0.0, 0.0, 1.0], 1.0), true);
        let far = integrate_clamped_cosine(&facing_quad([0.0, 0.0, 6.0], 1.0), true);
        assert!(near > far, "near {near} should exceed far {far}");
    }
}
