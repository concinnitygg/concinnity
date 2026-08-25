// `mirror`: derive the other side of an authored target by reflecting the
// mesh across X and reading each vertex's delta from the source vertex
// nearest its reflection, with the delta's X negated. Lets a generator
// author one side of an asymmetric key.

use super::SynthInput;
use crate::components::MorphDelta;
use std::collections::HashMap;

// Uniform grid over the positions, `cell` wide.
struct Grid {
    cell: f32,
    cells: HashMap<[i32; 3], Vec<usize>>,
}

impl Grid {
    fn new(positions: &[[f32; 3]]) -> Self {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in positions {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let extent = (0..3).map(|a| hi[a] - lo[a]).fold(0.0_f32, f32::max);
        let cell = (extent / 48.0).max(1e-4);
        let mut cells: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
        for (i, p) in positions.iter().enumerate() {
            cells.entry(Self::key(*p, cell)).or_default().push(i);
        }
        Self { cell, cells }
    }

    fn key(p: [f32; 3], cell: f32) -> [i32; 3] {
        [
            (p[0] / cell).floor() as i32,
            (p[1] / cell).floor() as i32,
            (p[2] / cell).floor() as i32,
        ]
    }

    // The index of the position nearest `p`, searching growing shells of
    // cells until one holds a candidate.
    fn nearest(&self, positions: &[[f32; 3]], p: [f32; 3]) -> usize {
        let k = Self::key(p, self.cell);
        let mut best: Option<(f32, usize)> = None;
        for radius in 0..64_i32 {
            for x in -radius..=radius {
                for y in -radius..=radius {
                    for z in -radius..=radius {
                        if x.abs() != radius && y.abs() != radius && z.abs() != radius {
                            continue;
                        }
                        let Some(bucket) = self.cells.get(&[k[0] + x, k[1] + y, k[2] + z]) else {
                            continue;
                        };
                        for &i in bucket {
                            let q = positions[i];
                            let d = (q[0] - p[0]).powi(2)
                                + (q[1] - p[1]).powi(2)
                                + (q[2] - p[2]).powi(2);
                            if best.is_none_or(|(b, _)| d < b) {
                                best = Some((d, i));
                            }
                        }
                    }
                }
            }
            // A hit in shell `radius` can be beaten by a point in shell
            // `radius + 1` only within one cell; one more shell settles it.
            if let Some((d, i)) = best
                && d.sqrt() <= (radius as f32) * self.cell
            {
                return i;
            }
        }
        best.map_or(0, |(_, i)| i)
    }
}

pub(crate) fn displace(input: &SynthInput, source: &[MorphDelta]) -> Vec<[f32; 3]> {
    let positions: Vec<[f32; 3]> = input.vertices.iter().map(|v| v.pos).collect();
    let grid = Grid::new(&positions);
    positions
        .iter()
        .map(|p| {
            let twin = grid.nearest(&positions, [-p[0], p[1], p[2]]);
            let s = source[twin].position;
            [-s[0], s[1], s[2]]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn a_one_sided_bump_lands_on_the_other_side_reflected() {
        let (verts, idx) = sphere(16, 8);
        let sk = vec![joint("bone", -1, [0.0, 0.0, 0.0])];
        let fx = Fixture::new(verts, idx, &sk);
        // Push the +X equator vertex outward and a little up.
        let mut source = vec![MorphDelta::default(); fx.vertices.len()];
        let plus_x = 4 * 16;
        source[plus_x].position = [0.2, 0.05, 0.0];
        let d = displace(&fx.input(), &source);
        let minus_x = 4 * 16 + 8;
        assert_eq!(d[minus_x], [-0.2, 0.05, 0.0]);
        assert_eq!(d[plus_x], [0.0; 3], "the authored side reads its own twin");
        assert_eq!(d.iter().filter(|v| **v != [0.0; 3]).count(), 1);
    }

    #[test]
    fn nearest_lookup_matches_brute_force() {
        let (verts, _) = sphere(12, 6);
        let positions: Vec<[f32; 3]> = verts.iter().map(|v| v.pos).collect();
        let grid = Grid::new(&positions);
        for probe in [
            [0.3, 0.2, -0.9],
            [-1.0, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.5, -0.5, 0.5],
        ] {
            let brute =
                (0..positions.len())
                    .min_by(|&a, &b| {
                        let da = concinnity_core::math::vec3::length(
                            concinnity_core::math::vec3::sub(positions[a], probe),
                        );
                        let db = concinnity_core::math::vec3::length(
                            concinnity_core::math::vec3::sub(positions[b], probe),
                        );
                        da.partial_cmp(&db).unwrap()
                    })
                    .unwrap();
            let got = grid.nearest(&positions, probe);
            let dist = |i: usize| {
                concinnity_core::math::vec3::length(concinnity_core::math::vec3::sub(
                    positions[i],
                    probe,
                ))
            };
            assert!(
                (dist(got) - dist(brute)).abs() < 1e-6,
                "{probe:?}: {got} vs {brute}"
            );
        }
    }
}
