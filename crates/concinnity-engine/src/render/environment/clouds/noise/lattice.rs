//! The lattice the clouds' noise is drawn on: gradient noise over a tile of
//! whole cells, hashed by `hash3`. `cloud_lattice.hlsl` mirrors it.

use super::hash3f;
use concinnity_core::math::{SaturatingFrom, floor};

// Lattice cell `c` wrapped into a tile `period` cells across.
fn wrap(c: i32, period: i32) -> u32 {
    u32::saturating_from(c.rem_euclid(period))
}

pub(super) fn wrap3(c: [i32; 3], period: i32) -> [u32; 3] {
    c.map(|v| wrap(v, period))
}

pub(super) fn cell_of(p: f32) -> i32 {
    // Lattice coordinates stay within a few thousand cells.
    #[expect(clippy::cast_possible_truncation, reason = "floored and small")]
    let c = floor(p) as i32;
    c
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

pub(super) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Gradient noise at `p` (in cells) over a tile `period` cells across, in about
// [-1, 1]. `cloud_perlin`.
pub(in crate::render::environment::clouds) fn perlin(p: [f32; 3], period: i32) -> f32 {
    let cell = p.map(cell_of);
    let f: [f32; 3] = core::array::from_fn(|i| p[i] - cell[i] as f32);
    let corner = |dx: i32, dy: i32, dz: i32| {
        let h = hash3f(wrap3([cell[0] + dx, cell[1] + dy, cell[2] + dz], period));
        let g = h.map(|v| v * 2.0 - 1.0);
        dot3(g, [f[0] - dx as f32, f[1] - dy as f32, f[2] - dz as f32])
    };
    let u = f.map(fade);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u[0]);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u[0]);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u[0]);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u[0]);
    lerp(lerp(x00, x10, u[1]), lerp(x01, x11, u[1]), u[2])
}
