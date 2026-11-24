//! Checks on `water.hlsl` itself, for what no host-side mirror of its math can show.

use crate::render::shader::embedded::WATER;

// The water fragments take the pixel footprint from camera rays, never screen
// derivatives, which Apple GPUs return inconsistently on partly covered quads
// and so made far triangles flicker between runs.
#[test]
fn the_water_shader_takes_no_screen_derivatives() {
    for (number, line) in WATER.lines().enumerate() {
        let code = line.split("//").next().unwrap_or_default();
        for derivative in ["ddx", "ddy", "fwidth"] {
            assert!(
                !code.contains(derivative),
                "water.hlsl:{}: `{derivative}` in `{}`; take the footprint from \
                 `water_footprint`, whose rays depend on the pixel and the view alone",
                number + 1,
                line.trim()
            );
        }
    }
}
