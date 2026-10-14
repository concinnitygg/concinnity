//! Checks on `water.hlsl` itself, for what no host-side mirror of its math can show.

use crate::render::shaders::WATER;

// The water fragments take the pixel's footprint from camera rays
// (`water_footprint`), never from screen derivatives. Over the ocean's far-field
// slivers most quads are only partly covered, and Apple GPUs return different
// derivatives for the same quad from one draw to the next; every fade, the
// cascade filtering and the roughness read the footprint, so whole far
// triangles shaded differently from run to run. No pixel comparison catches it
// reliably: the flicker shows in some frames of some launches.
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
