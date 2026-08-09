// src/editor/axes.rs
//
// The viewport's world-origin axes: one line per world axis, running from the
// origin out along its positive direction and fading to nothing at the camera's
// far plane, so the axis reads as unbounded without ending in a hard edge.
// Unlike the rest of the editor's viewport furniture these are world geometry,
// not overlay sprites: they go through the renderer's line pass, so
// scene geometry in front of an axis occludes it.
//
// Colours match the translate gizmo's handles (X red, Y green, Z blue), so the
// two teach the same axis mapping.

use concinnity_cpu::gfx::lines::Line;

// Fraction of the camera's far plane the line holds full alpha for, before it
// starts fading. The remainder ramps to zero, so the run dissolves into the
// distance instead of stopping.
const SOLID_FRACTION: f32 = 0.25;
// Screen thickness. Thin enough to read as a reference line rather than
// geometry, wide enough to survive the pass's edge antialiasing.
const WIDTH_PX: f32 = 2.0;
// Shortest run drawn when the camera's far plane is very close: a tiny far
// plane would otherwise leave the axes as stubs at the origin.
const MIN_EXTENT: f32 = 50.0;

// X red, Y green, Z blue: the gizmo handles' hues, carried at a higher
// saturation. Values are linear RGB written into the HDR scene target, so they
// are tone-mapped along with everything else, and the gizmo's softer tints
// (picked to sit on dark panel chrome) wash out to pastel over a lit scene.
const AXIS_COLORS: [[f32; 3]; 3] = [[0.90, 0.10, 0.12], [0.16, 0.75, 0.22], [0.16, 0.35, 0.95]];

const AXES: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// Append the origin axes for a camera whose far plane is `far`: per axis, a
// solid run out from the origin followed by a run that fades to nothing at the
// far plane. Appends into the frame's shared line buffer.
pub(crate) fn push_lines(out: &mut Vec<Line>, far: f32) {
    let extent = far.max(MIN_EXTENT);
    let solid = extent * SOLID_FRACTION;
    for (axis, rgb) in AXES.iter().zip(AXIS_COLORS) {
        let at = |d: f32| [axis[0] * d, axis[1] * d, axis[2] * d];
        let color = |a: f32| [rgb[0], rgb[1], rgb[2], a];
        out.push(Line {
            start: [0.0; 3],
            end: at(solid),
            start_color: color(1.0),
            end_color: color(1.0),
            width_px: WIDTH_PX,
        });
        out.push(Line {
            start: at(solid),
            end: at(extent),
            start_color: color(1.0),
            end_color: color(0.0),
            width_px: WIDTH_PX,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(far: f32) -> Vec<Line> {
        let mut out = Vec::new();
        push_lines(&mut out, far);
        out
    }

    #[test]
    fn one_solid_and_one_fading_run_per_axis() {
        let l = lines(200.0);
        assert_eq!(l.len(), 6);
        // Solid runs hold full alpha; the fading runs pick up where they end
        // and reach zero.
        for pair in l.chunks(2) {
            assert_eq!(pair[0].start_color[3], 1.0);
            assert_eq!(pair[0].end_color[3], 1.0);
            assert_eq!(pair[0].end, pair[1].start, "the runs meet");
            assert_eq!(pair[1].end_color[3], 0.0);
        }
    }

    #[test]
    fn every_run_leaves_the_origin_in_its_positive_direction() {
        let l = lines(200.0);
        assert_eq!(l[0].start, [0.0; 3]);
        // Each axis ends further out along exactly one positive component.
        let ends = [l[1].end, l[3].end, l[5].end];
        for (i, end) in ends.iter().enumerate() {
            assert!(end[i] > 0.0, "axis {i} runs positive: {end:?}");
            for (j, v) in end.iter().enumerate() {
                if j != i {
                    assert_eq!(*v, 0.0, "axis {i} stays on its own axis");
                }
            }
        }
    }

    #[test]
    fn the_run_reaches_the_far_plane() {
        assert_eq!(lines(500.0)[1].end[0], 500.0);
        // A near-sighted camera still gets a usable run rather than a stub.
        assert_eq!(lines(1.0)[1].end[0], MIN_EXTENT);
    }

    #[test]
    fn both_runs_of_an_axis_share_its_colour() {
        let l = lines(200.0);
        for (pair, rgb) in l.chunks(2).zip(AXIS_COLORS) {
            assert_eq!(pair[0].start_color[..3], rgb);
            assert_eq!(pair[1].start_color[..3], rgb);
        }
    }

    #[test]
    fn each_axis_reads_as_its_gizmo_hue() {
        // X red, Y green, Z blue, each dominated by its own channel by a wide
        // margin, so the three read apart at a glance and match the handles
        // the translate gizmo draws on the same axes.
        for (axis, rgb) in AXIS_COLORS.iter().enumerate() {
            let others: f32 = rgb
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, c)| *c)
                .fold(0.0, f32::max);
            assert!(rgb[axis] > others * 2.0, "axis {axis} hue is {rgb:?}");
        }
    }
}
