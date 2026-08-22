// src/gfx/graphics_system/lines.rs
//
// The per-frame bridge between the `WorldLines` resource a system publishes
// (world-space segments, in absolute world coordinates) and the ribbon
// geometry the backend's line pass rasterises. Producers stay ignorant
// of the camera and of camera-relative rendering; this is where both are
// applied.

use crate::ecs::PipelineContext;
use crate::gfx::lines::{Line, LineCamera, build_vertices_into};
use crate::gfx::render_types::LineVertex;

// The camera + space this frame draws with.
pub(super) struct LineFrame {
    pub view: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
    // World offset from authored space into the space the frame renders in.
    // Zero unless a streaming voxel world rebased the scene onto its chunk
    // render origin.
    pub(crate) rebase: [f32; 3],
    pub fov_y_radians: f32,
    pub near: f32,
    pub viewport: (f32, f32),
}

// Expand this frame's published lines into `out` (cleared first, capacity
// retained). A no-op past the resource lookup when nothing published any,
// which is every frame of a shipped runtime.
pub(super) fn build_into(ctx: &PipelineContext<'_>, frame: LineFrame, out: &mut Vec<LineVertex>) {
    out.clear();
    let Some(lines) = ctx.resource::<crate::ecs::WorldLines>() else {
        return;
    };
    if lines.0.is_empty() {
        return;
    }
    let cam = LineCamera {
        view: frame.view,
        cam_pos: frame.cam_pos,
        fov_y_radians: frame.fov_y_radians,
        viewport: [frame.viewport.0, frame.viewport.1],
        near: frame.near,
    };
    let rebased = lines.0.iter().map(|l| Line {
        start: offset(l.start, frame.rebase),
        end: offset(l.end, frame.rebase),
        ..*l
    });
    build_vertices_into(rebased, &cam, out);
}

fn offset(p: [f32; 3], by: [f32; 3]) -> [f32; 3] {
    [p[0] + by[0], p[1] + by[1], p[2] + by[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::BlobData;
    use crate::ecs::{ComponentStorage, PipelineContext, Resources};
    use crate::gfx::profile::FrameProfile;

    // Owns the storage a PipelineContext borrows from; the build reads only the
    // WorldLines resource, so the components / blob stay empty.
    struct TestWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
        scratch: crate::ecs::Arena,
    }

    impl TestWorld {
        fn new(lines: Vec<Line>) -> Self {
            let mut resources = Resources::new();
            if !lines.is_empty() {
                resources.insert(crate::ecs::WorldLines(lines));
            }
            Self {
                components: ComponentStorage::default(),
                blob: BlobData::new(vec![Some(Vec::new())]),
                profile: FrameProfile::default(),
                resources,
                scratch: crate::ecs::Arena::with_capacity(64 * 1024),
            }
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
                frame: crate::ecs::FrameContext::new(&self.scratch),
            }
        }
    }

    fn frame(rebase: [f32; 3]) -> LineFrame {
        LineFrame {
            view: crate::gfx::camera::view_matrix([0.0; 3], 0.0, 0.0),
            cam_pos: [0.0; 3],
            rebase,
            fov_y_radians: std::f32::consts::FRAC_PI_2,
            near: 0.1,
            viewport: (1280.0, 720.0),
        }
    }

    fn axis_line() -> Line {
        Line {
            start: [0.0, 0.0, -10.0],
            end: [0.0, 0.0, -100.0],
            start_color: [1.0, 0.2, 0.2, 1.0],
            end_color: [1.0, 0.2, 0.2, 0.0],
            width_px: 2.0,
        }
    }

    // Test shim over `build_into`, so the assertions stay value-shaped.
    fn build(ctx: &PipelineContext<'_>, frame: LineFrame) -> Vec<LineVertex> {
        let mut out = Vec::new();
        build_into(ctx, frame, &mut out);
        out
    }

    #[test]
    fn no_published_lines_means_no_geometry() {
        let mut w = TestWorld::new(Vec::new());
        assert!(build(&w.ctx(), frame([0.0; 3])).is_empty());
    }

    #[test]
    fn published_lines_expand_to_ribbons() {
        let mut w = TestWorld::new(vec![axis_line()]);
        assert_eq!(build(&w.ctx(), frame([0.0; 3])).len(), 6);
    }

    #[test]
    fn a_rebased_world_shifts_the_lines_with_it() {
        // The renderer draws a streaming voxel world relative to its chunk
        // origin, so an authored line must move by the same offset or it would
        // sit at the wrong place in the frame.
        let mut w = TestWorld::new(vec![axis_line()]);
        let plain = build(&w.ctx(), frame([0.0; 3]));
        let shifted = build(&w.ctx(), frame([5.0, 0.0, 0.0]));
        assert_eq!(plain.len(), shifted.len());
        // Compare ribbon centres: the two corners straddle the line, so their
        // midpoint is the segment endpoint itself (the corner offsets rotate
        // with the camera-facing normal and are not comparable directly).
        let centre = |v: &[LineVertex]| (v[0].pos[0] + v[1].pos[0]) * 0.5;
        assert!((centre(&shifted) - centre(&plain) - 5.0).abs() < 1e-3);
    }
}
