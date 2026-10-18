//! What a planet world's camera needs next: the ground patch moved under it
//! once it walks off the current one's middle, or the whole simulated frame
//! moved to it once it strays too far from the origin.

use super::{
    GroundPatch, LocalFrame, PlanetFrame, PlanetShape, REBASE_DISTANCE, Rebase, ground_patch,
    needs_rebase,
};

/// Cells along each side of a ground patch.
pub const PATCH_CELLS: u32 = 128;

/// Half the width of a ground patch, in meters: one-meter cells.
pub const PATCH_HALF_WIDTH: f32 = 64.0;

// How far the camera may walk from the ground patch's center before a new
// patch is built around it, in meters: well inside the patch's half width.
const FOLLOW_DISTANCE: f32 = PATCH_HALF_WIDTH * 0.375;

/// What a patch request is for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatchPurpose {
    /// The camera walked from the patch; same frame.
    Follow,
    /// The frame moves once the patch built in it arrives.
    Rebase {
        /// The frame the world moves to.
        frame: LocalFrame,
        /// The move into it.
        rebase: Rebase,
    },
}

/// A patch to build: its purpose, the frame it is built in, and the local
/// point it is centered on.
pub type PatchPlan = (PatchPurpose, LocalFrame, [f32; 3]);

/// What the camera at local point `camera` needs next, if anything.
pub fn next_patch(
    camera: [f32; 3],
    frame: &PlanetFrame,
    patch_center: Option<[f32; 3]>,
) -> Option<PatchPlan> {
    if needs_rebase(camera, REBASE_DISTANCE) {
        let up = frame.shape.up_at(frame.frame.to_world(camera));
        let (next, rebase) = frame.frame.recentered(camera, up);
        let around = rebase.apply_point(camera);
        return Some((
            PatchPurpose::Rebase {
                frame: next,
                rebase,
            },
            next,
            around,
        ));
    }
    let strayed = patch_center.is_none_or(|c| {
        let (dx, dz) = (camera[0] - c[0], camera[2] - c[2]);
        dx * dx + dz * dz > FOLLOW_DISTANCE * FOLLOW_DISTANCE
    });
    strayed.then_some((PatchPurpose::Follow, frame.frame, camera))
}

/// The ground patch around local point `around` of `frame`, or `None` where
/// the ground cannot be built there (a camera off the planet).
pub fn build_patch(
    shape: &PlanetShape,
    frame: &LocalFrame,
    around: [f32; 3],
) -> Option<GroundPatch> {
    ground_patch(shape, frame, around, PATCH_CELLS, PATCH_HALF_WIDTH)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: PlanetShape = PlanetShape {
        center: [0.0, -50_000.0, 0.0],
        radius: 50_000.0,
        amplitude: 20.0,
        feature_size: 1_000.0,
        octaves: 4,
        seed: 1,
    };

    fn frame() -> PlanetFrame {
        PlanetFrame {
            shape: SHAPE,
            frame: LocalFrame::AUTHORED,
        }
    }

    #[test]
    fn a_camera_near_the_patch_middle_needs_nothing() {
        assert_eq!(next_patch([5.0, 1.0, -3.0], &frame(), Some([0.0; 3])), None);
    }

    #[test]
    fn a_camera_off_the_patch_middle_moves_the_patch() {
        let (purpose, f, around) = next_patch([30.0, 1.0, 0.0], &frame(), Some([0.0; 3])).unwrap();
        assert_eq!(purpose, PatchPurpose::Follow);
        assert_eq!(f, LocalFrame::AUTHORED);
        assert_eq!(around, [30.0, 1.0, 0.0]);
        assert!(
            next_patch([0.0; 3], &frame(), None).is_some(),
            "no patch yet"
        );
    }

    // Far enough out, the frame moves to the camera and the patch is built
    // around it there.
    #[test]
    fn a_camera_far_from_the_origin_moves_the_frame() {
        let camera = [1_001.0, -9.0, 0.0];
        let (purpose, f, around) = next_patch(camera, &frame(), Some([1_000.0, 0.0, 0.0])).unwrap();
        let PatchPurpose::Rebase {
            frame: next,
            rebase,
        } = purpose
        else {
            panic!("expected a rebase, got {purpose:?}");
        };
        assert_eq!(next, f);
        assert!(around.iter().all(|c| c.abs() < 1e-3), "{around:?}");
        assert_eq!(rebase.apply_point(camera), around);
    }

    #[test]
    fn a_patch_is_built_around_the_point_asked_for() {
        let patch = build_patch(&SHAPE, &LocalFrame::AUTHORED, [10.0, 0.0, -4.0]).unwrap();
        assert_eq!(patch.center[0], 10.0);
        assert_eq!(patch.center[2], -4.0);
    }
}
