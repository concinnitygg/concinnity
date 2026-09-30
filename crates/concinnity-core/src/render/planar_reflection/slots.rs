//! Group reflector planes into a bounded number of mirror renders.

use super::mirror::normalize_plane;
use crate::math::vec3::{dot, length};
use alloc::vec::Vec;

type Vec4 = [f32; 4];

/// The result of grouping a list of reflection planes into a bounded number of
/// distinct slots: `slots[i]` is the slot a plane maps to (`None` when the budget
/// is exhausted by earlier distinct planes, i.e. it falls back to the probe cube),
/// and `representatives` is the deduplicated plane per slot (`representatives.len()`
/// is the number of mirror renders the frame needs).
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarAssignment {
    /// Per-reflector plane slot, `None` when the reflector fell back to a probe.
    pub slots: Vec<Option<usize>>,
    /// One representative plane per assigned slot.
    pub representatives: Vec<Vec4>,
}

/// Group near-coplanar reflection planes so each distinct plane renders one mirror
/// pass, capped at `max_slots`. Planes are matched sign-invariantly (a plane and
/// its flip are the same surface): two planes share a slot when their unit normals
/// are near-parallel and their offset along the normal matches. A plane coplanar
/// with an already-assigned slot always reuses it (even past the budget); only a
/// NEW distinct plane beyond `max_slots` overflows to `None`. Input order sets slot
/// priority, so callers list higher-priority planes (e.g. water) first.
pub fn assign_planar_slots(planes: &[Vec4], max_slots: usize) -> PlanarAssignment {
    // ~2.6 degrees of normal divergence and 0.1 world units of offset still count
    // as the same plane: tight enough to keep separate walls distinct, loose
    // enough to merge co-planar panes authored with slight slop.
    const NORMAL_DOT_EPS: f32 = 0.999;
    const OFFSET_EPS: f32 = 0.1;

    let mut representatives: Vec<Vec4> = Vec::new();
    let mut slots: Vec<Option<usize>> = Vec::with_capacity(planes.len());
    for &raw in planes {
        let p = normalize_plane(raw);
        let nlen = length([p[0], p[1], p[2]]);
        if nlen < 1e-6 {
            // Degenerate normal: no usable plane, fall back to the probe cube.
            slots.push(None);
            continue;
        }
        let mut found = None;
        for (i, r) in representatives.iter().enumerate() {
            let d = dot([p[0], p[1], p[2]], [r[0], r[1], r[2]]);
            if d.abs() >= NORMAL_DOT_EPS {
                // Align the representative to p's sign, then the two are the same
                // surface iff their plane constants match.
                let rd_aligned = if d < 0.0 { -r[3] } else { r[3] };
                if (p[3] - rd_aligned).abs() <= OFFSET_EPS {
                    found = Some(i);
                    break;
                }
            }
        }
        match found {
            Some(i) => slots.push(Some(i)),
            None => {
                if representatives.len() < max_slots {
                    representatives.push(p);
                    slots.push(Some(representatives.len() - 1));
                } else {
                    slots.push(None);
                }
            }
        }
    }
    PlanarAssignment {
        slots,
        representatives,
    }
}

/// Whether the transparent pass has to render its planar mirrors this frame.
///
/// Water prefers the mirror to the per-pixel ray trace: a water surface is a
/// plane, so one mirrored scene render resolves it exactly, at a fraction of the
/// cost of a ray per pixel, and the wave normal only has to perturb the lookup.
/// A trace off the per-fragment wave normal is hypersensitive at grazing angles
/// and lands on the probe / sky fallback wherever it misses, which reads as a
/// chrome sheet rather than water. Glass keeps the trace (a pane shows what is
/// genuinely behind it), so a world of glass alone still skips the re-render
/// while ray tracing is live.
///
/// `has_targets` is whether any mirror target exists at all, `water_has_slot`
/// whether a visible water surface holds one, and `rt_transparent_active`
/// whether the pass would otherwise trace.
pub fn planar_pass_needed(
    has_targets: bool,
    water_has_slot: bool,
    rt_transparent_active: bool,
) -> bool {
    has_targets && (water_has_slot || !rt_transparent_active)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_slots_dedups_coplanar_and_caps_distinct() {
        // Two coplanar panes (same wall), one distinct wall, plus a third distinct
        // wall that overflows a budget of 2. The coplanar pair shares slot 0, the
        // second wall takes slot 1, the third overflows to None.
        let wall_a0 = [0.0, 0.0, 1.0, -3.0];
        let wall_a1 = [0.0, 0.0, 1.0, -3.05]; // within OFFSET_EPS of a0
        let wall_b = [1.0, 0.0, 0.0, -5.0];
        let wall_c = [0.0, 1.0, 0.0, -1.0];
        let a = assign_planar_slots(&[wall_a0, wall_a1, wall_b, wall_c], 2);
        assert_eq!(a.representatives.len(), 2, "two slots allocated");
        assert_eq!(a.slots[0], Some(0));
        assert_eq!(a.slots[1], Some(0), "coplanar pane reuses slot 0");
        assert_eq!(a.slots[2], Some(1));
        assert_eq!(
            a.slots[3], None,
            "third distinct plane overflows the budget"
        );
    }

    #[test]
    fn assign_slots_is_sign_invariant() {
        // A plane and its flip (opposite normal, opposite offset) are the same
        // surface and must share a slot.
        let front = [0.0, 0.0, 1.0, -3.0];
        let back = [0.0, 0.0, -1.0, 3.0];
        let a = assign_planar_slots(&[front, back], 4);
        assert_eq!(a.representatives.len(), 1, "flip is the same surface");
        assert_eq!(a.slots[0], Some(0));
        assert_eq!(a.slots[1], Some(0));
    }

    #[test]
    fn assign_slots_overflow_still_reuses_existing_slot() {
        // With a budget of 1, a second distinct plane overflows, but a later plane
        // coplanar with slot 0 still maps to slot 0 (dedup precedes the cap).
        let a = assign_planar_slots(
            &[
                [0.0, 0.0, 1.0, -3.0],
                [1.0, 0.0, 0.0, -5.0], // overflow
                [0.0, 0.0, 1.0, -3.0], // coplanar with slot 0
            ],
            1,
        );
        assert_eq!(a.representatives.len(), 1);
        assert_eq!(a.slots[0], Some(0));
        assert_eq!(a.slots[1], None);
        assert_eq!(a.slots[2], Some(0));
    }

    #[test]
    fn planar_runs_for_water_even_while_ray_tracing() {
        // The whole point of the gate: a water surface holding a mirror slot keeps
        // the re-render alive under a live trace.
        assert!(planar_pass_needed(true, true, true));
        // Glass alone under a live trace still skips it.
        assert!(!planar_pass_needed(true, false, true));
        // With no trace, any reflector needs the mirror.
        assert!(planar_pass_needed(true, false, false));
        // No mirror target, nothing to render.
        assert!(!planar_pass_needed(false, true, true));
        assert!(!planar_pass_needed(false, false, false));
    }
}
