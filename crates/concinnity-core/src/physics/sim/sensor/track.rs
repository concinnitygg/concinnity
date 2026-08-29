// Which pairs started overlapping and which stopped, from the two lists of
// pairs that were overlapping.
//
// A crossing is a boundary, not a state, so something has to remember last
// step's answer to know this step's is different. That memory is a sorted
// array walked beside this step's, exactly as the contact cache carries
// impulses: no hashing, and so no iteration order that could differ between
// two runs of the same scene.
//
// The slot pair is the key, and the handles are compared as well as carried.
// A slot freed and refilled between steps is the same key holding a different
// body, and reporting that as continuing overlap would hand a caller a
// crossing its region never saw.

use core::cmp::Ordering;

use crate::physics::BodyHandle;

use crate::physics::sim::broadphase::Pair;

/// One sensor pair that was overlapping when a step ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Overlap {
    /// The two body slots, ordered so `0 < 1`. Both lists are sorted by it.
    pub(crate) pair: Pair,
    /// The body in the lower slot, as of the step that recorded the overlap.
    pub(crate) a: BodyHandle,
    /// The body in the higher slot.
    pub(crate) b: BodyHandle,
}

impl Overlap {
    /// Whether both slots hold the same bodies they held in `other`.
    fn same_bodies(&self, other: &Overlap) -> bool {
        self.a == other.a && self.b == other.b
    }
}

/// Walk last step's overlaps beside this step's, reporting every pair that
/// began or stopped overlapping.
///
/// Both lists arrive sorted by slot pair, so one walk over each is enough and
/// the reports come out in slot order, with a pair that left reported before
/// the one that took its place.
pub(crate) fn transitions(
    previous: &[Overlap],
    current: &[Overlap],
    mut report: impl FnMut(&Overlap, bool),
) {
    let (mut was, mut now) = (0usize, 0usize);
    while was < previous.len() && now < current.len() {
        let (left, right) = (&previous[was], &current[now]);
        match left.pair.cmp(&right.pair) {
            Ordering::Less => {
                report(left, false);
                was += 1;
            }
            Ordering::Greater => {
                report(right, true);
                now += 1;
            }
            Ordering::Equal => {
                if !left.same_bodies(right) {
                    report(left, false);
                    report(right, true);
                }
                was += 1;
                now += 1;
            }
        }
    }
    for left in &previous[was..] {
        report(left, false);
    }
    for right in &current[now..] {
        report(right, true);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    fn handle(slot: u32, generation: u32) -> BodyHandle {
        BodyHandle::from_parts(slot, generation)
    }

    fn overlap(a: u32, b: u32) -> Overlap {
        Overlap {
            pair: (a, b),
            a: handle(a, 0),
            b: handle(b, 0),
        }
    }

    fn crossed(previous: &[Overlap], current: &[Overlap]) -> Vec<(Pair, bool)> {
        let mut out = Vec::new();
        transitions(previous, current, |overlap, entered| {
            out.push((overlap.pair, entered));
        });
        out
    }

    #[test]
    fn a_pair_that_arrives_enters_and_a_pair_that_goes_leaves() {
        assert_eq!(crossed(&[], &[overlap(0, 1)]), [((0, 1), true)]);
        assert_eq!(crossed(&[overlap(0, 1)], &[]), [((0, 1), false)]);
    }

    // The point of the whole module: sustained overlap is silent.
    #[test]
    fn a_pair_that_stayed_reports_nothing() {
        assert!(crossed(&[overlap(0, 1)], &[overlap(0, 1)]).is_empty());
    }

    // The merge has to survive pairs appearing and disappearing on either
    // side, and report in slot order whatever the mixture.
    #[test]
    fn the_merge_reports_both_sides_in_slot_order() {
        let previous = [overlap(0, 1), overlap(0, 5), overlap(3, 4)];
        let current = [overlap(0, 2), overlap(0, 5), overlap(3, 9)];
        assert_eq!(
            crossed(&previous, &current),
            [
                ((0, 1), false),
                ((0, 2), true),
                ((3, 4), false),
                ((3, 9), true),
            ]
        );
    }

    // A slot freed and refilled keeps the key but changes the bodies, and
    // both halves of that have to be reported.
    #[test]
    fn a_reused_slot_leaves_and_arrives_rather_than_staying() {
        let previous = [overlap(0, 1)];
        let current = [Overlap {
            pair: (0, 1),
            a: handle(0, 0),
            b: handle(1, 7),
        }];
        assert_eq!(
            crossed(&previous, &current),
            [((0, 1), false), ((0, 1), true)]
        );
    }

    #[test]
    fn two_empty_lists_report_nothing() {
        assert!(crossed(&[], &[]).is_empty());
    }

    // Every remaining entry on either side has to be drained, not just the
    // ones the paired walk reached.
    #[test]
    fn the_tail_of_the_longer_list_is_still_reported() {
        assert_eq!(
            crossed(
                &[overlap(0, 1)],
                &[overlap(0, 1), overlap(2, 3), overlap(4, 5)]
            ),
            [((2, 3), true), ((4, 5), true)]
        );
        assert_eq!(
            crossed(
                &[overlap(0, 1), overlap(2, 3), overlap(4, 5)],
                &[overlap(0, 1)]
            ),
            [((2, 3), false), ((4, 5), false)]
        );
    }
}
