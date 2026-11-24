// The BLAS a trace can reach, kept resident by a residency set the command
// queues carry. A TLAS reaches its BLAS indirectly, which does not make them
// resident, so without the set every trace pass declares each one with
// `useResources`, and Metal allocates for each declaration every frame.

use std::collections::VecDeque;

use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2_metal::{
    MTLAccelerationStructure, MTLAllocation, MTLCommandQueue, MTLDevice, MTLResidencySet,
    MTLResidencySetDescriptor,
};

type Structure = Retained<ProtocolObject<dyn MTLAccelerationStructure>>;

pub(super) struct BlasResidency {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    // What the set holds, in the order the trace last listed it.
    held: Vec<Structure>,
    // Members no longer traced, with the frame that stopped tracing them. They
    // stay resident, and alive, until that frame's work has retired.
    leaving: VecDeque<(u64, Vec<Structure>)>,
}

impl BlasResidency {
    // A set every queue in `queues` keeps resident, or `None` on a device
    // without residency sets, where the trace passes declare the BLAS instead.
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        queues: &[&ProtocolObject<dyn MTLCommandQueue>],
    ) -> Option<Self> {
        if !device.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
            return None;
        }
        let set = device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .inspect_err(|e| tracing::warn!("BLAS residency set unavailable: {e}"))
            .ok()?;
        for queue in queues {
            queue.addResidencySet(&set);
        }
        Some(Self {
            set,
            held: Vec::new(),
            leaving: VecDeque::new(),
        })
    }

    // Bring the set in line with `traced`, the BLAS frame `frame_id` can trace.
    // Allocates only on a frame whose BLAS changed.
    pub(super) fn sync<'a>(
        &mut self,
        frame_id: u64,
        traced: impl Iterator<Item = &'a Structure> + Clone,
    ) {
        if self.held.iter().map(key).eq(traced.clone().map(key)) {
            return;
        }
        let next: Vec<Structure> = traced.cloned().collect();
        let (added, removed) = diff(&self.held, &next, key);
        for blas in &added {
            self.set.addAllocation(allocation(blas));
        }
        if !removed.is_empty() {
            self.leaving.push_back((frame_id, removed));
        }
        self.held = next;
        self.set.commit();
    }

    // Drop from the set what frames `depth` or more behind `frame_id` stopped
    // tracing: the frames-in-flight fence guarantees none of them still runs.
    pub(super) fn retire(&mut self, frame_id: u64, depth: u64) {
        let mut removed_any = false;
        while let Some((_, gone)) = self
            .leaving
            .pop_front_if(|(left, _)| left.saturating_add(depth) <= frame_id)
        {
            for blas in &gone {
                self.set.removeAllocation(allocation(blas));
            }
            removed_any = true;
        }
        if removed_any {
            self.set.commit();
        }
    }
}

fn key(blas: &Structure) -> *const ProtocolObject<dyn MTLAccelerationStructure> {
    Retained::as_ptr(blas)
}

fn allocation(blas: &Structure) -> &ProtocolObject<dyn MTLAllocation> {
    ProtocolObject::from_ref(&**blas)
}

// The members of `next` not in `held`, and of `held` not in `next`, by `key`.
fn diff<T: Clone, K: Ord>(held: &[T], next: &[T], key: impl Fn(&T) -> K) -> (Vec<T>, Vec<T>) {
    let sorted = |items: &[T]| {
        let mut items = items.to_vec();
        items.sort_by_key(&key);
        items
    };
    let (held, next) = (sorted(held), sorted(next));
    let only = |a: &[T], b: &[T]| -> Vec<T> {
        a.iter()
            .filter(|x| b.binary_search_by_key(&key(x), &key).is_err())
            .cloned()
            .collect()
    };
    (only(&next, &held), only(&held, &next))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_reports_what_joined_and_what_left() {
        let (added, removed) = diff(&[3, 1, 2], &[2, 4, 3, 5], |x| *x);
        assert_eq!(added, [4, 5]);
        assert_eq!(removed, [1]);
    }

    #[test]
    fn a_reordered_list_changes_nothing() {
        let (added, removed) = diff(&[1, 2, 3], &[3, 1, 2], |x| *x);
        assert!(added.is_empty() && removed.is_empty());
    }
}
