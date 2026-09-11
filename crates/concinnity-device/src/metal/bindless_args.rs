// Change detection for the per-frame Metal argument buffers.
//
// The bindless texture block and the probe cube block are written with an
// argument encoder, one Obj-C message send per texture handle. Their contents
// change only on a texture stream/evict, a probe bake, an env-map swap or a
// transient-pool repack, but they live in a per-frame ring slot, so a change
// has to be re-encoded into every slot before the encode can stop.
//
// [`SlotGates`] is that bookkeeping: one accepted signature per ring slot, so a
// producer re-encodes a slot only when the signature it computes differs from
// the one that slot already holds. [`ResidencySet`] does the same for the
// `useResource` calls an argument buffer's contents need, and collapses them
// into one batched call.

#![deny(unsafe_op_in_unsafe_fn)]

use core::ptr::NonNull;
use objc2::Message as _;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLRenderCommandEncoder, MTLRenderStages, MTLResource, MTLResourceUsage, MTLTexture,
};

// Folds resource identities and counters into one 64-bit value. Two signatures
// comparing equal means every input was identical, so the consumer of the
// signature can be left as it was.
pub(super) struct Signature(std::collections::hash_map::DefaultHasher);

impl Signature {
    pub(super) fn new() -> Self {
        Self(std::collections::hash_map::DefaultHasher::new())
    }

    pub(super) fn push_u64(&mut self, v: u64) {
        use std::hash::Hasher as _;
        self.0.write_u64(v);
    }

    // The object identity of a Metal resource. A handle replaced by a mutator
    // nothing thought to announce (a transient-pool repack re-points the AO
    // texture, for instance) still moves the signature.
    pub(super) fn push_texture(&mut self, tex: &ProtocolObject<dyn MTLTexture>) {
        use std::hash::Hasher as _;
        self.0.write_usize(tex as *const _ as *const () as usize);
    }

    pub(super) fn finish(&self) -> u64 {
        use std::hash::Hasher as _;
        self.0.finish()
    }
}

// One accepted signature per ring slot. A slot with no accepted signature (the
// initial state, and the state [`Self::invalidate`] returns it to) is always
// stale, so a freshly allocated buffer is always written before it is read.
pub(super) struct SlotGates {
    accepted: Vec<Option<u64>>,
}

impl SlotGates {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            accepted: vec![None; depth.max(1)],
        }
    }

    // Whether `slot` must be re-encoded for `sig`, accepting `sig` as its
    // contents if so. Slots are addressed modulo the gate count, matching
    // `TransientRing::slot`.
    pub(super) fn stale(&mut self, slot: usize, sig: u64) -> bool {
        let idx = slot % self.accepted.len();
        if self.accepted[idx] == Some(sig) {
            return false;
        }
        self.accepted[idx] = Some(sig);
        true
    }

    // Forget `slot`'s contents, so the next [`Self::stale`] re-encodes it. Call
    // when the underlying buffer was reallocated out from under the gate.
    pub(super) fn invalidate(&mut self, slot: usize) {
        let idx = slot % self.accepted.len();
        self.accepted[idx] = None;
    }
}

// The textures an argument buffer names, kept resident for indirect execution.
//
// An argument buffer's contents are not tracked by the encoder, so a texture
// reached only through one reads garbage unless the pass declares it. The set
// changes with the argument buffer itself, so it is rebuilt behind the same
// kind of signature gate and issued as a single batched `useResources`.
pub(super) struct ResidencySet {
    accepted: Option<u64>,
    // Owns a reference to each declared texture, which is what keeps the
    // pointers in `ptrs` live. Rebuilt together with `ptrs`, never apart.
    owned: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    ptrs: Vec<NonNull<ProtocolObject<dyn MTLResource>>>,
}

impl ResidencySet {
    pub(super) fn new() -> Self {
        Self {
            accepted: None,
            owned: Vec::new(),
            ptrs: Vec::new(),
        }
    }

    // Rebuild the set from `textures` when `sig` differs from the set it
    // currently holds. Cheap enough to call unconditionally once per frame.
    pub(super) fn refresh<'a>(
        &mut self,
        sig: u64,
        textures: impl Iterator<Item = &'a ProtocolObject<dyn MTLTexture>>,
    ) {
        if self.accepted == Some(sig) {
            return;
        }
        self.accepted = Some(sig);
        self.owned.clear();
        self.owned.extend(textures.map(|t| t.retain()));
        // Filled only once `owned` has stopped growing: a reallocation would
        // leave every pointer taken from an earlier push dangling.
        self.ptrs.clear();
        self.ptrs.extend(
            self.owned
                .iter()
                .map(|t| NonNull::from(ProtocolObject::from_ref(&**t))),
        );
    }

    // Declare the whole set resident for the fragment stage in one call. A
    // no-op before the first [`Self::refresh`].
    pub(super) fn declare_fragment(&self, enc: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        if self.ptrs.is_empty() {
            return;
        }
        let Some(first) = NonNull::new(self.ptrs.as_ptr().cast_mut()) else {
            return;
        };
        // SAFETY: `first` addresses a contiguous array of `self.ptrs.len()`
        // live resource pointers, each kept alive by the matching entry in
        // `self.owned`; the encoder reads the array for the call's duration only.
        unsafe {
            enc.useResources_count_usage_stages(
                first,
                self.ptrs.len(),
                MTLResourceUsage::Read,
                MTLRenderStages::Fragment,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Signature, SlotGates};

    #[test]
    fn signature_is_order_sensitive_over_counters() {
        let sig = |vals: &[u64]| {
            let mut s = Signature::new();
            for v in vals {
                s.push_u64(*v);
            }
            s.finish()
        };
        assert_eq!(sig(&[1, 2, 3]), sig(&[1, 2, 3]));
        assert_ne!(sig(&[1, 2, 3]), sig(&[3, 2, 1]));
        assert_ne!(sig(&[1, 2, 3]), sig(&[1, 2, 4]));
    }

    #[test]
    fn a_fresh_gate_is_stale_for_every_slot() {
        let mut gates = SlotGates::new(3);
        for slot in 0..3 {
            assert!(gates.stale(slot, 7));
        }
    }

    #[test]
    fn an_accepted_signature_is_not_stale_again() {
        let mut gates = SlotGates::new(2);
        assert!(gates.stale(0, 7));
        assert!(!gates.stale(0, 7));
        assert!(!gates.stale(0, 7));
    }

    // The point of the per-slot table: one change has to reach every slot in
    // the ring, because each holds its own copy of the argument buffer.
    #[test]
    fn a_change_is_stale_once_per_slot() {
        let mut gates = SlotGates::new(3);
        for slot in 0..3 {
            assert!(gates.stale(slot, 7));
        }
        for slot in 0..3 {
            assert!(gates.stale(slot, 8));
            assert!(!gates.stale(slot, 8));
        }
    }

    #[test]
    fn slots_wrap_modulo_the_gate_count() {
        let mut gates = SlotGates::new(2);
        assert!(gates.stale(0, 7));
        assert!(!gates.stale(2, 7));
        assert!(gates.stale(3, 7));
    }

    #[test]
    fn invalidate_forces_one_re_encode() {
        let mut gates = SlotGates::new(2);
        assert!(gates.stale(1, 7));
        gates.invalidate(1);
        assert!(gates.stale(1, 7));
        assert!(!gates.stale(1, 7));
    }
}
