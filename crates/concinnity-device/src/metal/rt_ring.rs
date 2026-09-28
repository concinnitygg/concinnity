//! Per-in-flight-frame storage for the ray-tracing structures the skinned update
//! rewrites every frame: the deformed-vertex buffer, one BLAS per skinned object,
//! the TLAS, and the instance / geometry-table / build-scratch buffers.
//!
//! The skinned update used to allocate all of those fresh every frame and park
//! the outgoing set in the `RetirePool`. That is correct but it is device-
//! allocator traffic at frame rate. Because the skinned update runs on EVERY
//! frame, its outputs fit the ring rule the upload buffers in `frame_rings.rs`
//! already follow: frame `R` writes slot `R % depth` and is the only frame that
//! binds it, and the frames-in-flight fence guarantees the previous writer of
//! that slot (frame `R - depth`) has retired on the GPU. So a slot's storage can
//! simply be rebuilt in place.
//!
//! The rule does NOT extend to the static `rebuild_tlas` path. A sparsely-moving
//! scene keeps tracing one TLAS across many frames without rebuilding, so that
//! structure is read by frames the fence does not pair with its writer; see the
//! `RetirePool` doc comment. Anything published from a ring slot must therefore
//! be unpublished the moment the skinned path stops running, which is what
//! `RtFrameSlot::release` is for.
//!
//! Sizes are high-water: a slot never shrinks, so a steady scene allocates once
//! and then does nothing.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAccelerationStructure, MTLBuffer, MTLDevice, MTLInstanceAccelerationStructureDescriptor,
    MTLPrimitiveAccelerationStructureDescriptor, MTLResource as _, MTLResourceOptions,
};

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::rt_refit::SkinnedRefit;

use super::error::allocation_failed;
use super::frame_rings::grow_to;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Structure = Retained<ProtocolObject<dyn MTLAccelerationStructure>>;
type PrimDesc = Retained<MTLPrimitiveAccelerationStructureDescriptor>;

// Identifies the TLAS descriptor a slot has cached. A descriptor pins the array
// of referenced BLAS, the instance buffer and the instance count; while all
// three are unchanged the same descriptor drives every rebuild (a build re-reads
// the instance buffer's current contents), so it does not have to be rebuilt --
// which is what keeps the per-frame `Vec` of BLAS references off the heap.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct TlasKey {
    // Bumped by the owner whenever the persistent BLAS head changes identity.
    pub head_generation: u64,
    // Bumped by the slot whenever its own BLAS or instance buffer are replaced.
    pub slot_generation: u64,
    pub instance_count: usize,
}

// A freshly-allocated set of skinned BLAS for one slot, with the descriptors
// they were sized from and the largest build scratch any of them needs. Built by
// the caller (which owns the descriptor shapes) and handed to the slot to own.
pub(super) struct SkinnedBlasSet {
    pub blas: Vec<Structure>,
    pub descs: Vec<PrimDesc>,
    pub scratch_bytes: usize,
}

// One in-flight frame's storage.
pub(super) struct RtFrameSlot {
    deformed: Option<Buffer>,
    blas: Vec<Structure>,
    descs: Vec<PrimDesc>,
    // Build scratch the descriptors above reported when `blas` was allocated.
    blas_scratch: usize,
    // The shapes `blas` were last built over and the refit run since.
    pub refit: SkinnedRefit,
    tlas: Option<Structure>,
    tlas_size: usize,
    tlas_desc: Option<(
        TlasKey,
        Retained<MTLInstanceAccelerationStructureDescriptor>,
    )>,
    instances: Option<Buffer>,
    geom_table: Option<Buffer>,
    scratch: Option<Buffer>,
    generation: u64,
}

impl RtFrameSlot {
    fn new() -> Self {
        Self {
            deformed: None,
            blas: Vec::new(),
            descs: Vec::new(),
            blas_scratch: 0,
            refit: SkinnedRefit::default(),
            tlas: None,
            tlas_size: 0,
            tlas_desc: None,
            instances: None,
            geom_table: None,
            scratch: None,
            generation: 0,
        }
    }

    // Bumped whenever this slot replaces a resource a cached TLAS descriptor
    // pins, so the owner's `TlasKey` stops matching and the descriptor rebuilds.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    // The deformed (posed) skinned vertex buffer, grown to `bytes`. Shared, not
    // Private: it is written by the skin compute pass and then read by both the
    // acceleration-structure build and the reflection fragment shader, which run
    // in separate command buffers -- a Private buffer in that cross-command-
    // buffer pattern was observed to GPU page-fault on the fragment read.
    //
    // The `bool` is true when the buffer was (re)allocated, which invalidates
    // every descriptor built over the old one.
    pub(super) fn deformed(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> RenderResult<(Buffer, bool)> {
        let have = self.deformed.as_ref().map_or(0, |b| b.length());
        let mut fresh = false;
        if let Some(cap) = grow_to(have, bytes) {
            let buf = device
                .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| allocation_failed("RT deformed-vertex buffer"))?;
            buf.setLabel(Some(&super::pipeline::ns_str("rt_deformed_verts")));
            self.deformed = Some(buf);
            self.generation = self.generation.wrapping_add(1);
            fresh = true;
        }
        let buf = self
            .deformed
            .as_ref()
            .expect("deformed slot was just ensured")
            .clone();
        Ok((buf, fresh))
    }

    // The TLAS instance-descriptor upload buffer, grown to `bytes`.
    pub(super) fn instances(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> RenderResult<Buffer> {
        let have = self.instances.as_ref().map_or(0, |b| b.length());
        if let Some(cap) = grow_to(have, bytes) {
            self.instances = Some(shared_buffer(
                device,
                cap,
                "rt_instances",
                "RT instance descriptors",
            )?);
            self.generation = self.generation.wrapping_add(1);
        }
        Ok(self
            .instances
            .as_ref()
            .expect("instance slot was just ensured")
            .clone())
    }

    // The per-instance geometry table the reflection kernel indexes by
    // `instance_id`, grown to `bytes`.
    pub(super) fn geom_table(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> RenderResult<Buffer> {
        let have = self.geom_table.as_ref().map_or(0, |b| b.length());
        if let Some(cap) = grow_to(have, bytes) {
            self.geom_table = Some(shared_buffer(
                device,
                cap,
                "rt_geom_table",
                "RT geometry table",
            )?);
        }
        Ok(self
            .geom_table
            .as_ref()
            .expect("geometry-table slot was just ensured")
            .clone())
    }

    // Private build / refit scratch, grown to `bytes`. Shared by every build on
    // this frame's command buffer: separate encoders serialize, so one buffer
    // covers them all.
    pub(super) fn scratch(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> RenderResult<Buffer> {
        let have = self.scratch.as_ref().map_or(0, |b| b.length());
        if let Some(cap) = grow_to(have, bytes) {
            let buf = device
                .newBufferWithLength_options(cap, MTLResourceOptions::StorageModePrivate)
                .ok_or_else(|| allocation_failed("RT scratch buffer"))?;
            buf.setLabel(Some(&super::pipeline::ns_str("rt_scratch")));
            self.scratch = Some(buf);
        }
        Ok(self
            .scratch
            .as_ref()
            .expect("scratch slot was just ensured")
            .clone())
    }

    // Replace this slot's skinned BLAS with fresh structures, along with the
    // descriptors they were sized from and the build scratch they need. The new
    // structures hold no tree, so the refit record resets. The outgoing ones are
    // dropped in place: a slot is written only by the frame that owns it, and the
    // fence guarantees the previous writer retired.
    pub(super) fn set_skinned(&mut self, built: SkinnedBlasSet) {
        self.blas = built.blas;
        self.descs = built.descs;
        self.blas_scratch = built.scratch_bytes;
        self.refit.reset();
        self.generation = self.generation.wrapping_add(1);
    }

    // Build scratch the slot's skinned BLAS need, from the sizes their
    // descriptors reported when they were allocated.
    pub(super) fn blas_scratch(&self) -> usize {
        self.blas_scratch
    }

    pub(super) fn skinned_blas(&self) -> &[Structure] {
        &self.blas
    }

    pub(super) fn skinned_descs(&self) -> &[PrimDesc] {
        &self.descs
    }

    // The top-level structure, (re)allocated when `size` outgrows it. Sizing is
    // high-water so an instance count that oscillates does not reallocate.
    pub(super) fn tlas(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        size: usize,
    ) -> RenderResult<Structure> {
        if self.tlas.is_none() || self.tlas_size < size {
            let tlas = device
                .newAccelerationStructureWithSize(size.max(1))
                .ok_or_else(|| allocation_failed("TLAS"))?;
            tlas.setLabel(Some(&super::pipeline::ns_str("rt_tlas")));
            self.tlas = Some(tlas);
            self.tlas_size = size;
        }
        Ok(self
            .tlas
            .as_ref()
            .expect("TLAS slot was just ensured")
            .clone())
    }

    // The cached TLAS descriptor, if it was built for `key`.
    pub(super) fn tlas_desc(
        &self,
        key: TlasKey,
    ) -> Option<Retained<MTLInstanceAccelerationStructureDescriptor>> {
        self.tlas_desc
            .as_ref()
            .filter(|(cached, _)| *cached == key)
            .map(|(_, desc)| desc.clone())
    }

    pub(super) fn set_tlas_desc(
        &mut self,
        key: TlasKey,
        desc: Retained<MTLInstanceAccelerationStructureDescriptor>,
    ) {
        self.tlas_desc = Some((key, desc));
    }

    // Forget the structures built over this slot's deformed buffer. Called when
    // the skinned path stops publishing (no skinned object is visible this
    // frame): the owner must stop binding this slot's resources at the same
    // moment, or a later rewrite of the slot could race a frame that still has
    // them bound. The buffers are kept -- only the pose-dependent structures are
    // invalid.
    pub(super) fn release(&mut self) {
        self.refit.reset();
        if self.blas.is_empty() {
            return;
        }
        self.blas.clear();
        self.descs.clear();
        self.blas_scratch = 0;
        self.generation = self.generation.wrapping_add(1);
    }
}

// One slot per frame in flight.
pub(super) struct RtFrameRing {
    slots: Vec<RtFrameSlot>,
}

impl RtFrameRing {
    // `depth` is the frames-in-flight count; clamped to >= 1. Every slot starts
    // empty and allocates on its first use.
    pub(super) fn new(depth: usize) -> Self {
        Self {
            slots: (0..depth.max(1)).map(|_| RtFrameSlot::new()).collect(),
        }
    }

    pub(super) fn slot(&mut self, ring_slot: usize) -> &mut RtFrameSlot {
        let idx = ring_slot % self.slots.len();
        &mut self.slots[idx]
    }

    // Drop the pose-dependent structures in every slot. Used when the skinned
    // path stops publishing, so no slot stays reachable through a stale handle.
    pub(super) fn release_all(&mut self) {
        for slot in &mut self.slots {
            slot.release();
        }
    }
}

fn shared_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
    label: &str,
    what: &str,
) -> RenderResult<Buffer> {
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| allocation_failed(format_args!("buffer for {what}")))?;
    buf.setLabel(Some(&super::pipeline::ns_str(label)));
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tlas_key_separates_head_slot_and_instance_count() {
        let base = TlasKey {
            head_generation: 1,
            slot_generation: 2,
            instance_count: 3,
        };
        assert_eq!(base, base);
        assert_ne!(
            base,
            TlasKey {
                head_generation: 2,
                ..base
            }
        );
        assert_ne!(
            base,
            TlasKey {
                slot_generation: 3,
                ..base
            }
        );
        assert_ne!(
            base,
            TlasKey {
                instance_count: 4,
                ..base
            }
        );
    }
}
