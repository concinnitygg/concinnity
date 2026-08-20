// src/metal/rt_ring.rs
//
// Per-in-flight-frame storage for the ray-tracing structures the skinned update
// rewrites every frame: the deformed-vertex buffer, one BLAS per skinned object,
// the TLAS, and the instance / geometry-table / build-scratch buffers.
//
// The skinned update used to allocate all of those fresh every frame and park
// the outgoing set in the `RetirePool`. That is correct but it is device-
// allocator traffic at frame rate. Because the skinned update runs on EVERY
// frame, its outputs fit the ring rule the upload buffers in `transient.rs`
// already follow: frame `R` writes slot `R % depth` and is the only frame that
// binds it, and the frames-in-flight fence guarantees the previous writer of
// that slot (frame `R - depth`) has retired on the GPU. So a slot's storage can
// simply be rebuilt in place.
//
// The rule does NOT extend to the static `rebuild_tlas` path. A sparsely-moving
// scene keeps tracing one TLAS across many frames without rebuilding, so that
// structure is read by frames the fence does not pair with its writer; see the
// `RetirePool` doc comment. Anything published from a ring slot must therefore
// be unpublished the moment the skinned path stops running, which is what
// `RtFrameSlot::release` is for.
//
// Sizes are high-water: a slot never shrinks, so a steady scene allocates once
// and then does nothing.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAccelerationStructure, MTLBuffer, MTLDevice, MTLInstanceAccelerationStructureDescriptor,
    MTLPrimitiveAccelerationStructureDescriptor, MTLResource as _, MTLResourceOptions,
};

use super::transient::grow_to;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Structure = Retained<ProtocolObject<dyn MTLAccelerationStructure>>;
type PrimDesc = Retained<MTLPrimitiveAccelerationStructureDescriptor>;

// Full rebuilds per slot: after this many consecutive refits, the next skinned
// update rebuilds the slot's BLAS from scratch. A refit keeps the tree the last
// full build produced and only re-fits its bounding boxes, so traversal quality
// decays as the pose drifts away from the one the tree was built for. Each slot
// counts independently and they are touched on different frames, so the
// rebuilds stagger rather than landing on one frame.
const REFIT_LIMIT: u32 = 32;

// The geometry one skinned BLAS is built over: its slice of the shared skinned
// index buffer. Equal signatures mean the same triangles with only the vertex
// positions moved, which is exactly when Metal allows a refit; anything else
// (a mesh hot-reload, a different mesh becoming visible) changes the triangle
// set and needs a full rebuild.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct SkinnedShape {
    pub index_offset: usize,
    pub index_count: usize,
}

// Whether a slot's skinned BLAS can be refit from this frame's pose or must be
// rebuilt from scratch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum BlasUpdate {
    Build,
    Refit,
}

// Rebuild when the triangle set changed (a refit is then illegal), when the slot
// has no built tree to refit, or once every `limit` refits to bound the quality
// drift. Pure so the cadence is unit-testable without a GPU.
fn blas_update(shape_changed: bool, built: bool, refits: u32, limit: u32) -> BlasUpdate {
    if shape_changed || !built || refits >= limit {
        BlasUpdate::Build
    } else {
        BlasUpdate::Refit
    }
}

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
    shape: Vec<SkinnedShape>,
    // Build scratch the descriptors above reported when `blas` was allocated.
    blas_scratch: usize,
    // Whether `blas` holds trees a refit can update (false until first built).
    built: bool,
    refits: u32,
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
            shape: Vec::new(),
            blas_scratch: 0,
            built: false,
            refits: 0,
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
    ) -> Result<(Buffer, bool), String> {
        let have = self.deformed.as_ref().map_or(0, |b| b.length());
        let mut fresh = false;
        if let Some(cap) = grow_to(have, bytes) {
            let buf = device
                .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
                .ok_or("failed to allocate RT deformed-vertex buffer")?;
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
    ) -> Result<Buffer, String> {
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
    ) -> Result<Buffer, String> {
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
    ) -> Result<Buffer, String> {
        let have = self.scratch.as_ref().map_or(0, |b| b.length());
        if let Some(cap) = grow_to(have, bytes) {
            let buf = device
                .newBufferWithLength_options(cap, MTLResourceOptions::StorageModePrivate)
                .ok_or("failed to allocate RT scratch buffer")?;
            buf.setLabel(Some(&super::pipeline::ns_str("rt_scratch")));
            self.scratch = Some(buf);
        }
        Ok(self
            .scratch
            .as_ref()
            .expect("scratch slot was just ensured")
            .clone())
    }

    // Whether this slot's BLAS were last built over exactly `shapes`.
    pub(super) fn shape_matches(&self, shapes: &[SkinnedShape]) -> bool {
        self.shape == shapes
    }

    // Replace this slot's skinned BLAS with structures built over `shapes`, along
    // with the descriptors they were sized from and the build scratch they need.
    // The outgoing structures are dropped in place: a slot is written only by the
    // frame that owns it, and the fence guarantees the previous writer retired.
    pub(super) fn set_skinned(&mut self, built: SkinnedBlasSet, shapes: &[SkinnedShape]) {
        self.blas = built.blas;
        self.descs = built.descs;
        self.blas_scratch = built.scratch_bytes;
        self.shape.clear();
        self.shape.extend_from_slice(shapes);
        self.built = false;
        self.refits = 0;
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

    // How this frame's skinned BLAS should be updated, recording the choice so
    // the refit run stays bounded. `shape_changed` must be set when the
    // structures or the buffer they trace were just replaced.
    pub(super) fn plan_blas_update(&mut self, shape_changed: bool) -> BlasUpdate {
        let update = blas_update(shape_changed, self.built, self.refits, REFIT_LIMIT);
        match update {
            BlasUpdate::Build => {
                self.built = true;
                self.refits = 0;
            }
            BlasUpdate::Refit => self.refits += 1,
        }
        update
    }

    // The top-level structure, (re)allocated when `size` outgrows it. Sizing is
    // high-water so an instance count that oscillates does not reallocate.
    pub(super) fn tlas(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        size: usize,
    ) -> Result<Structure, String> {
        if self.tlas.is_none() || self.tlas_size < size {
            let tlas = device
                .newAccelerationStructureWithSize(size.max(1))
                .ok_or("failed to allocate TLAS")?;
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
        if self.blas.is_empty() && !self.built {
            return;
        }
        self.blas.clear();
        self.descs.clear();
        self.shape.clear();
        self.blas_scratch = 0;
        self.built = false;
        self.refits = 0;
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

// Copy a `#[repr(C)]` slice into a shared-storage GPU buffer at offset 0. The
// ring's instance-descriptor and geometry-table slots are filled this way each
// frame instead of being reallocated around a fresh upload.
pub(super) fn write_slice<T: Copy>(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    data: &[T],
) -> Result<(), String> {
    let bytes = std::mem::size_of_val(data);
    if bytes == 0 {
        return Ok(());
    }
    let len = buffer.length();
    if bytes > len {
        return Err(format!(
            "RT upload of {bytes} bytes exceeds buffer length {len}"
        ));
    }
    // SAFETY: `buffer` is shared storage so `contents()` is a live CPU mapping, and the bounds
    // check above proved it holds `bytes`. `data` is a separate live allocation of exactly that
    // many bytes, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            bytes,
        );
    }
    Ok(())
}

fn shared_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
    label: &str,
    what: &str,
) -> Result<Buffer, String> {
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| format!("failed to allocate buffer for {what}"))?;
    buf.setLabel(Some(&super::pipeline::ns_str(label)));
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_changed_triangle_set_forces_a_full_build() {
        // A refit cannot add or remove geometry, so a changed shape always
        // rebuilds even when the slot has a tree and refits to spare.
        assert_eq!(blas_update(true, true, 0, 32), BlasUpdate::Build);
    }

    #[test]
    fn an_unbuilt_slot_cannot_be_refit() {
        assert_eq!(blas_update(false, false, 0, 32), BlasUpdate::Build);
    }

    #[test]
    fn a_stable_shape_refits_until_the_limit() {
        assert_eq!(blas_update(false, true, 0, 32), BlasUpdate::Refit);
        assert_eq!(blas_update(false, true, 31, 32), BlasUpdate::Refit);
        // The 32nd refit is instead a rebuild, bounding the quality drift.
        assert_eq!(blas_update(false, true, 32, 32), BlasUpdate::Build);
        assert_eq!(blas_update(false, true, 99, 32), BlasUpdate::Build);
    }

    #[test]
    fn a_zero_limit_never_refits() {
        assert_eq!(blas_update(false, true, 0, 0), BlasUpdate::Build);
    }

    #[test]
    fn shape_equality_is_offset_and_count() {
        let a = SkinnedShape {
            index_offset: 12,
            index_count: 300,
        };
        assert_eq!(a, a);
        assert_ne!(
            a,
            SkinnedShape {
                index_offset: 13,
                index_count: 300,
            }
        );
        assert_ne!(
            a,
            SkinnedShape {
                index_offset: 12,
                index_count: 303,
            }
        );
    }

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
