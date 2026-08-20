// src/vulkan/barrier_translate.rs
//
// Translate the render graph's coarse `ResourceState`, for a given resource
// class, into the concrete Vulkan (layout, access, stage) triple the executor
// feeds into an image memory barrier. The graph tracks only Undefined / Read /
// Write; the resource class (assigned by the executor's resolver) disambiguates
// what a `Write` means: a colour target writes COLOR_ATTACHMENT, a depth target
// writes DEPTH_STENCIL_ATTACHMENT. Both are sampled (SHADER_READ_ONLY) when
// read.
//
// For a barrier `from -> to`, the executor uses `from`'s triple for the source
// and `to`'s triple for the destination, and skips the barrier when the two
// layouts match (a no-op).
//
// A `Read`'s layout is SHADER_READ_ONLY either way, but its pipeline stage
// follows the consuming-stage union (`ReadStages`) carried on the barrier: a
// fragment consumer waits in FRAGMENT_SHADER, a compute consumer in
// COMPUTE_SHADER, and a resource read in both stages on one version waits in
// both so the single transition makes the producing write visible to each.
//
// `Undefined` never reaches the class mapping as a real transition: the executor
// resolves a barrier whose `from` is Undefined to the resource's *resting* layout
// (carried per-resource in the registry, not derived from the class) before
// translating, so the first per-frame transition names the layout the image is
// really in. Resting is per-resource because class does not determine it:
// `shadow_map` and `hdr_depth` are both depth targets, but the first rests
// sampled between frames -- its staggered cascades keep the depth they were last
// rendered with, and its producer barrier is the real SHADER_READ_ONLY ->
// DEPTH_STENCIL_ATTACHMENT reset -- while main depth carries nothing across the
// frame boundary and its first use discards.
//
// The reverse direction is `vk_restore`: a frame that leaves a resource somewhere
// other than its resting layout gets one transition back at the end of the frame,
// so the next frame's producer opens from the layout it names.

use ash::vk;

use crate::gfx::render_graph::{GraphResourceClass, ReadStages, ResourceState};

// The layout a resource sits in between frames, from which its first transition
// of a frame opens and to which the frame's last transition returns it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum VkResting {
    // Nothing survives the frame boundary: the first use may name UNDEFINED,
    // which is legal from any layout and discards the contents. Main depth and
    // the pooled colour transients rest this way.
    Discarded,
    // Sampled (SHADER_READ_ONLY) in the fragment stage, where the previous
    // frame's last consumer left it. The staggered `shadow_map` / `spot_shadow_map`
    // cascades, the froxel volume, and the Hi-Z pyramid rest this way: their
    // contents are read after the frame that wrote them.
    Sampled,
}

impl VkResting {
    // The layout alone, for the executor's per-frame check that the frame's
    // restores really do leave every driven resource where its next first use
    // expects it. That check runs under `debug_assertions`, and so does this.
    #[cfg(debug_assertions)]
    pub(super) fn layout(self) -> vk::ImageLayout {
        self.triple().0
    }

    // The (layout, access, stage) triple a resting resource sits in. Stage-fixed:
    // it describes where the previous frame left the image, not what the current
    // barrier's `read_stages` asks for.
    fn triple(self) -> (vk::ImageLayout, vk::AccessFlags, vk::PipelineStageFlags) {
        match self {
            VkResting::Discarded => (
                vk::ImageLayout::UNDEFINED,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            ),
            // Both shader stages: the resting stage is where the *previous*
            // frame's consumers left the image, and the executor does not carry a
            // read-stage union across the frame boundary. `shadow_map` is read in
            // both (Main samples it, the fog froxel kernel taps it), so a producer
            // that named only the fragment stage would leave the compute read
            // unordered against next frame's write.
            VkResting::Sampled => (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
        }
    }
}

// Map a `Read`'s consuming-stage union to the pipeline stages the transition
// must synchronise against. FRAGMENT -> FRAGMENT_SHADER, COMPUTE ->
// COMPUTE_SHADER, both -> both. An empty union (no Read side, or a resource no
// consumer reads) falls back to FRAGMENT_SHADER, the historical resting stage;
// the deriver never emits a `Read` barrier with an empty union, so the fallback
// is purely defensive.
fn read_stage_mask(stages: ReadStages) -> vk::PipelineStageFlags {
    let mut mask = vk::PipelineStageFlags::empty();
    if stages.contains(ReadStages::FRAGMENT) {
        mask |= vk::PipelineStageFlags::FRAGMENT_SHADER;
    }
    if stages.contains(ReadStages::COMPUTE) {
        mask |= vk::PipelineStageFlags::COMPUTE_SHADER;
    }
    if mask.is_empty() {
        mask = vk::PipelineStageFlags::FRAGMENT_SHADER;
    }
    mask
}

pub(super) fn vk_state(
    class: GraphResourceClass,
    state: ResourceState,
    read_stages: ReadStages,
) -> (vk::ImageLayout, vk::AccessFlags, vk::PipelineStageFlags) {
    let depth_attachment = (
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
    );
    match (class, state) {
        // Never a real transition source: `vk_transition` substitutes the
        // resource's resting triple. Kept as the total-match fallback.
        (_, ResourceState::Undefined) => VkResting::Discarded.triple(),
        (GraphResourceClass::ColorTarget, ResourceState::Write) => (
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        (GraphResourceClass::DepthTarget, ResourceState::Write) => depth_attachment,
        (GraphResourceClass::StorageImage, ResourceState::Write) => (
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        // Buffers carry no layout, so their triple keeps `UNDEFINED` throughout and
        // only the access + stage matter; `vk_transition` recognises the class and
        // emits a buffer barrier rather than skipping on the equal layouts.
        // Indirect draw arguments are consumed by the draw-indirect stage, which is
        // neither shader stage `read_stage_mask` models.
        (GraphResourceClass::IndirectBuffer, ResourceState::Read) => (
            vk::ImageLayout::UNDEFINED,
            vk::AccessFlags::INDIRECT_COMMAND_READ,
            vk::PipelineStageFlags::DRAW_INDIRECT,
        ),
        (
            GraphResourceClass::StorageBuffer | GraphResourceClass::UnorderedBuffer,
            ResourceState::Read,
        ) => (
            vk::ImageLayout::UNDEFINED,
            vk::AccessFlags::SHADER_READ,
            read_stage_mask(read_stages),
        ),
        (
            GraphResourceClass::IndirectBuffer
            | GraphResourceClass::StorageBuffer
            | GraphResourceClass::UnorderedBuffer,
            ResourceState::Write,
        ) => (
            vk::ImageLayout::UNDEFINED,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        // Every image class reads as a sampled image; the stage follows the
        // consuming run's union so a compute consumer synchronises in
        // COMPUTE_SHADER.
        (_, ResourceState::Read) => (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            read_stage_mask(read_stages),
        ),
    }
}

// Resolve a graph barrier `from -> to` for a resource of `class` whose
// between-frames layout is `resting`, into the concrete (old, new, src_access,
// dst_access, src_stage, dst_stage) the executor feeds into an image memory
// barrier + `cmd_pipeline_barrier`. A first-use `Undefined` source resolves to
// `resting` so the old layout names the one the image is really in. Returns
// `None` when old == new: a no-op the executor skips, e.g. a producer whose
// resting layout already equals its write layout.
//
// `read_stages` is the barrier's consuming-stage union (see `ReadStages`); it
// applies to whichever side is `Read` (the `to` of a consumer transition or the
// `from` of a Read -> Write WAR), driving that side's pipeline stage. The Write /
// Undefined side ignores it, so threading the single union through both
// `vk_state` calls is correct.
type VkTransition = (
    vk::ImageLayout,
    vk::ImageLayout,
    vk::AccessFlags,
    vk::AccessFlags,
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
);

pub(super) fn vk_transition(
    class: GraphResourceClass,
    resting: VkResting,
    from: ResourceState,
    to: ResourceState,
    read_stages: ReadStages,
) -> Option<VkTransition> {
    let (old, src_access, src_stage) = if from == ResourceState::Undefined {
        resting.triple()
    } else {
        vk_state(class, from, read_stages)
    };
    let (new, dst_access, dst_stage) = vk_state(class, to, read_stages);
    // A write following a write keeps one layout but still needs the dependency:
    // consecutive writers of one resource record into separate command buffers,
    // and command buffers in a submission may overlap in execution, so nothing
    // orders the second write after the first without this. Emitting it with
    // old == new makes it a pure execution + memory barrier.
    //
    // A buffer never changes layout at all, so the layout comparison would skip
    // every one of its transitions; its access + stage change is the whole point
    // of the barrier.
    let write_after_write = from == ResourceState::Write && to == ResourceState::Write;
    (old != new || write_after_write || class.is_buffer())
        .then_some((old, new, src_access, dst_access, src_stage, dst_stage))
}

// The end-of-frame transition returning a resource the frame left in `state` to
// its `resting` layout, so the next frame's first transition opens from the
// layout it names. `None` when the frame already ended there, or for a resource
// that rests discarded (its next first use may name UNDEFINED from any layout),
// or for a buffer (no layout to restore, and the frame fence retires the
// accesses that would need ordering).
pub(super) fn vk_restore(
    class: GraphResourceClass,
    resting: VkResting,
    state: ResourceState,
    read_stages: ReadStages,
) -> Option<VkTransition> {
    if resting == VkResting::Discarded || state == ResourceState::Undefined || class.is_buffer() {
        return None;
    }
    let (old, src_access, src_stage) = vk_state(class, state, read_stages);
    let (new, dst_access, dst_stage) = resting.triple();
    (old != new).then_some((old, new, src_access, dst_access, src_stage, dst_stage))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every graph-driven resource today is read in the fragment stage, so the
    // existing assertions pass the fragment union.
    const FRAG: ReadStages = ReadStages::FRAGMENT;

    #[test]
    fn class_state_mapping_is_pinned() {
        // Colour target: ao_output. Write is the colour attachment; read is
        // sampled.
        let (layout, access, stage) =
            vk_state(GraphResourceClass::ColorTarget, ResourceState::Write, FRAG);
        assert_eq!(layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(access, vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        assert_eq!(stage, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);

        // Depth target: write is the depth attachment, read is sampled like any
        // other image class.
        let read = vk_state(GraphResourceClass::DepthTarget, ResourceState::Read, FRAG);
        let write = vk_state(GraphResourceClass::DepthTarget, ResourceState::Write, FRAG);
        assert_eq!(write.0, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        assert_eq!(read.0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(
            vk_state(GraphResourceClass::ColorTarget, ResourceState::Read, FRAG),
            read
        );

        // Storage image: fog_froxel_volume / hiz_pyramid. Write is GENERAL in the
        // compute stage; read is sampled.
        let (sl, sa, ss) = vk_state(GraphResourceClass::StorageImage, ResourceState::Write, FRAG);
        assert_eq!(sl, vk::ImageLayout::GENERAL);
        assert_eq!(sa, vk::AccessFlags::SHADER_WRITE);
        assert_eq!(ss, vk::PipelineStageFlags::COMPUTE_SHADER);
        assert_eq!(
            vk_state(GraphResourceClass::StorageImage, ResourceState::Read, FRAG).0,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }

    #[test]
    fn read_stage_follows_consuming_union() {
        // The Read layout is always SHADER_READ_ONLY, but the stage tracks the
        // consuming union: fragment-only waits in FRAGMENT_SHADER, compute-only in
        // COMPUTE_SHADER, both in both (main depth, sampled by the decoration
        // passes and by the terminal Hi-Z build, is the case that exercises it).
        // Layout + access are unchanged across the three.
        let frag = vk_state(
            GraphResourceClass::ColorTarget,
            ResourceState::Read,
            ReadStages::FRAGMENT,
        );
        assert_eq!(frag.0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(frag.2, vk::PipelineStageFlags::FRAGMENT_SHADER);

        let comp = vk_state(
            GraphResourceClass::ColorTarget,
            ResourceState::Read,
            ReadStages::COMPUTE,
        );
        assert_eq!(comp.0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(comp.2, vk::PipelineStageFlags::COMPUTE_SHADER);

        let both = vk_state(
            GraphResourceClass::ColorTarget,
            ResourceState::Read,
            ReadStages::FRAGMENT | ReadStages::COMPUTE,
        );
        assert_eq!(
            both.2,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER
        );

        // Empty union falls back to the fragment stage; this is the resting stage
        // a sampled-resting resource's producer open keeps on its source side.
        let empty = vk_state(
            GraphResourceClass::ColorTarget,
            ResourceState::Read,
            ReadStages::empty(),
        );
        assert_eq!(empty.2, vk::PipelineStageFlags::FRAGMENT_SHADER);
    }

    #[test]
    fn a_first_use_opens_from_the_resource_resting_layout() {
        // The two restings a depth target can carry, on the same class: shadow_map
        // rests sampled, so its producer is the real cross-frame reset; main depth
        // rests discarded, so its producer opens from UNDEFINED and may run after
        // any layout the previous frame left.
        let (po, pn, ..) = vk_transition(
            GraphResourceClass::DepthTarget,
            VkResting::Sampled,
            ResourceState::Undefined,
            ResourceState::Write,
            FRAG,
        )
        .expect("a real cross-frame reset");
        assert_eq!(po, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(pn, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let (po, pn, ..) = vk_transition(
            GraphResourceClass::DepthTarget,
            VkResting::Discarded,
            ResourceState::Undefined,
            ResourceState::Write,
            FRAG,
        )
        .expect("a discarding open");
        assert_eq!(po, vk::ImageLayout::UNDEFINED);
        assert_eq!(pn, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    #[test]
    fn transition_resolves_migrated_producers_and_consumers() {
        // ao_output rests discarded: producer UNDEFINED -> COLOR_ATTACHMENT,
        // consumer COLOR_ATTACHMENT -> SHADER_READ.
        assert!(
            vk_transition(
                GraphResourceClass::ColorTarget,
                VkResting::Discarded,
                ResourceState::Undefined,
                ResourceState::Write,
                FRAG,
            )
            .is_some()
        );
        assert!(
            vk_transition(
                GraphResourceClass::ColorTarget,
                VkResting::Discarded,
                ResourceState::Write,
                ResourceState::Read,
                FRAG,
            )
            .is_some()
        );
        // shadow_map's consumer: DEPTH_STENCIL_ATTACHMENT -> SHADER_READ.
        let (old, new, ..) = vk_transition(
            GraphResourceClass::DepthTarget,
            VkResting::Sampled,
            ResourceState::Write,
            ResourceState::Read,
            FRAG,
        )
        .expect("a real close");
        assert_eq!(old, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        assert_eq!(new, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        // Generic no-op: a transition whose old and new layouts coincide (here a
        // sampled-resting resource asked for Undefined -> Read) is skipped.
        assert!(
            vk_transition(
                GraphResourceClass::DepthTarget,
                VkResting::Sampled,
                ResourceState::Undefined,
                ResourceState::Read,
                FRAG,
            )
            .is_none()
        );
        // fog_froxel_volume: producer SHADER_READ_ONLY -> GENERAL (real open),
        // consumer GENERAL -> SHADER_READ_ONLY (real close).
        let (po, pn, ..) = vk_transition(
            GraphResourceClass::StorageImage,
            VkResting::Sampled,
            ResourceState::Undefined,
            ResourceState::Write,
            FRAG,
        )
        .expect("a real open");
        assert_eq!(po, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(pn, vk::ImageLayout::GENERAL);
        let (co, cn, ..) = vk_transition(
            GraphResourceClass::StorageImage,
            VkResting::Sampled,
            ResourceState::Write,
            ResourceState::Read,
            FRAG,
        )
        .expect("a real close");
        assert_eq!(co, vk::ImageLayout::GENERAL);
        assert_eq!(cn, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    #[test]
    fn transition_threads_compute_read_stage() {
        // A compute consumer of a colour resource: the consumer Write -> Read keeps
        // the SHADER_READ_ONLY layout but its dst_stage is COMPUTE_SHADER; a mixed
        // run waits in both stages.
        let (.., src_stage, dst_stage) = vk_transition(
            GraphResourceClass::ColorTarget,
            VkResting::Discarded,
            ResourceState::Write,
            ResourceState::Read,
            ReadStages::COMPUTE,
        )
        .expect("a real close");
        assert_eq!(src_stage, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
        assert_eq!(dst_stage, vk::PipelineStageFlags::COMPUTE_SHADER);

        let (.., dst_stage) = vk_transition(
            GraphResourceClass::ColorTarget,
            VkResting::Discarded,
            ResourceState::Write,
            ResourceState::Read,
            ReadStages::FRAGMENT | ReadStages::COMPUTE,
        )
        .expect("a real close");
        assert_eq!(
            dst_stage,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER
        );

        // The WAR side: a Read -> Write whose prior run read in the compute stage
        // waits its src_stage on COMPUTE_SHADER (the readers it must not race).
        let (old, new, .., src_stage, _dst_stage) = vk_transition(
            GraphResourceClass::ColorTarget,
            VkResting::Discarded,
            ResourceState::Read,
            ResourceState::Write,
            ReadStages::COMPUTE,
        )
        .expect("a real WAR");
        assert_eq!(old, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(new, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(src_stage, vk::PipelineStageFlags::COMPUTE_SHADER);
    }

    #[test]
    fn a_frame_that_ends_off_resting_takes_one_restore() {
        // The Hi-Z pyramid rests sampled and the frame leaves it written, so the
        // executor owes it a GENERAL -> SHADER_READ_ONLY transition after the last
        // pass; without it next frame's producer names a layout the image is not in.
        let (old, new, ..) = vk_restore(
            GraphResourceClass::StorageImage,
            VkResting::Sampled,
            ResourceState::Write,
            ReadStages::empty(),
        )
        .expect("a real restore");
        assert_eq!(old, vk::ImageLayout::GENERAL);
        assert_eq!(new, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        // A frame that already ends sampled owes nothing.
        assert!(
            vk_restore(
                GraphResourceClass::StorageImage,
                VkResting::Sampled,
                ResourceState::Read,
                FRAG,
            )
            .is_none()
        );
        // Nor does one the frame never touched.
        assert!(
            vk_restore(
                GraphResourceClass::StorageImage,
                VkResting::Sampled,
                ResourceState::Undefined,
                ReadStages::empty(),
            )
            .is_none()
        );
        // A discard-resting resource never needs one: its next first use may name
        // UNDEFINED from whatever layout the frame left.
        assert!(
            vk_restore(
                GraphResourceClass::DepthTarget,
                VkResting::Discarded,
                ResourceState::Read,
                FRAG,
            )
            .is_none()
        );
        // Neither does a buffer: it has no layout at all.
        assert!(
            vk_restore(
                GraphResourceClass::StorageBuffer,
                VkResting::Sampled,
                ResourceState::Write,
                ReadStages::empty(),
            )
            .is_none()
        );
    }
}
