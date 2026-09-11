// src/render/render_graph/alias.rs
//
// Transient-resource memory aliasing planner. The compile pass records which
// passes touch each resource and the schedule's happens-before relation over
// those passes; this module turns them into a physical-memory plan: transient
// resources the schedule never leaves live at the same time can share one
// backing allocation.
//
// The planner is backend-agnostic and pure. It only decides *which resources
// share a slot* and *how big each slot must be*; the per-backend executor
// realises the plan (allocates a pool, creates the aliased resources, binds
// them, and inserts aliasing barriers at the slot's reuse boundaries). This
// mirrors how the graph plans barriers (`barriers_before`) while each backend
// emits them.
//
// Only `Transient` textures are candidates: `Imported` resources are
// engine-owned and outlive the frame (cross-frame TAA history, the resting
// shadow map, the `scene_pre_taa` alias), so the graph never reuses their
// memory. Buffers are not aliased yet (none of today's graph buffers are large
// or short-lived enough to matter).
//
// The packing is a linear scan over lifetime-start order (the classic
// interval-graph greedy, kept for the slot *count*): each resource takes the
// first compatible slot every member of which the schedule orders strictly
// before it, else opens a new slot. A slot is sized to its largest member.
// Compatibility is [`SlotClass`]: resources that differ on it never share,
// whatever their lifetimes.
//
// The disjointness test is the schedule's happens-before relation
// (`CompiledGraph::resource_precedes`), not a `[first, last]` index comparison.
// An index interval is a disjointness test only under a total order: once a
// compute pass runs on the async queue, two resources with disjoint index
// ranges can be concurrently live on two queues. The relation the planner asks
// is "does every reader and writer of A complete before every reader and writer
// of B starts?", answered over the dependency DAG plus each queue's own serial
// order -- so two graphics passes with no data dependency are still ordered
// (their queue runs them in order) while a graphics pass and an unsynchronised
// async pass are not.

use super::compile::CompiledGraph;
use super::types::{ResourceOrigin, TextureDesc};
use alloc::vec;
use alloc::vec::Vec;

// The memory class two resources must agree on before they can share a slot.
// Not the whole desc: differing extents and formats are fine (a slot is sized
// to its largest member and each member is created with its own descriptor),
// but these two change what kind of allocation the backend must make.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct SlotClass {
    // Depth targets and colour targets take different heap flags / memory
    // types on every backend.
    depth: bool,
    // A multisample target's layout is not the single-sample one, so a 4x
    // attachment and a resolved target cannot share bytes even across
    // disjoint lifetimes.
    sample_count: u32,
}

impl SlotClass {
    fn of(desc: &TextureDesc) -> Self {
        Self {
            depth: desc.format.is_depth(),
            sample_count: desc.sample_count.max(1),
        }
    }
}

// One physical memory slot shared by one or more transient resources the
// schedule never leaves simultaneously live. `byte_size` is the max footprint of its members
// (the allocation the backend must make); `members` are resource indices into
// `CompiledGraph.resources`, in assignment order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AliasSlot {
    pub byte_size: u64,
    pub members: Vec<usize>,
}

// The computed aliasing plan for one compiled graph at one drawable extent.
#[derive(Debug, Clone)]
pub(crate) struct AliasPlan {
    // Physical slots; the backend allocates one pool entry per slot.
    pub slots: Vec<AliasSlot>,
    // Per-resource slot index, indexed by `ResourceId`. `None` for every
    // resource the planner does not place (imported, buffer, or a transient
    // with no texture desc). Measured output: the executor consumes `slots`,
    // this module's tests assert the rest.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "measured output asserted by this module's tests")
    )]
    pub assignment: Vec<Option<usize>>,
    // Total bytes the slots occupy (the aliased footprint).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "measured output asserted by this module's tests")
    )]
    pub aliased_bytes: u64,
    // Total bytes the same resources would occupy with no aliasing (one
    // allocation each).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "measured output asserted by this module's tests")
    )]
    pub unaliased_bytes: u64,
}

impl AliasPlan {
    // Bytes saved by aliasing: the unaliased footprint minus the slot
    // footprint. Zero when no two transients are separable.
    #[cfg(test)]
    pub(crate) fn saved_bytes(&self) -> u64 {
        self.unaliased_bytes.saturating_sub(self.aliased_bytes)
    }
}

// Compute the aliasing plan over the transients `poolable` accepts by label,
// i.e. the ones a backend's pool actually owns, at the given drawable extent.
// Pure: the same graph + extent + predicate always produces the same plan. See
// the module header for the packing strategy.
//
// Restricting the *candidate set*
// rather than filtering the finished plan is what lets a pool plan against the
// real frame graph: the greedy then packs the pooled resources against each
// other, instead of pairing one of them with an unpooled resource and leaving
// the other alone. Everything `poolable` rejects is left unplaced, exactly like
// an imported resource.
pub(crate) fn plan_aliasing_for(
    graph: &CompiledGraph,
    drawable_w: u32,
    drawable_h: u32,
    poolable: &dyn Fn(&str) -> bool,
) -> AliasPlan {
    // Gather the transient texture candidates with their lifetime start + size.
    struct Cand {
        idx: usize,
        first: usize,
        size: u64,
        class: SlotClass,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for (idx, res) in graph.resources.iter().enumerate() {
        if res.origin != ResourceOrigin::Transient || !poolable(res.label) {
            continue;
        }
        let Some(desc) = res.tex_desc else {
            continue;
        };
        cands.push(Cand {
            idx,
            first: res.lifetime.first,
            size: desc.byte_size(drawable_w, drawable_h),
            class: SlotClass::of(&desc),
        });
    }

    let unaliased_bytes: u64 = cands.iter().map(|c| c.size).sum();

    // Process in lifetime-start order (ties by resource index for determinism),
    // which keeps the greedy's slot count and makes the plan a pure function of
    // the graph. A partial order has no single "free at" instant per slot, so
    // the compatibility test below asks the schedule about every member.
    cands.sort_by(|a, b| a.first.cmp(&b.first).then(a.idx.cmp(&b.idx)));

    // Slot bookkeeping kept alongside the public `AliasSlot`.
    struct SlotMeta {
        class: SlotClass,
        byte_size: u64,
        members: Vec<usize>,
    }
    let mut slots: Vec<SlotMeta> = Vec::new();
    let mut assignment: Vec<Option<usize>> = vec![None; graph.resources.len()];

    for c in &cands {
        // First compatible slot every member of which the schedule orders
        // strictly before this candidate. Every member, not just the latest:
        // under a partial order a slot's occupants are not themselves totally
        // ordered, so there is no single last one to compare against.
        let chosen = slots.iter().position(|s| {
            s.class == c.class && s.members.iter().all(|&m| graph.resource_precedes(m, c.idx))
        });
        let si = match chosen {
            Some(si) => {
                let s = &mut slots[si];
                s.byte_size = s.byte_size.max(c.size);
                s.members.push(c.idx);
                si
            }
            None => {
                slots.push(SlotMeta {
                    class: c.class,
                    byte_size: c.size,
                    members: vec![c.idx],
                });
                slots.len() - 1
            }
        };
        assignment[c.idx] = Some(si);
    }

    let aliased_bytes: u64 = slots.iter().map(|s| s.byte_size).sum();
    let slots: Vec<AliasSlot> = slots
        .into_iter()
        .map(|s| AliasSlot {
            byte_size: s.byte_size,
            members: s.members,
        })
        .collect();

    AliasPlan {
        slots,
        assignment,
        aliased_bytes,
        unaliased_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::render_graph::builder::GraphBuilder;
    use crate::render::render_graph::frame::{FrameGraphInputs, build_frame_graph};
    use crate::render::render_graph::passes::PassId;
    use crate::render::render_graph::schedule::PassQueue;
    use crate::render::render_graph::transient::slot_conflicts;
    use crate::render::render_graph::types::{
        BufferDesc, BufferUsage, PassKind, PixelFormat, TextureDesc, TextureSize, TextureUsage,
    };

    // The plan over every transient in the graph. Only the tests want this:
    // production planning always restricts to one pool's own label set, so the
    // unrestricted packing is exercised here rather than exported.
    fn plan_aliasing(graph: &CompiledGraph, drawable_w: u32, drawable_h: u32) -> AliasPlan {
        plan_aliasing_for(graph, drawable_w, drawable_h, &|_| true)
    }

    // The production defaults at a 1080p drawable, which the packing
    // assertions below are sized against.
    fn all_off() -> FrameGraphInputs {
        FrameGraphInputs {
            hdr_width: 1920,
            hdr_height: 1080,
            ..FrameGraphInputs::all_off()
        }
    }

    // A transient texture desc of the given format at full drawable size.
    fn tex(format: PixelFormat) -> TextureDesc {
        TextureDesc::texture_2d(
            TextureSize::Drawable,
            TextureSize::Drawable,
            format,
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        )
    }

    // Bytes a full-drawable texture of `format` occupies at 100x100.
    fn size_at_100(format: PixelFormat) -> u64 {
        100 * 100 * format.bytes_per_texel() as u64
    }

    #[test]
    fn byte_size_resolves_drawable_and_format() {
        let d = tex(PixelFormat::Rgba16Float);
        assert_eq!(d.byte_size(64, 32), 64 * 32 * 8);
        // Half-res quarter-byte format.
        let half_r8 = TextureDesc::texture_2d(
            TextureSize::DrawableScaled(0.5),
            TextureSize::DrawableScaled(0.5),
            PixelFormat::R8Unorm,
            TextureUsage::SHADER_READ,
        );
        assert_eq!(half_r8.byte_size(64, 64), 32 * 32);
        // Sample count + array layers multiply.
        let msaa = tex(PixelFormat::Rgba8Unorm)
            .with_sample_count(4)
            .with_array_layers(2);
        assert_eq!(msaa.byte_size(10, 10), 10 * 10 * 4 * 4 * 2);
    }

    #[test]
    fn disjoint_transients_share_one_slot_sized_to_largest() {
        // Main writes `a` (R8, small), read by SsaoBlur; Fog writes `b` (RGBA16,
        // big), read by Composite. `a` lifetime [0,1], `b` [2,3] -> disjoint, so
        // they pack into one slot sized to the larger (`b`).
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex(PixelFormat::R8Unorm));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float));
        let a1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(a);
        g.add_pass(PassId::SsaoBlur, PassKind::Render)
            .read_texture(a1);
        let b1 = g.add_pass(PassId::Fog, PassKind::Render).write_texture(b);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(b1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 1, "disjoint a + b share one slot");
        assert_eq!(
            plan.slots[0].byte_size,
            size_at_100(PixelFormat::Rgba16Float)
        );
        assert_eq!(plan.slots[0].members.len(), 2);
        // Both resources point at slot 0.
        assert_eq!(plan.assignment[a.resource.index()], Some(0));
        assert_eq!(plan.assignment[b.resource.index()], Some(0));
        // Saved = the smaller resource's footprint (it reuses b's slot).
        assert_eq!(plan.saved_bytes(), size_at_100(PixelFormat::R8Unorm));
        assert_eq!(
            plan.unaliased_bytes,
            size_at_100(PixelFormat::R8Unorm) + size_at_100(PixelFormat::Rgba16Float)
        );
    }

    #[test]
    fn overlapping_transients_get_separate_slots() {
        // Main writes both `a` and `b`; Composite reads both. Their lifetimes
        // both span [0,1], so they overlap and cannot share memory.
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex(PixelFormat::Rgba16Float));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float));
        let (a1, b1) = {
            let mut p = g.add_pass(PassId::Main, PassKind::Render);
            (p.write_texture(a), p.write_texture(b))
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(a1)
            .read_texture(b1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 2, "overlapping a + b need two slots");
        assert_eq!(plan.saved_bytes(), 0);
    }

    #[test]
    fn touching_lifetimes_do_not_alias() {
        // `a` lifetime [0,1], `b` [1,2]: they share pass 1 (a read there, b
        // written there), so the strict `free_at < first` rule keeps them apart.
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex(PixelFormat::Rgba16Float));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float));
        let a1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(a);
        let b1 = {
            let mut p = g.add_pass(PassId::Decals, PassKind::Render);
            p.read_texture(a1);
            p.write_texture(b)
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(b1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 2, "touching lifetimes overlap at pass 1");
    }

    #[test]
    fn depth_and_colour_do_not_share() {
        // Two disjoint transients, one depth one colour. Even though their
        // lifetimes don't overlap, the planner keeps them in separate pools
        // (different backend memory class).
        let mut g = GraphBuilder::new();
        let depth = g.create_texture("depth", tex(PixelFormat::Depth32Float));
        let colour = g.create_texture("colour", tex(PixelFormat::Rgba16Float));
        let d1 = g
            .add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(depth);
        g.add_pass(PassId::SsaoBlur, PassKind::Render)
            .read_texture(d1);
        let c1 = g
            .add_pass(PassId::Fog, PassKind::Render)
            .write_texture(colour);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(c1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 2, "depth + colour never share a slot");
        assert_eq!(plan.saved_bytes(), 0);
    }

    #[test]
    fn differing_sample_counts_do_not_share() {
        // The MSAA attachment and the resolved target are both colour, and here
        // their lifetimes are disjoint -- but a 4x attachment's layout is not
        // the single-sample one, so they must not land on the same bytes.
        let mut g = GraphBuilder::new();
        let multi = g.create_texture("multi", tex(PixelFormat::Rgba16Float).with_sample_count(4));
        let single = g.create_texture("single", tex(PixelFormat::Rgba16Float));
        let m1 = g
            .add_pass(PassId::Main, PassKind::Render)
            .write_texture(multi);
        g.add_pass(PassId::SsaoBlur, PassKind::Render)
            .read_texture(m1);
        let s1 = g
            .add_pass(PassId::Fog, PassKind::Render)
            .write_texture(single);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(s1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(
            plan.slots.len(),
            2,
            "a multisample target never shares with a single-sample one"
        );
        assert_eq!(plan.saved_bytes(), 0);
        // Same lifetimes at a matching sample count *do* share, so the test
        // above is measuring the sample-count axis and not the lifetimes.
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex(PixelFormat::Rgba16Float).with_sample_count(4));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float).with_sample_count(4));
        let a1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(a);
        g.add_pass(PassId::SsaoBlur, PassKind::Render)
            .read_texture(a1);
        let b1 = g.add_pass(PassId::Fog, PassKind::Render).write_texture(b);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(b1)
            .presents();
        let g = g.compile().expect("compiles");
        assert_eq!(plan_aliasing(&g, 100, 100).slots.len(), 1);
    }

    #[test]
    fn imported_resources_are_not_placed() {
        // An imported texture is engine-owned and outlives the frame; the
        // planner never assigns it a slot even if its lifetime would fit one.
        let mut g = GraphBuilder::new();
        let imported = g.import_texture("imported", tex(PixelFormat::Rgba16Float));
        let transient = g.create_texture("transient", tex(PixelFormat::Rgba16Float));
        let i1 = {
            let mut p = g.add_pass(PassId::Main, PassKind::Render);
            p.read_texture(imported);
            p.write_texture(transient)
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(i1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.assignment[imported.resource.index()], None);
        assert!(plan.assignment[transient.resource.index()].is_some());
    }

    #[test]
    fn three_disjoint_chain_packs_into_one_slot() {
        // Three RGBA16 transients with gapped, non-touching lifetimes:
        // a [0,1] (Main writes, Decals reads), b [2,3] (Fog writes, Particles
        // reads), c [4,5] (SsrResolve writes, Composite reads). Each begins
        // strictly after the previous ends, so all three reuse one slot:
        // saved = two of the three footprints. (A write-then-immediately-read
        // chain a->b->c would instead TOUCH at the shared pass and not alias;
        // the gaps here are deliberate.)
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex(PixelFormat::Rgba16Float));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float));
        let c = g.create_texture("c", tex(PixelFormat::Rgba16Float));
        let a1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(a);
        g.add_pass(PassId::Decals, PassKind::Render)
            .read_texture(a1);
        let b1 = g.add_pass(PassId::Fog, PassKind::Render).write_texture(b);
        g.add_pass(PassId::ParticlesDraw, PassKind::Render)
            .read_texture(b1);
        let c1 = g
            .add_pass(PassId::SsrResolve, PassKind::Render)
            .write_texture(c);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(c1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 1);
        assert_eq!(plan.slots[0].members.len(), 3);
        assert_eq!(
            plan.saved_bytes(),
            2 * size_at_100(PixelFormat::Rgba16Float)
        );
    }

    // Two write-only transients, one written by the graph's first pass and one by
    // its second, with a buffer producer / consumer pair beside them so the
    // first pass can earn the async-compute queue. `kind` is what that first
    // pass declares, which is the only difference between the two graphs the
    // test below compares.
    fn split_transient_graph(kind: PassKind) -> CompiledGraph {
        let mut g = GraphBuilder::new();
        let args = g.create_buffer(
            "draw_args",
            BufferDesc {
                size_bytes: None,
                usage: BufferUsage::STORAGE,
            },
        );
        let a = g.create_texture("a", tex(PixelFormat::Rgba16Float));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float));
        let scene = g.create_texture("scene", tex(PixelFormat::Rgba16Float));

        let args1 = {
            let mut p = g.add_pass(PassId::Cull, kind);
            let _ = p.write_texture(a);
            p.write_buffer(args)
        };
        g.add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(b);
        let scene1 = {
            let mut p = g.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.write_texture(scene)
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();
        g.compile().expect("compiles")
    }

    #[test]
    fn a_transient_on_the_async_queue_does_not_alias_a_concurrent_one() {
        // The case the index interval could not see. `a` is touched only by the
        // first pass and `b` only by the second, so their index ranges are
        // disjoint either way -- but with the first pass on the async queue
        // nothing orders the two, and the bytes must not be shared.
        let async_graph = split_transient_graph(PassKind::Compute);
        let a = async_graph
            .resources
            .iter()
            .position(|r| r.label == "a")
            .expect("a present");
        let b = async_graph
            .resources
            .iter()
            .position(|r| r.label == "b")
            .expect("b present");
        assert_eq!(
            async_graph.pass(PassId::Cull).expect("present").queue,
            PassQueue::AsyncCompute,
            "the control needs the first pass on the async queue"
        );
        assert!(
            async_graph.resources[a].lifetime.last < async_graph.resources[b].lifetime.first,
            "the index intervals are disjoint, so only the schedule can separate these"
        );
        assert!(async_graph.resources_may_be_live_together(a, b));
        assert_eq!(
            plan_aliasing(&async_graph, 100, 100).slots.len(),
            2,
            "a concurrent pair must not share bytes"
        );
        // The pool-side check reads the same relation, so a grouping that pairs
        // them is reported rather than silently corrupting one of them.
        let conflicts = slot_conflicts(&async_graph, &[vec!["a", "b"]]);
        assert_eq!(conflicts.len(), 1, "{conflicts:?}");

        // Same graph with the first pass declared as a render pass: everything is
        // on the graphics queue, which runs its passes in order, so the pair is
        // separable again and packs into one slot. That is what makes this test
        // measure the queue split and not the two labels.
        let serial_graph = split_transient_graph(PassKind::Render);
        assert!(!serial_graph.resources_may_be_live_together(a, b));
        assert_eq!(plan_aliasing(&serial_graph, 100, 100).slots.len(), 1);
        assert!(slot_conflicts(&serial_graph, &[vec!["a", "b"]]).is_empty());
    }

    #[test]
    fn real_frame_graph_aliases_some_transients() {
        // The actual per-frame graph with SSAO + bloom on: `ao_output` (early,
        // SsaoBlur -> Main) and `bloom_top` (late, Bloom -> Composite) are both
        // transient with disjoint lifetimes, so the planner saves at least
        // `ao_output`'s footprint. Guards the import->create reclassification +
        // the end-to-end planner against a representative graph.
        let mut inputs = all_off();
        inputs.ssao_enabled = true;
        inputs.bloom_enabled = true;
        let g = build_frame_graph(&inputs).expect("frame graph compiles");
        let plan = plan_aliasing(&g, 1920, 1080);
        assert!(
            plan.saved_bytes() > 0,
            "ao_output + bloom_top have disjoint lifetimes and should alias"
        );
    }
}
