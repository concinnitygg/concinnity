// src/render_graph/transient.rs
//
// The slot list a backend's transient pool is built from, and the check that
// keeps that list sound.
//
// A pool takes one [`TransientSlot`] per aliasing-plan slot and makes one
// allocation for it, sized to its largest member, with every member placed at
// offset 0. Each member arrives as a resolved [`TransientTexture`] -- concrete
// pixel extents and the graph's own format / usage / sample count -- so a
// backend translates one description into its native descriptor rather than
// keeping a per-label table of its own. That is the point: a table restating
// what the graph already declares can disagree with it, and a disagreement
// about a format or an extent is silent.
//
// Pools are built at init / resize; graphs compile per frame. So the plan a
// pool is built from describes ONE graph, and reusing its grouping for the
// frames that follow is safe only while no slot has two members live at once in
// any of them. That is not free: a resource whose lifetime a pass *extends*
// looks more disjoint in a graph missing that pass, and there is no single
// maximal graph to plan against -- `unified_gbuffer_prepass`,
// `rt_reflections_enabled` and `upscale_enabled` substitute passes rather than
// adding them, so no one graph contains every lifetime. Three things cover it,
// and each catches what the others cannot:
//
//   1. The pool plans against its build configuration, and treats every input
//      it cannot rebuild on as live. [`planning_inputs`].
//   2. `slot_conflicts_over_reachable_graphs` in this module's tests sweeps the
//      reachable input space and fails on any slot with two overlapping
//      members.
//   3. Each executor asserts [`slot_conflicts`] per frame under
//      `debug_assertions`, over the graph it is about to run -- which covers
//      the combinations the sweep did not reach.

use super::alias::plan_aliasing_for;
use super::compile::CompiledGraph;
use super::frame::FrameGraphInputs;
use super::types::{ClearValue, PixelFormat, TextureUsage};

// One pooled transient, resolved against a concrete drawable extent. The
// backend translates this into its native texture descriptor; nothing here is
// backend-specific.
//
// `PartialEq` but not `Eq`: the clear value is floats.
#[derive(Clone, Debug, PartialEq)]
pub struct TransientTexture {
    // The graph label, which is also how a feature reads the texture back out
    // of the pool and how the barrier registry names it.
    pub label: &'static str,
    pub width: u32,
    pub height: u32,
    // 1 for a 2D texture, > 1 for a volume.
    pub depth: u32,
    pub format: PixelFormat,
    pub sample_count: u32,
    pub array_layers: u32,
    pub mip_levels: u32,
    pub usage: TextureUsage,
    // What the writing pass clears this target to. Carried through because
    // D3D12 bakes it into the resource at creation; see `TextureDesc::clear`.
    pub clear: ClearValue,
}

// One slot: the members that share a backing allocation, in the order they
// reuse it (lifetime-start). A single-member slot is a plain pooled target; a
// multi-member slot is a realised alias, and the order is what each backend's
// aliasing barriers are wired from.
#[derive(Clone, Debug, PartialEq)]
pub struct TransientSlot {
    pub members: Vec<TransientTexture>,
}

impl TransientSlot {
    pub fn labels(&self) -> Vec<&'static str> {
        self.members.iter().map(|m| m.label).collect()
    }
}

// The inputs a pool plans its slots against, given the configuration it was
// built for. `build` carries the flags the pool is rebuilt on (SSAO and bloom
// being switched on or off both rebuild it, as does a resize); every gated pass
// is forced on here, so no lifetime a pass would extend is missing from the
// graph the grouping is decided on.
//
// `composite_reads_ao` is ON, and the history is worth keeping. It used to be
// off, on the argument that it describes a different frame rather than a fuller
// one (it is reachable only in the occlusion view, which forces bloom off) and
// that planning against it would refuse the only aliasing the pool had. The
// second half has expired now that the G-buffer channels are pooled: there is
// plenty else to alias, and turning it on costs this plan nothing.
//
// The first half turned out to be a trap. Modelling `ao_output` as short-lived
// is only safe while nothing else is pooled around the reflection resolve --
// the moment a one-pass post-stack target joins the pool, the greedy pairs it
// with `ao_output` and the sweep reports the overlap the occlusion view really
// has. Measured, not argued: adding such a target made both sweeps fail here.
// Extending a lifetime is always the safe direction, so it stays on.
//
// `upscale_enabled` is NOT forced on, and a lifetime read out of this graph can
// therefore be shorter than the real maximal one. Upscale substitutes for
// TaaResolve, so forcing it would drop the TAA branch instead; neither branch
// dominates the other and one graph cannot hold both. The concrete casualty is
// `gbuffer_depth`, whose only consumer is the upscaler and which looks one-pass
// here -- see `the_prepass_depth_is_short_lived_only_in_the_planning_graph`.
//
// Nothing here is load-bearing on its own. What makes the grouping sound is the
// sweep over the reachable space in this module's tests plus each executor's
// per-frame assertion; if this graph ever becomes too permissive the sweep is
// what fails.
pub fn planning_inputs(build: &FrameGraphInputs) -> FrameGraphInputs {
    FrameGraphInputs {
        // `world_hidden` masks passes off rather than on, so leaving it false
        // keeps the richer graph.
        world_hidden: false,
        composite_reads_ao: true,
        shadow_enabled: true,
        bindless_cull_enabled: true,
        auto_exposure_enabled: true,
        velocity_enabled: true,
        taa_enabled: true,
        ssr_enabled: true,
        particles_enabled: true,
        fog_enabled: true,
        decals_enabled: true,
        ssr_prepass_enabled: true,
        transparent_enabled: true,
        lines_enabled: true,
        raymarch_enabled: true,
        two_pass_occlusion_enabled: true,
        ssgi_enabled: true,
        clustered_lighting_enabled: true,
        hiz_build_enabled: true,
        ..*build
    }
}

// The slots a pool built for `build` should allocate, over the transients
// `poolable` accepts, at `drawable_w` x `drawable_h`. Empty when the pool owns
// nothing. Returns `None` when the planning graph fails to compile, which is a
// caller's cue to fall back to one slot per managed resource rather than
// silently aliasing on a plan that was never made.
pub fn plan_transient_slots(
    build: &FrameGraphInputs,
    poolable: &dyn Fn(&str) -> bool,
    drawable_w: u32,
    drawable_h: u32,
) -> Option<Vec<TransientSlot>> {
    let graph = super::frame::build_frame_graph(&planning_inputs(build)).ok()?;
    let plan = plan_aliasing_for(&graph, drawable_w, drawable_h, poolable);
    Some(
        plan.slots
            .iter()
            .map(|slot| TransientSlot {
                members: slot
                    .members
                    .iter()
                    .map(|&idx| resolve(&graph, idx, drawable_w, drawable_h))
                    .collect(),
            })
            .collect(),
    )
}

// One graph resource as the pool must create it.
fn resolve(
    graph: &CompiledGraph,
    idx: usize,
    drawable_w: u32,
    drawable_h: u32,
) -> TransientTexture {
    let res = &graph.resources[idx];
    let desc = res
        .tex_desc
        .expect("the planner only places resources carrying a texture desc");
    let (width, height, depth) = desc.extent(drawable_w, drawable_h);
    TransientTexture {
        label: res.label,
        width,
        height,
        depth,
        format: desc.format,
        sample_count: desc.sample_count.max(1),
        array_layers: desc.array_layers.max(1),
        mip_levels: desc.mip_levels.max(1),
        usage: desc.usage,
        clear: desc.clear,
    }
}

// Two members of one slot whose lifetimes overlap in a graph, i.e. two
// resources that would be live at once on the same bytes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SlotConflict {
    pub slot: usize,
    pub a: &'static str,
    pub b: &'static str,
}

impl std::fmt::Display for SlotConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "slot {}: {} and {} are both live",
            self.slot, self.a, self.b
        )
    }
}

// Every pair of slot members whose `[first, last]` lifetimes overlap in
// `graph`. Empty when the grouping is sound for this graph, which is the
// invariant a pool's aliasing rests on: members of a slot share bytes, so two
// live at once means one is reading memory the other overwrote.
//
// Labels absent from `graph` are skipped -- a pool holds a resource for as long
// as its build configuration says so, and a frame that omits the pass writing
// it simply does not use it.
pub fn slot_conflicts(graph: &CompiledGraph, slots: &[Vec<&'static str>]) -> Vec<SlotConflict> {
    let lifetime = |label: &str| {
        graph
            .resources
            .iter()
            .find(|r| r.label == label)
            .map(|r| (r.lifetime.first, r.lifetime.last))
    };
    let mut conflicts = Vec::new();
    for (slot, members) in slots.iter().enumerate() {
        for (i, &a) in members.iter().enumerate() {
            let Some((a_first, a_last)) = lifetime(a) else {
                continue;
            };
            for &b in &members[i + 1..] {
                let Some((b_first, b_last)) = lifetime(b) else {
                    continue;
                };
                if a_first <= b_last && b_first <= a_last {
                    conflicts.push(SlotConflict { slot, a, b });
                }
            }
        }
    }
    conflicts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::frame::build_frame_graph;
    use concinnity_core::gfx::view_modes::{ShowFlags, ViewMode};

    // The flags a build configuration carries, i.e. the ones a pool is rebuilt
    // on. Everything else `planning_inputs` forces live.
    //
    // The G-buffer gate is one of them because `unified_gbuffer_prepass`
    // *substitutes* for the separate SsrPrepass / Velocity nodes rather than
    // adding to them, so `planning_inputs` cannot force it on the way it forces
    // the purely additive passes.
    fn build_inputs(ssao: bool, bloom: bool) -> FrameGraphInputs {
        build_inputs_with(ssao, bloom, true)
    }

    fn build_inputs_with(ssao: bool, bloom: bool, gbuffer: bool) -> FrameGraphInputs {
        let mut i = FrameGraphInputs::all_off();
        i.ssao_enabled = ssao;
        i.bloom_enabled = bloom;
        i.unified_gbuffer_prepass = gbuffer;
        i.velocity_enabled = gbuffer;
        i.hdr_width = 1920;
        i.hdr_height = 1080;
        i
    }

    // The pooled set the widest backend manages today (DirectX). Keeping this
    // in step with the backends' `pooled()` is what makes the sweep below mean
    // anything: a label pooled by a backend and missing here is a grouping no
    // headless test ever checks.
    //
    // `gbuffer_depth` is absent because no backend pools it: D3D12 needs a
    // typeless resource format for a shader-readable depth target, which
    // `PixelFormat` does not model.
    fn poolable(label: &str) -> bool {
        matches!(
            label,
            "ao_output"
                | "bloom_top"
                | "gbuffer_normal_depth"
                | "gbuffer_roughness"
                | "gbuffer_velocity"
        )
    }

    #[test]
    fn slots_carry_the_graphs_own_shape() {
        // The whole point of B1: the extent, format and usage a pool creates
        // come from the graph's desc, so there is no second description to
        // disagree with it.
        let slots = plan_transient_slots(&build_inputs(true, true), &poolable, 1920, 1080)
            .expect("planning graph compiles");
        let member = |label: &str| {
            slots
                .iter()
                .flat_map(|s| &s.members)
                .find(|m| m.label == label)
                .unwrap_or_else(|| panic!("{label} pooled"))
                .clone()
        };

        // `ao_output` is render-resolution R8.
        let ao = member("ao_output");
        assert_eq!((ao.width, ao.height, ao.depth), (1920, 1080, 1));
        assert_eq!(ao.format, PixelFormat::R8Unorm);
        assert_eq!((ao.sample_count, ao.array_layers, ao.mip_levels), (1, 1, 1));
        assert!(ao.usage.contains(TextureUsage::RENDER_TARGET));
        assert!(ao.usage.contains(TextureUsage::SHADER_READ));

        // `bloom_top` is half the *drawable* extent, which is what every
        // backend builds its bloom chain from -- not half the render
        // resolution, which differs from it under temporal upscaling.
        let bloom = member("bloom_top");
        assert_eq!((bloom.width, bloom.height), (960, 540));
        assert_eq!(bloom.format, PixelFormat::Rgba16Float);
    }

    #[test]
    fn bloom_top_follows_the_drawable_not_the_render_resolution() {
        // An upscaled configuration: render resolution well under the
        // drawable. `ao_output` follows the render resolution and `bloom_top`
        // the drawable, and the two must not track each other.
        let mut build = build_inputs(true, true);
        build.hdr_width = 1280;
        build.hdr_height = 720;
        let slots =
            plan_transient_slots(&build, &poolable, 2560, 1440).expect("planning graph compiles");
        let member = |label: &str| {
            slots
                .iter()
                .flat_map(|s| &s.members)
                .find(|m| m.label == label)
                .unwrap_or_else(|| panic!("{label} pooled"))
                .clone()
        };
        assert_eq!(
            (member("ao_output").width, member("ao_output").height),
            (1280, 720)
        );
        assert_eq!(
            (member("bloom_top").width, member("bloom_top").height),
            (1280, 720),
            "half of 2560x1440, which happens to equal the render resolution here"
        );
    }

    #[test]
    fn an_unpooled_label_gets_no_slot() {
        // A pool owns what its build configuration says it owns; the rest of
        // the graph's transients stay backend-owned and unplanned.
        let slots = plan_transient_slots(&build_inputs(true, true), &|_| false, 1920, 1080)
            .expect("planning graph compiles");
        assert!(slots.is_empty());

        let only_ao =
            plan_transient_slots(&build_inputs(true, true), &|l| l == "ao_output", 1920, 1080)
                .expect("planning graph compiles");
        assert_eq!(only_ao.len(), 1);
        assert_eq!(only_ao[0].labels(), vec!["ao_output"]);
    }

    #[test]
    fn ssao_off_leaves_only_the_bloom_target() {
        // No SSAO and no G-buffer pre-pass: `bloom_top` is the only pooled
        // resource the graph declares.
        let slots = plan_transient_slots(
            &build_inputs_with(false, true, false),
            &poolable,
            1920,
            1080,
        )
        .expect("planning graph compiles");
        let labels: Vec<&str> = slots.iter().flat_map(|s| s.labels()).collect();
        assert_eq!(labels, vec!["bloom_top"]);
    }

    #[test]
    fn nothing_pooled_when_neither_feature_is_built() {
        let slots = plan_transient_slots(
            &build_inputs_with(false, false, false),
            &poolable,
            1920,
            1080,
        )
        .expect("planning graph compiles");
        assert!(slots.is_empty());
    }

    #[test]
    fn the_gbuffer_gate_places_its_colour_targets() {
        // The gate is a build flag rather than something `planning_inputs`
        // forces, so a pool built without it must place none of them -- which is
        // what makes it safe for a backend to pool them only when the pre-pass
        // exists.
        let off =
            plan_transient_slots(&build_inputs_with(true, true, false), &poolable, 1920, 1080)
                .expect("planning graph compiles");
        let off_labels: Vec<&str> = off.iter().flat_map(|s| s.labels()).collect();
        assert!(
            !off_labels.contains(&"gbuffer_normal_depth"),
            "{off_labels:?}"
        );

        let on = plan_transient_slots(&build_inputs(true, true), &poolable, 1920, 1080)
            .expect("planning graph compiles");
        let on_labels: Vec<&str> = on.iter().flat_map(|s| s.labels()).collect();
        for want in [
            "gbuffer_normal_depth",
            "gbuffer_roughness",
            "gbuffer_velocity",
        ] {
            assert!(on_labels.contains(&want), "{want}: {on_labels:?}");
        }
    }

    #[test]
    fn slot_conflicts_reports_an_overlapping_pair() {
        // Negative control for the predicate the executors assert. `ao_output`
        // and `bloom_top` overlap in the occlusion view (which extends
        // `ao_output` to the composite) with bloom also on, so grouping them
        // must report. Without this the sweep below could pass on a predicate
        // that never reports anything.
        let mut i = FrameGraphInputs::all_off();
        i.ssao_enabled = true;
        i.bloom_enabled = true;
        i.composite_reads_ao = true;
        let graph = build_frame_graph(&i).expect("compiles");

        let grouped = vec![vec!["ao_output", "bloom_top"]];
        let conflicts = slot_conflicts(&graph, &grouped);
        assert_eq!(conflicts.len(), 1, "{conflicts:?}");
        assert_eq!(conflicts[0].slot, 0);

        // One per slot: split them and the same graph is sound.
        let split = vec![vec!["ao_output"], vec!["bloom_top"]];
        assert_eq!(slot_conflicts(&graph, &split), vec![]);
    }

    #[test]
    fn a_label_absent_from_the_graph_is_not_a_conflict() {
        // A pool holds `ao_output` for as long as SSAO is built; a frame whose
        // graph omits the SSAO pass simply does not use it, which is not a
        // reason to alarm.
        let graph = build_frame_graph(&FrameGraphInputs::all_off()).expect("compiles");
        let grouped = vec![vec!["ao_output", "bloom_top"]];
        assert_eq!(slot_conflicts(&graph, &grouped), vec![]);
    }

    #[test]
    fn planning_inputs_forces_the_gated_passes_on() {
        // A pass the planning graph omits is a lifetime it under-reports, so
        // every gated pass is on regardless of the build configuration.
        let planned = planning_inputs(&build_inputs(false, false));
        assert!(planned.ssgi_enabled);
        assert!(planned.transparent_enabled);
        assert!(planned.raymarch_enabled);
        assert!(!planned.world_hidden, "masking off passes is not the risk");
        // On, and it has to be: the occlusion view extends `ao_output` to the
        // Composite, past the reflection resolve. A plan made without it pairs
        // `ao_output` with `ssr_reflection`, which the sweep rejects.
        assert!(planned.composite_reads_ao);
        // The build flags pass through, because the pool *is* rebuilt on them.
        assert!(!planned.ssao_enabled);
        assert!(!planned.bloom_enabled);
        assert!(planning_inputs(&build_inputs(true, true)).ssao_enabled);
    }

    #[test]
    fn the_pool_actually_aliases_something() {
        // Anti-vacuity guard for the two sweeps below. Single-member slots are
        // trivially conflict-free, so a plan that stopped aliasing would leave
        // the sweeps passing while checking nothing.
        //
        // What aliases is worth reading: `bloom_top` (Bloom -> Composite, late)
        // pairs with whichever early resource is largest. The three G-buffer
        // colour targets do NOT alias each other -- every one is written by the
        // pre-pass and read by a late consumer, so their lifetimes span most of
        // the frame. That is why pooling this group reclaims far less than the
        // HDR / post groups will.
        let slots = plan_transient_slots(&build_inputs(true, true), &poolable, 1920, 1080)
            .expect("planning graph compiles");
        let shared: Vec<Vec<&'static str>> = slots
            .iter()
            .map(|s| s.labels())
            .filter(|l| l.len() > 1)
            .collect();
        assert!(
            !shared.is_empty(),
            "no slot aliases anything, so the sweeps check nothing: {:?}",
            labels_of(&slots)
        );
        assert!(
            shared.iter().any(|l| l.contains(&"bloom_top")),
            "bloom_top is the late resource that makes an alias possible: {:?}",
            labels_of(&slots)
        );
    }

    fn labels_of(slots: &[TransientSlot]) -> Vec<Vec<&'static str>> {
        slots.iter().map(|s| s.labels()).collect()
    }

    // Bytes a slot list costs (each slot sized to its largest member) against
    // what the same members would cost unaliased. This is the measurement that
    // decides whether a group migration is worth its wiring, and it runs
    // headlessly -- a group's footprint is NOT its saving, because members with
    // overlapping lifetimes each need their own slot.
    // The members already carry resolved pixel extents, so no drawable is
    // needed here.
    fn slot_bytes(slots: &[TransientSlot]) -> (u64, u64) {
        let member_bytes = |m: &TransientTexture| -> u64 {
            let texels = (m.width as u64) * (m.height as u64) * (m.depth.max(1) as u64);
            texels
                * m.format.bytes_per_texel() as u64
                * m.sample_count.max(1) as u64
                * m.array_layers.max(1) as u64
        };
        let mut aliased = 0;
        let mut unaliased = 0;
        for slot in slots {
            let mut largest = 0;
            for m in &slot.members {
                let b = member_bytes(m);
                unaliased += b;
                largest = largest.max(b);
            }
            aliased += largest;
        }
        (aliased, unaliased)
    }

    #[test]
    fn the_pooled_set_reclaims_what_the_plan_says() {
        // A regression guard on the *saving*, which is the point of the pool and
        // is otherwise invisible until someone measures a running frame: every
        // other test here would still pass if the plan quietly stopped aliasing.
        //
        // The number is small on purpose, and knowing why is what keeps the next
        // group migration honest. At 1920x1080 the plan is
        //   [gbuffer_normal_depth + bloom_top] [gbuffer_roughness]
        //   [gbuffer_velocity] [ao_output]
        // i.e. only `bloom_top` aliases at all. Every other pooled member is
        // live across most of the frame -- the G-buffer channels from the
        // pre-pass to their last consumer, `ao_output` from the SSAO node to
        // Main (to the Composite in the occlusion view) -- so they overlap each
        // other and each needs its own slot. Aliasing pays for *short* lifetimes,
        // and this renderer has few.
        let slots = plan_transient_slots(&build_inputs(true, true), &poolable, 1920, 1080)
            .expect("planning graph compiles");
        let (aliased, unaliased) = slot_bytes(&slots);
        let saved = unaliased - aliased;
        assert!(
            saved >= 3 * 1024 * 1024,
            "aliasing reclaims {} MiB (aliased {} MiB of {} MiB) from {:?}",
            saved / (1024 * 1024),
            aliased / (1024 * 1024),
            unaliased / (1024 * 1024),
            labels_of(&slots)
        );
    }

    #[test]
    fn the_prepass_depth_is_short_lived_only_in_the_planning_graph() {
        // A trap that already cost one wrong measurement, pinned so it cannot
        // cost another.
        //
        // `gbuffer_depth` looks like the best aliasing candidate in the whole
        // graph: the planning graph gives it a ONE-PASS lifetime, and reading
        // that at face value says it could share memory with `hdr_depth` and
        // reclaim ~8 MiB at 1080p. It cannot. Its only consumer is the temporal
        // upscaler, and `planning_inputs` cannot force `upscale_enabled` on
        // because Upscale *substitutes* for TaaResolve rather than adding to it
        // -- the same mutually-exclusive shape as `unified_gbuffer_prepass`. So
        // the planning graph models the TAA branch, in which nothing reads the
        // pre-pass depth at all.
        //
        // In the upscaling branch it is live from the pre-pass to Upscale, which
        // is past the point `hdr_depth` starts, so the two overlap and each
        // needs its own slot. Pool either of them on that reading and the sweeps
        // fail -- which is how this was caught.
        let build = build_inputs(true, true);
        let planned = build_frame_graph(&planning_inputs(&build)).expect("compiles");
        let life = |g: &CompiledGraph, label: &str| {
            let r = g
                .resources
                .iter()
                .find(|r| r.label == label)
                .unwrap_or_else(|| panic!("{label} declared"));
            (r.lifetime.first, r.lifetime.last)
        };
        let (first, last) = life(&planned, "gbuffer_depth");
        assert_eq!(
            first, last,
            "the planning graph gives the pre-pass depth a one-pass lifetime"
        );

        let mut upscaling = build;
        upscaling.upscale_enabled = true;
        let real = build_frame_graph(&planning_inputs(&upscaling)).expect("compiles");
        let (up_first, up_last) = life(&real, "gbuffer_depth");
        assert!(
            up_last > up_first,
            "the upscaler reads the pre-pass depth, so its real lifetime spans passes"
        );
        let (hdr_first, hdr_last) = life(&real, "hdr_depth");
        assert!(
            up_first <= hdr_last && hdr_first <= up_last,
            "the two depth targets overlap once the upscale branch is modelled: \
             gbuffer_depth [{up_first},{up_last}] vs hdr_depth [{hdr_first},{hdr_last}]"
        );
    }

    // Every gated flag, mirroring `validate::tests::FLAGS`: the sweep is only
    // as wide as this table, so extend it when a gated pass is added.
    type FlagSetter = fn(&mut FrameGraphInputs);
    const FLAGS: &[(&str, FlagSetter)] = &[
        ("shadow", |i| i.shadow_enabled = true),
        ("bindless_cull", |i| i.bindless_cull_enabled = true),
        ("auto_exposure", |i| i.auto_exposure_enabled = true),
        ("bloom", |i| i.bloom_enabled = true),
        ("velocity", |i| i.velocity_enabled = true),
        ("taa", |i| i.taa_enabled = true),
        ("ssr", |i| i.ssr_enabled = true),
        ("particles", |i| i.particles_enabled = true),
        ("fog", |i| i.fog_enabled = true),
        ("decals", |i| i.decals_enabled = true),
        ("ssr_prepass", |i| i.ssr_prepass_enabled = true),
        ("ssao", |i| i.ssao_enabled = true),
        ("upscale", |i| i.upscale_enabled = true),
        ("transparent", |i| i.transparent_enabled = true),
        ("lines", |i| i.lines_enabled = true),
        ("raymarch", |i| i.raymarch_enabled = true),
        ("two_pass_occlusion", |i| {
            i.two_pass_occlusion_enabled = true
        }),
        ("ssgi", |i| i.ssgi_enabled = true),
        ("rt_reflections", |i| i.rt_reflections_enabled = true),
        ("unified_gbuffer", |i| i.unified_gbuffer_prepass = true),
        ("world_hidden", |i| i.world_hidden = true),
        ("clustered_lighting", |i| {
            i.clustered_lighting_enabled = true
        }),
        ("composite_reads_ao", |i| i.composite_reads_ao = true),
        ("shadowed_spots", |i| i.shadowed_spot_count = 2),
        ("hiz_build", |i| i.hiz_build_enabled = true),
    ];

    // Assert the slots a pool built for `build` would allocate are conflict-free
    // in the graph `inputs` compiles to.
    fn assert_sound(build: &FrameGraphInputs, inputs: &FrameGraphInputs, what: &str) {
        let slots = plan_transient_slots(build, &poolable, 1920, 1080)
            .expect("the planning graph compiles");
        let grouped: Vec<Vec<&'static str>> = slots.iter().map(|s| s.labels()).collect();
        let graph = build_frame_graph(inputs)
            .unwrap_or_else(|e| panic!("graph failed to compile for {what}: {e}"));
        let conflicts = slot_conflicts(&graph, &grouped);
        assert!(
            conflicts.is_empty(),
            "aliasing conflict for {what}: {}",
            conflicts
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    #[test]
    fn slot_conflicts_over_reachable_graphs() {
        // The check the pool's whole aliasing rests on: for every build
        // configuration and every graph a session can reach from it, no slot
        // has two live members. `apply_view` is applied because the reachable
        // space is the *masked* one -- the occlusion view extends `ao_output`
        // to the composite but forces bloom off, so the pair that would
        // conflict is not actually reachable, and a sweep over raw flag
        // combinations would report a hazard no session can hit.
        let builds = [
            build_inputs(false, false),
            build_inputs(true, false),
            build_inputs(false, true),
            build_inputs(true, true),
        ];
        for build in &builds {
            for (i, (a_name, set_a)) in FLAGS.iter().enumerate() {
                for (b_name, set_b) in FLAGS.iter().skip(i) {
                    let mut inputs = FrameGraphInputs::all_off();
                    set_a(&mut inputs);
                    set_b(&mut inputs);
                    let what = format!("{a_name} + {b_name}");
                    assert_sound(build, &inputs, &what);
                    for mode in ViewMode::ALL {
                        for show in [ShowFlags::all(), ShowFlags(0)] {
                            let masked = crate::render_graph::apply_view(&inputs, mode, show);
                            assert_sound(
                                build,
                                &masked,
                                &format!("{what} under {mode:?} / {show:?}"),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn slot_conflicts_over_the_fully_loaded_graph_in_every_view() {
        // The wide end: every pass on at once, swept across every view mode and
        // every show-flag subset, which is where a mask that turns one pass off
        // while leaving a lifetime-extending one on would show up.
        let mut loaded = FrameGraphInputs::all_off();
        for (name, set) in FLAGS {
            if *name != "world_hidden" {
                set(&mut loaded);
            }
        }
        let build = build_inputs(true, true);
        for mode in ViewMode::ALL {
            for bits in 0..(1u32 << ShowFlags::LABELED.len()) {
                let show = ShowFlags(bits);
                let masked = crate::render_graph::apply_view(&loaded, mode, show);
                assert_sound(
                    &build,
                    &masked,
                    &format!("loaded under {mode:?} / {bits:b}"),
                );
            }
        }
    }
}
