// src/render_graph/validate.rs
//
// Barrier-coverage check over a `CompiledGraph`. The compile pass derives
// `barriers_before` by walking each resource's timeline; this module replays
// those barriers in execution order and checks the resulting state against what
// each pass's read / write declarations require. The two directions are
// structurally independent -- a per-resource timeline versus a per-pass replay --
// so a deriver bug (a dropped transition, a mis-ordered run, a read-run stage
// union that misses a consumer) shows up as a gap here.
//
// The check is pure and GPU-free, so it runs both as a headless sweep over the
// `FrameGraphInputs` space and as a per-frame `debug_assertions` assertion inside
// each backend executor, where it covers the graphs a test sweep never builds.

use super::compile::CompiledGraph;
use super::passes::PassId;
use super::types::{ReadStages, ResourceState};

// What kind of coverage a pass is missing for one resource.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GapKind {
    // The pass reads the resource, but the replayed barrier state is not `Read`:
    // no transition made the producing write visible to this consumer.
    UncoveredRead,
    // The pass writes the resource, but the replayed barrier state is not `Write`:
    // no transition opened it for writing.
    UncoveredWrite,
    // The resource is in `Read`, but the barrier that opened the read run does not
    // name this consumer's shader stage, so the producing write was never made
    // visible to it.
    MissingReadStage,
}

// One pass / resource pair whose declared access is not covered by a barrier.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BarrierGap {
    pub pass: PassId,
    pub resource_label: &'static str,
    pub kind: GapKind,
}

impl std::fmt::Display for BarrierGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.kind {
            GapKind::UncoveredRead => "reads",
            GapKind::UncoveredWrite => "writes",
            GapKind::MissingReadStage => "reads (stage not in the run union)",
        };
        write!(f, "pass {:?} {} {}", self.pass, what, self.resource_label)
    }
}

// Every declared access in `graph` that no barrier covers, in execution order.
// Empty for a correctly compiled graph.
pub fn barrier_coverage_gaps(graph: &CompiledGraph) -> Vec<BarrierGap> {
    gaps_over(graph, &|_| true)
}

// The same check restricted to the resources `driven[resource_index]` marks, i.e.
// the ones a backend executor resolves to a native target and emits transitions
// for. A backend calls this on the graphs a real session builds, so it covers the
// input combinations a headless sweep does not reach, and it asserts specifically
// that everything the backend claims to drive is fully covered. Resources outside
// the driven set keep whatever synchronisation their encoder owns and are skipped.
pub fn barrier_coverage_gaps_for_driven(graph: &CompiledGraph, driven: &[bool]) -> Vec<BarrierGap> {
    gaps_over(graph, &|i| driven.get(i).copied().unwrap_or(false))
}

// The access each resource is left in once the graph's last barrier for it has
// run, indexed by resource id: the state plus the stage union that barrier
// carried, since a `Read`'s native state can depend on its consuming stages.
// `(Undefined, empty)` for a resource no barrier touches.
//
// A backend pairs this with its own state translation to check the cross-frame
// contract: a resource whose first-use transition names a resting state must
// actually end the frame in it, or the next frame's producer barrier declares a
// source state the resource is not in.
pub fn final_states(graph: &CompiledGraph) -> Vec<(ResourceState, ReadStages)> {
    let mut state = vec![(ResourceState::Undefined, ReadStages::empty()); graph.resources.len()];
    for pass in &graph.passes {
        for op in &pass.barriers_before {
            state[op.resource_index()] = (op.to_state(), op.read_stages());
        }
    }
    state
}

fn gaps_over(graph: &CompiledGraph, driven: &dyn Fn(usize) -> bool) -> Vec<BarrierGap> {
    let mut state = vec![ResourceState::Undefined; graph.resources.len()];
    // Stage union carried by the barrier that opened each resource's current read
    // run; meaningless unless that resource is in `Read`.
    let mut run_stages = vec![ReadStages::empty(); graph.resources.len()];
    let mut gaps = Vec::new();

    for pass in &graph.passes {
        for op in &pass.barriers_before {
            let i = op.resource_index();
            state[i] = op.to_state();
            if op.to_state() == ResourceState::Read {
                run_stages[i] = op.read_stages();
            }
        }

        let stage = ReadStages::for_pass_kind(pass.kind);
        // A pass that writes a resource leaves it in `Write` whether or not it also
        // reads it, so writes are checked first and shadow the read check.
        for w in &pass.writes {
            let i = w.resource_index();
            if !driven(i) {
                continue;
            }
            if state[i] != ResourceState::Write {
                gaps.push(BarrierGap {
                    pass: pass.id,
                    resource_label: graph.resources[i].label,
                    kind: GapKind::UncoveredWrite,
                });
            }
        }
        for r in &pass.reads {
            let i = r.resource_index();
            if !driven(i) || pass.writes.iter().any(|w| w.resource_index() == i) {
                continue;
            }
            if state[i] != ResourceState::Read {
                gaps.push(BarrierGap {
                    pass: pass.id,
                    resource_label: graph.resources[i].label,
                    kind: GapKind::UncoveredRead,
                });
            } else if !run_stages[i].contains(stage) {
                gaps.push(BarrierGap {
                    pass: pass.id,
                    resource_label: graph.resources[i].label,
                    kind: GapKind::MissingReadStage,
                });
            }
        }
    }

    gaps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::builder::GraphBuilder;
    use crate::render_graph::frame::{FrameGraphInputs, build_frame_graph};
    use crate::render_graph::types::{
        PassKind, PixelFormat, TextureDesc, TextureSize, TextureUsage,
    };

    // Every gated flag on `FrameGraphInputs`, so the sweep below can name the
    // combination that failed rather than reporting an opaque struct. Extend when a
    // gated pass is added; the sweep is only as wide as this table.
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
    ];

    // Compile the graph for `combo` and assert it has no barrier gaps.
    fn assert_covered(combo: &[usize]) {
        let mut inputs = FrameGraphInputs::all_off();
        for &f in combo {
            FLAGS[f].1(&mut inputs);
        }
        let names: Vec<&str> = combo.iter().map(|&f| FLAGS[f].0).collect();
        let graph = build_frame_graph(&inputs)
            .unwrap_or_else(|e| panic!("graph failed to compile for {names:?}: {e}"));
        let gaps = barrier_coverage_gaps(&graph);
        assert!(
            gaps.is_empty(),
            "barrier gaps for {names:?}: {}",
            gaps.iter()
                .map(|g| g.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    #[test]
    fn every_single_and_paired_flag_graph_is_fully_covered() {
        // Exhaustive over the whole flag space is 2^24 graphs; singles + pairs is
        // ~300 and catches the interaction bugs that matter (a pass inserted between
        // a producer and its consumer, a substituted pass -- unified G-buffer for
        // the split pre-passes, RT for SSR -- rerouting a read). The all-off and
        // all-on ends are covered separately below.
        assert_covered(&[]);
        for a in 0..FLAGS.len() {
            assert_covered(&[a]);
            for b in (a + 1)..FLAGS.len() {
                assert_covered(&[a, b]);
            }
        }
    }

    #[test]
    fn the_driven_subset_check_ignores_resources_outside_it() {
        // A backend drives only the resources its registry resolves; the rest keep
        // whatever synchronisation their encoder owns. Stripping a barrier for an
        // undriven resource must stay silent, and the same strip on a driven one
        // must report -- otherwise the subset check would either alarm on every
        // partially-migrated frame or never alarm at all.
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex());
        let b = g.create_texture("b", tex());
        let (a1, b1) = {
            let mut p = g.add_pass(PassId::Main, PassKind::Render);
            (p.write_texture(a), p.write_texture(b))
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(a1)
            .read_texture(b1)
            .presents();
        let mut g = g.compile().expect("compiles");
        let composite = g.passes.len() - 1;
        g.passes[composite].barriers_before.clear();

        let only_a = {
            let mut d = vec![false; g.resources.len()];
            d[a.resource.index()] = true;
            d
        };
        let gaps = barrier_coverage_gaps_for_driven(&g, &only_a);
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert_eq!(gaps[0].resource_label, "a");

        let none = vec![false; g.resources.len()];
        assert_eq!(barrier_coverage_gaps_for_driven(&g, &none), vec![]);
    }

    #[test]
    fn every_frame_graph_resource_classifies_as_its_backend_expects() {
        // Both explicit backends now take a resource's barrier class from the
        // graph instead of restating it, so a desc edit here silently changes what
        // transitions they emit. These are the classes the executors' translators
        // are written against; changing one means changing the translator too.
        use crate::render_graph::types::GraphResourceClass as C;

        let mut inputs = FrameGraphInputs::all_off();
        for (name, set) in FLAGS {
            if *name != "world_hidden" {
                set(&mut inputs);
            }
        }
        let graph = build_frame_graph(&inputs).expect("compiles");

        let expected = [
            ("draw_args", C::IndirectBuffer),
            ("draw_args2", C::IndirectBuffer),
            ("cull_status", C::UnorderedBuffer),
            ("cluster_light_list", C::StorageBuffer),
            ("ao_output", C::ColorTarget),
            ("shadow_map", C::DepthTarget),
            ("spot_shadow_map", C::DepthTarget),
            ("fog_froxel_volume", C::StorageImage),
            ("hdr_depth", C::DepthTarget),
            ("hdr_color", C::ColorTarget),
            ("hiz_pyramid", C::StorageImage),
        ];
        for (label, want) in expected {
            let res = graph
                .resources
                .iter()
                .find(|r| r.label == label)
                .unwrap_or_else(|| panic!("{label} missing from the fully-loaded graph"));
            assert_eq!(res.class(), Some(want), "{label}");
        }

        // Nothing in the graph may be unclassifiable: a resource with no class
        // gets no registry entry and so silently loses its barriers.
        for res in &graph.resources {
            assert!(res.class().is_some(), "{} has no class", res.label);
        }
    }

    #[test]
    fn the_fully_loaded_graph_is_covered() {
        // Every gated pass at once except `world_hidden`, which masks them all off
        // (its collapsed graph is covered as a single above).
        let all: Vec<usize> = (0..FLAGS.len())
            .filter(|&f| FLAGS[f].0 != "world_hidden")
            .collect();
        assert_covered(&all);
    }

    fn tex() -> TextureDesc {
        TextureDesc {
            width: TextureSize::Drawable,
            height: TextureSize::Drawable,
            format: PixelFormat::Rgba16Float,
            sample_count: 1,
            array_layers: 1,
            usage: TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        }
    }

    #[test]
    fn a_well_formed_graph_has_no_gaps() {
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let g = g.compile().expect("compiles");
        assert_eq!(barrier_coverage_gaps(&g), vec![]);
    }

    #[test]
    fn a_mixed_stage_read_run_is_covered_for_both_consumers() {
        // The case the read-stage union exists for: one write consumed by a compute
        // pass and a render pass. A single producer barrier must name both stages,
        // or the second consumer races the write.
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(t1);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let g = g.compile().expect("compiles");
        assert_eq!(barrier_coverage_gaps(&g), vec![]);
    }

    #[test]
    fn a_dropped_barrier_is_reported() {
        // Negative control: strip the consumer's barrier and the replay must flag
        // the read it no longer covers. Without this the test above could pass on a
        // validator that never reports anything.
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let mut g = g.compile().expect("compiles");
        let composite = g.passes.len() - 1;
        g.passes[composite].barriers_before.clear();

        let gaps = barrier_coverage_gaps(&g);
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert_eq!(gaps[0].kind, GapKind::UncoveredRead);
        assert_eq!(gaps[0].pass, PassId::Composite);
        assert_eq!(gaps[0].resource_label, "t");
    }

    #[test]
    fn a_read_run_missing_a_consumer_stage_is_reported() {
        // Negative control for the stage union: narrow the producer barrier to the
        // fragment stage only and the compute consumer must be flagged, even though
        // the resource is correctly in `Read`.
        let mut g = GraphBuilder::new();
        let t = g.create_texture("t", tex());
        let t1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        g.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(t1);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();
        let mut g = g.compile().expect("compiles");
        for pass in &mut g.passes {
            for op in &mut pass.barriers_before {
                if op.to_state() == ResourceState::Read {
                    op.read_stages = ReadStages::FRAGMENT;
                }
            }
        }

        let gaps = barrier_coverage_gaps(&g);
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert_eq!(gaps[0].kind, GapKind::MissingReadStage);
        assert_eq!(gaps[0].pass, PassId::AutoExposure);
    }
}
