// src/render_graph/barrier_place.rs
//
// Where a read run's `* -> Read` transition is recorded, and who relies on it.
//
// `super::compile::derive_barriers` emits one transition per read run and puts
// it on the run's first reader, which is correct as long as one serial order
// covers the whole run. Once the run's readers sit on different queues that
// stops being true: a continuation reader on the other queue carries no
// transition of its own and nothing orders it after the first reader's.
//
// The fix is to record such a run's transition on the *producer* instead, as a
// barrier that runs once the producing pass's own work has completed. Every
// reader of the run reads the version that pass wrote, so the producer is an
// ancestor of all of them and its signal orders the transition for consumers on
// either queue. A run whose readers all share one queue keeps the first-reader
// placement, so a graph that schedules nothing asynchronously compiles to the
// barrier lists it always did.
//
// The queue assignment and the placement depend on each other, so
// `super::schedule` resolves them as a fixed point over the calls here (see
// `read_runs`, `spans_queues` and `reliance`) and applies the result once with
// `apply`.

use alloc::vec;
use alloc::vec::Vec;

use super::compile::CompiledPass;
use super::schedule::PassQueue;
use super::types::ResourceState;

// One read run: the pass whose write the run follows, the pass carrying the
// run's transition, and every pass that reads the resource while the run is
// open. Derived from the passes' declared accesses, the same walk the barrier
// deriver makes, so a run is the same object whether or not its transition has
// already moved. Only runs that follow a write in this graph are tracked; a read
// of an imported resource nothing here produces has no producer to move onto.
pub(super) struct ReadRun {
    resource: usize,
    producer: usize,
    first_reader: usize,
    // Ascending, and never empty: a run is its readers.
    readers: Vec<usize>,
}

// Every read run in `passes`, resource-major so a producer collecting more than
// one moved transition collects them in ascending resource order, matching the
// order the executors expect a barrier list in.
pub(super) fn read_runs(passes: &[CompiledPass], n_resources: usize) -> Vec<ReadRun> {
    let mut runs = Vec::new();
    for resource in 0..n_resources {
        let mut producer: Option<usize> = None;
        let mut readers: Vec<usize> = Vec::new();
        let close = |producer: Option<usize>, readers: &mut Vec<usize>, runs: &mut Vec<ReadRun>| {
            let taken = core::mem::take(readers);
            if let (Some(producer), Some(&first_reader)) = (producer, taken.first()) {
                runs.push(ReadRun {
                    resource,
                    producer,
                    first_reader,
                    readers: taken,
                });
            }
        };
        for (i, pass) in passes.iter().enumerate() {
            if pass.writes.iter().any(|w| w.resource_index() == resource) {
                close(producer, &mut readers, &mut runs);
                producer = Some(i);
            } else if pass.reads.iter().any(|r| r.resource_index() == resource) {
                readers.push(i);
            }
        }
        close(producer, &mut readers, &mut runs);
    }
    runs
}

// Whether the run's readers are split across queues, which is what makes the
// first-reader placement unsound and the producer-side one necessary.
pub(super) fn spans_queues(run: &ReadRun, queues: &[PassQueue]) -> bool {
    let first = queues[run.readers[0]];
    run.readers.iter().any(|&r| queues[r] != first)
}

// Move every spanning run's transition from its first reader's `barriers_before`
// onto its producer's `barriers_after`. Called once, after the queue assignment
// has settled.
pub(super) fn apply(passes: &mut [CompiledPass], runs: &[ReadRun], queues: &[PassQueue]) {
    for run in runs {
        if !spans_queues(run, queues) {
            continue;
        }
        let at = passes[run.first_reader]
            .barriers_before
            .iter()
            .position(|op| op.resource_index() == run.resource)
            .expect("a read run's transition is on its first reader until it moves");
        let op = passes[run.first_reader].barriers_before.remove(at);
        passes[run.producer].barriers_after.push(op);
    }
}

// Every `(transition pass, relying pass)` pair the graph would have under
// `queues`: the pass whose barrier last put a resource in the state some other
// pass's declared access needs. Replayed in serial order, which is the order the
// transitions are recorded in, with each spanning run's transition credited to
// its producer rather than to its first reader. Pairs where the relying pass
// carries the transition itself are omitted: they need no ordering.
//
// The credit is the same whether the move has been applied yet or not: a
// producer's `barriers_after` says it directly, and before `apply` runs the
// spanning runs say it instead. That is what lets the fixed point in
// `super::schedule` ask the question about an assignment it has not committed
// to.
pub(super) fn reliance(
    passes: &[CompiledPass],
    runs: &[ReadRun],
    queues: &[PassQueue],
    n_resources: usize,
) -> Vec<(usize, usize)> {
    // Per first-reader pass, the resources whose transition the producer carries
    // instead. Applied after that pass's own `barriers_before`, which is where
    // the transition would otherwise have been recorded.
    let mut moved: Vec<Vec<(usize, usize)>> = (0..passes.len()).map(|_| Vec::new()).collect();
    for run in runs {
        if spans_queues(run, queues) {
            moved[run.first_reader].push((run.resource, run.producer));
        }
    }

    let mut setter: Vec<Option<usize>> = vec![None; n_resources];
    let mut reliance: Vec<(usize, usize)> = Vec::new();
    for (i, pass) in passes.iter().enumerate() {
        for op in &pass.barriers_before {
            debug_assert_ne!(op.to_state(), ResourceState::Undefined);
            setter[op.resource_index()] = Some(i);
        }
        for &(resource, producer) in &moved[i] {
            setter[resource] = Some(producer);
        }
        for v in pass.writes.iter().chain(pass.reads.iter()) {
            if let Some(barrier) = setter[v.resource_index()]
                && barrier != i
                && !reliance.contains(&(barrier, i))
            {
                reliance.push((barrier, i));
            }
        }
        // A producer-side transition runs once this pass's own accesses are
        // done, so it is credited last, exactly as `super::validate` replays it.
        for op in &pass.barriers_after {
            setter[op.resource_index()] = Some(i);
        }
    }
    reliance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::render_graph::builder::GraphBuilder;
    use crate::render::render_graph::passes::PassId;
    use crate::render::render_graph::types::{
        BufferDesc, BufferUsage, PassKind, PixelFormat, TextureDesc, TextureSize, TextureUsage,
    };
    use alloc::vec::Vec;

    fn tex() -> TextureDesc {
        TextureDesc::texture_2d(
            TextureSize::Drawable,
            TextureSize::Drawable,
            PixelFormat::Rgba16Float,
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        )
    }

    fn buf() -> BufferDesc {
        BufferDesc {
            size_bytes: None,
            usage: BufferUsage::STORAGE,
        }
    }

    // Shadow writes a map two passes read: Main first, then a compute pass that
    // taps it as a continuation reader. That is the shape the producer-side
    // placement exists for.
    fn two_reader_graph() -> crate::render::render_graph::compile::CompiledGraph {
        let mut b = GraphBuilder::new();
        let shadow = b.create_texture("shadow_map", tex());
        let volume = b.create_texture("fog_froxel_volume", tex());
        let args = b.create_buffer("draw_args", buf());
        let scene = b.create_texture("scene", tex());

        let shadow1 = b
            .add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(shadow);
        let args1 = b
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        let scene1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_texture(shadow1);
            p.read_buffer(args1);
            p.write_texture(scene)
        };
        let volume1 = {
            let mut p = b.add_pass(PassId::FogFroxel, PassKind::Compute);
            p.read_texture(shadow1);
            p.write_texture(volume)
        };
        let scene2 = {
            let mut p = b.add_pass(PassId::Fog, PassKind::Render);
            p.read_texture(volume1);
            p.write_texture(scene1)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene2)
            .presents();
        b.compile().expect("compiles")
    }

    #[test]
    fn a_run_read_from_two_queues_moves_onto_its_producer() {
        let g = two_reader_graph();
        let shadow = g
            .resources
            .iter()
            .position(|r| r.label == "shadow_map")
            .expect("declared");
        let shadow_pass = g.pass(PassId::Shadow).expect("present");
        assert_eq!(
            shadow_pass
                .barriers_after
                .iter()
                .filter(|op| op.resource_index() == shadow)
                .count(),
            1,
            "the shadow map's read run should transition on its producer"
        );
        let main = g.pass(PassId::Main).expect("present");
        assert!(
            !main
                .barriers_before
                .iter()
                .any(|op| op.resource_index() == shadow),
            "and no longer on the run's first reader"
        );
        assert_eq!(
            g.pass(PassId::FogFroxel).expect("present").queue,
            PassQueue::AsyncCompute,
            "the continuation reader is what the move frees"
        );
    }

    #[test]
    fn a_run_read_from_one_queue_keeps_the_first_reader_placement() {
        let g = two_reader_graph();
        // `draw_args` is written by Cull and read only by Main, so its run sits
        // wholly on the graphics queue whatever queue Cull runs on.
        let args = g
            .resources
            .iter()
            .position(|r| r.label == "draw_args")
            .expect("declared");
        let main = g.pass(PassId::Main).expect("present");
        assert!(
            main.barriers_before
                .iter()
                .any(|op| op.resource_index() == args)
        );
        assert!(
            g.pass(PassId::Cull)
                .expect("present")
                .barriers_after
                .is_empty()
        );
    }

    #[test]
    fn a_read_with_no_producer_in_the_graph_is_not_a_run() {
        // An imported resource read at v0 has no producing pass here, so there is
        // nothing to move its transition onto and `read_runs` skips it.
        let mut b = GraphBuilder::new();
        let imported = b.import_texture("hdr_depth", tex());
        let scene = b.create_texture("scene", tex());
        let scene1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_texture(imported);
            p.write_texture(scene)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();
        let g = b.compile().expect("compiles");
        let depth = g
            .resources
            .iter()
            .position(|r| r.label == "hdr_depth")
            .expect("declared");
        let queues: Vec<PassQueue> = g.passes.iter().map(|p| p.queue).collect();
        let runs = read_runs(&g.passes, g.resources.len());
        assert!(runs.iter().all(|r| r.resource != depth));
        assert!(
            g.pass(PassId::Main)
                .expect("present")
                .barriers_before
                .iter()
                .any(|op| op.resource_index() == depth)
        );
        assert!(g.passes.iter().all(|p| p.barriers_after.is_empty()));
        assert!(!queues.is_empty());
    }

    #[test]
    fn every_run_names_its_readers_in_order() {
        let g = two_reader_graph();
        let runs = read_runs(&g.passes, g.resources.len());
        assert!(!runs.is_empty());
        for run in &runs {
            assert!(!run.readers.is_empty());
            assert_eq!(run.readers[0], run.first_reader);
            assert!(run.readers.windows(2).all(|w| w[0] < w[1]));
            assert!(run.producer < run.first_reader);
            for &r in &run.readers {
                assert!(
                    g.depends_on(run.producer, r),
                    "a run's producer must reach every one of its readers"
                );
            }
        }
    }

    #[test]
    fn reliance_credits_a_moved_run_to_its_producer() {
        let g = two_reader_graph();
        let runs = read_runs(&g.passes, g.resources.len());
        let queues: Vec<PassQueue> = g.passes.iter().map(|p| p.queue).collect();
        let shadow = g.pass_index(PassId::Shadow).expect("present");
        let main = g.pass_index(PassId::Main).expect("present");
        let froxel = g.pass_index(PassId::FogFroxel).expect("present");

        let pairs = reliance(&g.passes, &runs, &queues, g.resources.len());
        assert!(
            pairs.contains(&(shadow, froxel)),
            "the continuation reader relies on the producer now: {pairs:?}"
        );
        assert!(!pairs.contains(&(main, froxel)));
    }

    #[test]
    fn one_queue_leaves_every_run_where_the_deriver_put_it() {
        // The same runs against an all-graphics assignment: nothing spans, so
        // nothing moves and the continuation reader relies on the first reader
        // again. This is the shape a graph with no async work compiles to.
        let g = two_reader_graph();
        let runs = read_runs(&g.passes, g.resources.len());
        let flat = alloc::vec![PassQueue::Graphics; g.passes.len()];
        assert!(runs.iter().all(|run| !spans_queues(run, &flat)));
    }
}
