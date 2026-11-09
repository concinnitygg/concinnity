//! Running a compiled graph's passes: every pass but the presenter recorded,
//! possibly in parallel, then handed back in compiled order. Passes are recorded
//! in batches of consecutive passes on one queue, each batch into one command
//! buffer: a backend whose command buffers cost the GPU a gap at every boundary
//! records several passes into each, one that keeps a buffer per pass records
//! batches of one. A backend supplies the recording ([`BatchRecorder`] or
//! [`PassRecorder`]) and the host the parallelism ([`FanOut`]); the walk itself
//! is written once.

use alloc::vec::Vec;
use core::ops::Range;
use core::sync::atomic::{AtomicU32, Ordering};

use super::{CompiledGraph, CompiledPass, PassId};
use crate::profile::MAX_PASS_TIMINGS;
use crate::render::backend::error::RenderResult;
use concinnity_core::fanout::FanOut;
use concinnity_core::fanout::FanOutTiming;

/// Records one pass of a compiled graph into a command buffer of its own.
/// Called from several threads at once, one pass each.
pub trait PassRecorder: Sync {
    /// A recorded, not yet submitted, command buffer.
    type Recorded: Send;
    /// Record graph pass `index`, its barriers included. Never the presenter.
    ///
    /// # Errors
    ///
    /// Returns an error if the pass cannot be recorded.
    fn record(&self, index: usize, pass: &CompiledPass) -> RenderResult<Self::Recorded>;
}

/// Records a batch of consecutive passes on one queue into one command buffer.
/// Called from several threads at once, one batch each.
pub trait BatchRecorder: Sync {
    /// A command buffer being recorded.
    type Open;
    /// A recorded, not yet submitted, command buffer.
    type Recorded: Send;
    /// Begin the command buffer for the batch that starts at graph pass
    /// `first`.
    ///
    /// # Errors
    ///
    /// Returns an error if no command buffer can be begun.
    fn open(&self, first: usize, pass: &CompiledPass) -> RenderResult<Self::Open>;
    /// Record graph pass `index` into `open`, its barriers and queue
    /// synchronization included. Never the presenter.
    ///
    /// # Errors
    ///
    /// Returns an error if the pass cannot be recorded.
    fn record(&self, open: &mut Self::Open, index: usize, pass: &CompiledPass) -> RenderResult<()>;
    /// Finish the batch's command buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the command buffer cannot be finished.
    fn close(&self, open: Self::Open) -> RenderResult<Self::Recorded>;
}

// Every pass in a command buffer of its own.
struct OnePerBatch<'r, R>(&'r R);

impl<R: PassRecorder> BatchRecorder for OnePerBatch<'_, R> {
    type Open = Option<R::Recorded>;
    type Recorded = R::Recorded;

    fn open(&self, _first: usize, _pass: &CompiledPass) -> RenderResult<Self::Open> {
        Ok(None)
    }

    fn record(&self, open: &mut Self::Open, index: usize, pass: &CompiledPass) -> RenderResult<()> {
        *open = Some(self.0.record(index, pass)?);
        Ok(())
    }

    fn close(&self, open: Self::Open) -> RenderResult<Self::Recorded> {
        open.ok_or_else(|| {
            crate::render::backend::error::RenderError::Other(
                "graph executor: an empty batch".into(),
            )
        })
    }
}

/// One batch's slot in a fan-out.
#[derive(Debug)]
pub struct PassJob<T> {
    passes: Range<usize>,
    result: Option<RenderResult<T>>,
}

/// The per-batch slots, kept across frames so a frame allocates nothing once
/// the slot list has grown to the graph's pass count.
#[derive(Debug)]
pub struct PassScratch<T> {
    jobs: Vec<PassJob<T>>,
}

impl<T> Default for PassScratch<T> {
    fn default() -> Self {
        Self { jobs: Vec::new() }
    }
}

/// The batches one frame recorded, in compiled order.
pub struct ExecutedPasses<'s, T> {
    /// The presenting pass's index, which the backend records itself on the
    /// thread that presents. `None` for a graph that does not present.
    pub presenter: Option<usize>,
    /// How the fan-out spent its time.
    pub timing: FanOutTiming,
    /// Each pass's recording time in microseconds, indexed by [`PassId`].
    pub record_us: [u32; MAX_PASS_TIMINGS],
    jobs: &'s mut [PassJob<T>],
}

impl<T> ExecutedPasses<'_, T> {
    /// Take every recorded batch with the graph indices it holds, in compiled
    /// order.
    pub fn take(&mut self) -> impl Iterator<Item = (Range<usize>, T)> + '_ {
        self.jobs
            .iter_mut()
            .filter_map(|job| match job.result.take() {
                Some(Ok(recorded)) => Some((job.passes.clone(), recorded)),
                _ => None,
            })
    }
}

/// The index of the pass that presents, which records on the presenting
/// thread rather than in the fan-out.
pub fn presenter_index(graph: &CompiledGraph) -> Option<usize> {
    graph.pass_index(PassId::Composite)
}

/// Split the graph's passes, the presenter left out, into batches of
/// consecutive passes on one queue, written to `out`. A batch takes passes
/// while their recording cost, from `cost_us` by [`PassId`], stays within
/// `budget_us`; a pass that alone exceeds it takes a batch of its own. Passes
/// with no recorded cost (the first frame, or newly enabled ones) join the open
/// batch freely.
pub fn plan_batches(
    graph: &CompiledGraph,
    cost_us: &[u32; MAX_PASS_TIMINGS],
    budget_us: u32,
    out: &mut Vec<Range<usize>>,
) {
    out.clear();
    let presenter = presenter_index(graph);
    let mut open: Option<(Range<usize>, u32)> = None;
    for (i, pass) in graph.passes.iter().enumerate() {
        if Some(i) == presenter {
            if let Some((range, _)) = open.take() {
                out.push(range);
            }
            continue;
        }
        let cost = cost_us[pass.id as usize];
        open = match open.take() {
            Some((range, spent))
                if graph.passes[range.start].queue == pass.queue
                    && range.end == i
                    && spent.saturating_add(cost) <= budget_us =>
            {
                Some((range.start..i + 1, spent + cost))
            }
            other => {
                if let Some((range, _)) = other {
                    out.push(range);
                }
                Some((i..i + 1, cost))
            }
        };
    }
    if let Some((range, _)) = open {
        out.push(range);
    }
}

/// The recording budget that spreads `cost_us`'s total over `workers` batches
/// recording at once, never below the costliest pass, which no batch can
/// split.
pub fn batch_budget(
    graph: &CompiledGraph,
    cost_us: &[u32; MAX_PASS_TIMINGS],
    workers: usize,
) -> u32 {
    let presenter = presenter_index(graph);
    let (total, longest) = graph
        .passes
        .iter()
        .enumerate()
        .filter(|&(i, _)| Some(i) != presenter)
        .map(|(_, p)| cost_us[p.id as usize])
        .fold((0u32, 0u32), |(t, l), c| (t.saturating_add(c), l.max(c)));
    let workers = u32::try_from(workers.max(1)).unwrap_or(u32::MAX);
    longest.max(total.div_ceil(workers))
}

/// Record every pass of `graph` except the presenter through `recorder`, one
/// command buffer per pass, fanned out over `fan`. The first failure in
/// compiled order is returned and every recorded buffer is dropped with it.
///
/// # Errors
///
/// Returns the first pass's recording error, in compiled order.
pub fn execute_passes<'s, R: PassRecorder, F: FanOut>(
    graph: &CompiledGraph,
    recorder: &R,
    fan: &F,
    scratch: &'s mut PassScratch<R::Recorded>,
) -> RenderResult<ExecutedPasses<'s, R::Recorded>> {
    let presenter = presenter_index(graph);
    scratch.jobs.clear();
    scratch.jobs.extend(
        (0..graph.passes.len())
            .filter(|&index| Some(index) != presenter)
            .map(|index| PassJob {
                passes: index..index + 1,
                result: None,
            }),
    );
    run_batches(graph, &OnePerBatch(recorder), fan, scratch)
}

/// Record every batch of `batches` (from [`plan_batches`]) through `recorder`,
/// one command buffer per batch, fanned out over `fan`. The first failure in
/// compiled order is returned and every recorded buffer is dropped with it.
///
/// # Errors
///
/// Returns the first batch's recording error, in compiled order.
pub fn execute_batches<'s, R: BatchRecorder, F: FanOut>(
    graph: &CompiledGraph,
    batches: &[Range<usize>],
    recorder: &R,
    fan: &F,
    scratch: &'s mut PassScratch<R::Recorded>,
) -> RenderResult<ExecutedPasses<'s, R::Recorded>> {
    scratch.jobs.clear();
    scratch.jobs.extend(batches.iter().map(|passes| PassJob {
        passes: passes.clone(),
        result: None,
    }));
    run_batches(graph, recorder, fan, scratch)
}

fn run_batches<'s, R: BatchRecorder, F: FanOut>(
    graph: &CompiledGraph,
    recorder: &R,
    fan: &F,
    scratch: &'s mut PassScratch<R::Recorded>,
) -> RenderResult<ExecutedPasses<'s, R::Recorded>> {
    let presenter = presenter_index(graph);
    let record_us: [AtomicU32; MAX_PASS_TIMINGS] = [const { AtomicU32::new(0) }; MAX_PASS_TIMINGS];
    let timer = fan.timer();
    fan.spawn_each(&mut scratch.jobs, |job| {
        job.result = Some(timer.job(|| record_batch(graph, recorder, fan, &record_us, job)));
    });
    let timing = timer.finish();
    if let Some(error) = scratch
        .jobs
        .iter_mut()
        .find_map(|job| match job.result.take() {
            Some(Err(error)) => Some(error),
            other => {
                job.result = other;
                None
            }
        })
    {
        scratch.jobs.clear();
        return Err(error);
    }
    Ok(ExecutedPasses {
        presenter,
        timing,
        record_us: record_us.map(AtomicU32::into_inner),
        jobs: &mut scratch.jobs,
    })
}

// One batch's command buffer, each pass's recording time stored by its id.
fn record_batch<R: BatchRecorder, F: FanOut>(
    graph: &CompiledGraph,
    recorder: &R,
    fan: &F,
    record_us: &[AtomicU32; MAX_PASS_TIMINGS],
    job: &PassJob<R::Recorded>,
) -> RenderResult<R::Recorded> {
    let first = job.passes.start;
    let mut open = recorder.open(first, &graph.passes[first])?;
    for index in job.passes.clone() {
        let pass = &graph.passes[index];
        let start = fan.clock_ns();
        recorder.record(&mut open, index, pass)?;
        let us = u32::try_from(fan.clock_ns().saturating_sub(start) / 1_000).unwrap_or(u32::MAX);
        record_us[pass.id as usize].store(us, Ordering::Relaxed);
    }
    recorder.close(open)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::backend::error::RenderError;
    use crate::render::render_graph::{FrameGraphInputs, build_frame_graph};
    use concinnity_core::fanout::Inline;
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct Counting {
        fail: Option<PassId>,
        calls: AtomicUsize,
    }

    impl PassRecorder for Counting {
        type Recorded = (usize, PassId);

        fn record(&self, index: usize, pass: &CompiledPass) -> RenderResult<(usize, PassId)> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail == Some(pass.id) {
                Err(RenderError::Other(alloc::format!("{:?}", pass.id)))
            } else {
                Ok((index, pass.id))
            }
        }
    }

    fn graph() -> CompiledGraph {
        let inputs = FrameGraphInputs {
            shadow_enabled: true,
            shadow_map_size: 1024,
            bloom_enabled: true,
            ..FrameGraphInputs::all_off()
        };
        build_frame_graph(&inputs).unwrap()
    }

    #[test]
    fn every_pass_but_the_presenter_is_recorded_in_compiled_order() {
        let graph = graph();
        let recorder = Counting {
            fail: None,
            calls: AtomicUsize::new(0),
        };
        let mut scratch = PassScratch::default();
        let mut done = execute_passes(&graph, &recorder, &Inline, &mut scratch).unwrap();
        let presenter = done.presenter.unwrap();
        assert_eq!(graph.passes[presenter].id, PassId::Composite);
        let recorded: Vec<(Range<usize>, (usize, PassId))> = done.take().collect();
        let expected: Vec<usize> = (0..graph.passes.len())
            .filter(|&i| i != presenter)
            .collect();
        assert_eq!(
            recorded.iter().map(|r| r.0.start).collect::<Vec<_>>(),
            expected
        );
        assert!(
            recorded
                .iter()
                .all(|(r, (j, id))| r.len() == 1 && r.start == *j && graph.passes[*j].id == *id)
        );
        assert_eq!(recorder.calls.load(Ordering::Relaxed), expected.len());
    }

    #[test]
    fn a_failed_pass_fails_the_frame() {
        let graph = graph();
        let recorder = Counting {
            fail: Some(PassId::Shadow),
            calls: AtomicUsize::new(0),
        };
        let mut scratch = PassScratch::default();
        let result = execute_passes(&graph, &recorder, &Inline, &mut scratch);
        assert_eq!(
            result.err(),
            Some(RenderError::Other(alloc::format!("{:?}", PassId::Shadow)))
        );
        assert!(scratch.jobs.is_empty(), "the recorded buffers drop with it");
    }

    #[test]
    fn the_slots_are_reused_from_frame_to_frame() {
        let graph = graph();
        let recorder = Counting {
            fail: None,
            calls: AtomicUsize::new(0),
        };
        let mut scratch = PassScratch::default();
        execute_passes(&graph, &recorder, &Inline, &mut scratch).unwrap();
        let capacity = scratch.jobs.capacity();
        let ptr = scratch.jobs.as_ptr();
        execute_passes(&graph, &recorder, &Inline, &mut scratch).unwrap();
        assert_eq!(scratch.jobs.capacity(), capacity);
        assert_eq!(scratch.jobs.as_ptr(), ptr);
    }

    // Records which passes each batch held, in order.
    struct Batching;

    impl BatchRecorder for Batching {
        type Open = Vec<usize>;
        type Recorded = Vec<usize>;

        fn open(&self, _first: usize, _pass: &CompiledPass) -> RenderResult<Vec<usize>> {
            Ok(Vec::new())
        }

        fn record(
            &self,
            open: &mut Vec<usize>,
            index: usize,
            _pass: &CompiledPass,
        ) -> RenderResult<()> {
            open.push(index);
            Ok(())
        }

        fn close(&self, open: Vec<usize>) -> RenderResult<Vec<usize>> {
            Ok(open)
        }
    }

    fn costs(us: u32) -> [u32; MAX_PASS_TIMINGS] {
        [us; MAX_PASS_TIMINGS]
    }

    #[test]
    fn a_batch_holds_consecutive_passes_on_one_queue_within_the_budget() {
        let graph = graph();
        let presenter = presenter_index(&graph).unwrap();
        let mut batches = Vec::new();
        plan_batches(&graph, &costs(10), 25, &mut batches);
        let mut covered = Vec::new();
        for b in &batches {
            assert!(
                !b.is_empty() && b.len() <= 2,
                "{b:?} over a 25 us budget of 10 us passes"
            );
            assert!(!b.contains(&presenter));
            let queue = graph.passes[b.start].queue;
            assert!(b.clone().all(|i| graph.passes[i].queue == queue));
            covered.extend(b.clone());
        }
        let expected: Vec<usize> = (0..graph.passes.len())
            .filter(|&i| i != presenter)
            .collect();
        assert_eq!(covered, expected);
    }

    #[test]
    fn a_zero_budget_gives_every_pass_a_batch_and_a_large_one_merges_runs() {
        let graph = graph();
        let mut batches = Vec::new();
        plan_batches(&graph, &costs(10), 0, &mut batches);
        assert!(batches.iter().all(|b| b.len() == 1));
        let singles = batches.len();
        plan_batches(&graph, &costs(10), u32::MAX, &mut batches);
        assert!(batches.len() < singles);
    }

    #[test]
    fn the_budget_spreads_the_work_but_never_splits_a_pass() {
        let graph = graph();
        let presenter = presenter_index(&graph).unwrap();
        let passes = u32::try_from(graph.passes.len() - 1).unwrap();
        assert_eq!(batch_budget(&graph, &costs(10), 1), 10 * passes);
        let mut lopsided = costs(1);
        lopsided[graph.passes[usize::from(presenter == 0)].id as usize] = 1_000;
        assert_eq!(batch_budget(&graph, &lopsided, 64), 1_000);
    }

    #[test]
    fn each_batch_records_its_passes_into_one_buffer_in_order() {
        let graph = graph();
        let mut batches = Vec::new();
        plan_batches(&graph, &costs(10), u32::MAX, &mut batches);
        let mut scratch = PassScratch::default();
        let mut done = execute_batches(&graph, &batches, &Batching, &Inline, &mut scratch).unwrap();
        for (range, recorded) in done.take() {
            assert_eq!(recorded, range.collect::<Vec<_>>());
        }
    }
}
