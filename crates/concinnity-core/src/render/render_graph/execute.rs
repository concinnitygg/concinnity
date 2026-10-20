//! Running a compiled graph's passes: every pass but the presenter recorded
//! into its own command buffer, possibly in parallel, then handed back in
//! compiled order. A backend supplies the recording ([`PassRecorder`]) and the
//! host the parallelism ([`Fanout`]); the walk itself is written once.

use alloc::vec::Vec;
use core::sync::atomic::AtomicU32;

use super::{CompiledGraph, CompiledPass, PassId};
use crate::fanout::Fanout;
use crate::profile::{FanOutTiming, MAX_PASS_TIMINGS};
use crate::render::backend::error::RenderResult;

/// Records one pass of a compiled graph into a command buffer of its own.
/// Called from several threads at once, one pass each.
pub trait PassRecorder: Sync {
    /// A recorded, not yet submitted, command buffer.
    type Recorded: Send;
    /// Record graph pass `index`, its barriers included. Never the presenter.
    fn record(&self, index: usize, pass: &CompiledPass) -> RenderResult<Self::Recorded>;
}

/// One pass's slot in a fan-out.
#[derive(Debug)]
pub struct PassJob<T> {
    index: usize,
    id: PassId,
    result: Option<RenderResult<T>>,
}

/// The per-pass slots, kept across frames so a frame allocates nothing once the
/// slot list has grown to the graph's pass count.
#[derive(Debug)]
pub struct PassScratch<T> {
    jobs: Vec<PassJob<T>>,
}

impl<T> Default for PassScratch<T> {
    fn default() -> Self {
        Self { jobs: Vec::new() }
    }
}

/// The passes one frame recorded, in compiled order.
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
    /// Take every recorded pass with its graph index, in compiled order.
    pub fn take(&mut self) -> impl Iterator<Item = (usize, T)> + '_ {
        self.jobs
            .iter_mut()
            .filter_map(|job| match job.result.take() {
                Some(Ok(recorded)) => Some((job.index, recorded)),
                _ => None,
            })
    }
}

/// The index of the pass that presents, which records on the presenting
/// thread rather than in the fan-out.
pub fn presenter_index(graph: &CompiledGraph) -> Option<usize> {
    graph.pass_index(PassId::Composite)
}

/// Record every pass of `graph` except the presenter through `recorder`, fanned
/// out over `fan`, one job per pass. The first failure in compiled order is returned and every
/// recorded buffer is dropped with it.
pub fn execute_passes<'s, R: PassRecorder, F: Fanout>(
    graph: &CompiledGraph,
    recorder: &R,
    fan: &F,
    scratch: &'s mut PassScratch<R::Recorded>,
) -> RenderResult<ExecutedPasses<'s, R::Recorded>> {
    let presenter = presenter_index(graph);
    scratch.jobs.clear();
    scratch.jobs.extend(
        graph
            .passes
            .iter()
            .enumerate()
            .filter(|&(index, _)| Some(index) != presenter)
            .map(|(index, pass)| PassJob {
                index,
                id: pass.id,
                result: None,
            }),
    );
    let record_us: [AtomicU32; MAX_PASS_TIMINGS] = [const { AtomicU32::new(0) }; MAX_PASS_TIMINGS];
    let timer = fan.timer();
    fan.spawn_each(&mut scratch.jobs, |job| {
        let slot = &record_us[job.id as usize];
        job.result = Some(timer.job_into(slot, || {
            recorder.record(job.index, &graph.passes[job.index])
        }));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fanout::Inline;
    use crate::render::backend::error::RenderError;
    use crate::render::render_graph::{FrameGraphInputs, build_frame_graph};
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
        let recorded: Vec<(usize, (usize, PassId))> = done.take().collect();
        let expected: Vec<usize> = (0..graph.passes.len())
            .filter(|&i| i != presenter)
            .collect();
        assert_eq!(recorded.iter().map(|r| r.0).collect::<Vec<_>>(), expected);
        assert!(
            recorded
                .iter()
                .all(|&(i, (j, id))| i == j && graph.passes[i].id == id)
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
}
