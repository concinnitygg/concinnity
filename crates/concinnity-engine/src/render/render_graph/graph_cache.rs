//! The compiled frame graph a backend reuses from frame to frame. Its topology
//! only changes when a feature toggles or a target resizes, so a frame whose
//! inputs match the last one's skips the rebuild.

use super::{CompiledGraph, FrameGraphInputs, GraphError, build_frame_graph};

/// The last frame's compiled graph and the inputs it was built from.
#[derive(Debug, Default)]
pub struct GraphCache {
    entry: Option<(FrameGraphInputs, CompiledGraph)>,
}

impl GraphCache {
    /// The graph for `inputs`: the cached one when its inputs match, else a
    /// fresh build. Taken out of the cache so the frame can execute it while
    /// the backend is borrowed mutably; [`Self::put`] returns it.
    pub(crate) fn take(&mut self, inputs: &FrameGraphInputs) -> Result<CompiledGraph, GraphError> {
        self.take_or_build(inputs, build_frame_graph)
    }

    fn take_or_build(
        &mut self,
        inputs: &FrameGraphInputs,
        build: impl FnOnce(&FrameGraphInputs) -> Result<CompiledGraph, GraphError>,
    ) -> Result<CompiledGraph, GraphError> {
        match self.entry.take() {
            Some((cached, graph)) if cached == *inputs => Ok(graph),
            _ => build(inputs),
        }
    }

    /// Keep `graph`, built from `inputs`, for the next frame. A frame that
    /// failed before this leaves the cache cold, so the next frame rebuilds.
    pub fn put(&mut self, inputs: FrameGraphInputs, graph: CompiledGraph) {
        self.entry = Some((inputs, graph));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    fn counting(
        builds: &Cell<u32>,
    ) -> impl FnOnce(&FrameGraphInputs) -> Result<CompiledGraph, GraphError> + '_ {
        move |inputs| {
            builds.set(builds.get() + 1);
            build_frame_graph(inputs)
        }
    }

    #[test]
    fn matching_inputs_reuse_the_graph_and_changed_ones_rebuild() {
        let builds = Cell::new(0);
        let mut cache = GraphCache::default();
        let a = FrameGraphInputs::all_off();
        let g = cache.take_or_build(&a, counting(&builds)).unwrap();
        cache.put(a, g);
        let g = cache.take_or_build(&a, counting(&builds)).unwrap();
        assert_eq!(builds.get(), 1, "matching inputs reuse");
        cache.put(a, g);
        let b = FrameGraphInputs {
            hdr_width: a.hdr_width + 1,
            ..a
        };
        let g = cache.take_or_build(&b, counting(&builds)).unwrap();
        assert_eq!(builds.get(), 2, "changed inputs rebuild");
        drop(g);
        // A frame that never put its graph back leaves the cache cold.
        cache.take_or_build(&b, counting(&builds)).unwrap();
        assert_eq!(builds.get(), 3);
    }

    #[test]
    fn take_builds_the_frame_graph() {
        let mut cache = GraphCache::default();
        let inputs = FrameGraphInputs::all_off();
        let graph = cache.take(&inputs).unwrap();
        assert_eq!(
            graph.passes.len(),
            build_frame_graph(&inputs).unwrap().passes.len()
        );
    }
}
