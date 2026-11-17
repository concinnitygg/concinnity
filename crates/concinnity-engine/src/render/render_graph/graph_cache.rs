//! The compiled frame graphs a backend reuses from frame to frame. A graph's
//! topology changes when a feature toggles or a target resizes, and between a
//! few shapes that alternate frame by frame, such as a frame that redraws the
//! far shadow and one that keeps it. The cache holds the most recent of them,
//! so neither the alternation nor a toggle back rebuilds a graph.

use super::{CompiledGraph, FrameGraphInputs, GraphError, build_frame_graph};

// How many graphs are kept: the shapes a steady scene alternates between, with
// room for one feature toggled back and forth.
const GRAPH_CACHE_ENTRIES: usize = 4;

/// The most recently used compiled graphs and the inputs each was built from.
#[derive(Debug, Default)]
pub struct GraphCache {
    entries: [Option<(FrameGraphInputs, CompiledGraph)>; GRAPH_CACHE_ENTRIES],
    // When each entry was last put back, by `uses`: the least recent is the
    // one a new graph replaces.
    last_used: [u64; GRAPH_CACHE_ENTRIES],
    uses: u64,
}

impl GraphCache {
    /// The graph for `inputs`: a cached one built from equal inputs, else a
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
        let hit = self
            .entries
            .iter_mut()
            .find(|e| e.as_ref().is_some_and(|(cached, _)| cached == inputs));
        match hit.and_then(Option::take) {
            Some((_, graph)) => Ok(graph),
            None => build(inputs),
        }
    }

    /// Keep `graph`, built from `inputs`, for the frames after this one, in an
    /// empty entry or in place of the least recently used. A frame that failed
    /// before this keeps no graph for its inputs, so the next such frame
    /// rebuilds.
    pub fn put(&mut self, inputs: FrameGraphInputs, graph: CompiledGraph) {
        self.uses += 1;
        let slot = self
            .entries
            .iter()
            .position(Option::is_none)
            .or_else(|| (0..GRAPH_CACHE_ENTRIES).min_by_key(|&i| self.last_used[i]))
            .unwrap_or(0);
        self.entries[slot] = Some((inputs, graph));
        self.last_used[slot] = self.uses;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use concinnity_core::math::SaturatingFrom;
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

    fn with_width(inputs: FrameGraphInputs, width: u32) -> FrameGraphInputs {
        FrameGraphInputs {
            hdr_width: width,
            ..inputs
        }
    }

    // Frames alternating between two shapes, as a far shadow redrawn every
    // other frame makes them, build each once.
    #[test]
    fn alternating_inputs_build_each_graph_once() {
        let builds = Cell::new(0);
        let mut cache = GraphCache::default();
        let a = FrameGraphInputs::all_off();
        let b = with_width(a, a.hdr_width + 1);
        for _ in 0..8 {
            for inputs in [a, b] {
                let g = cache.take_or_build(&inputs, counting(&builds)).unwrap();
                cache.put(inputs, g);
            }
        }
        assert_eq!(builds.get(), 2);
    }

    // Past its room the cache drops the graph used longest ago.
    #[test]
    fn a_full_cache_replaces_the_least_recently_used() {
        let builds = Cell::new(0);
        let mut cache = GraphCache::default();
        let base = FrameGraphInputs::all_off();
        let shapes: Vec<_> = (0..=u32::saturating_from(GRAPH_CACHE_ENTRIES))
            .map(|k| with_width(base, base.hdr_width + k))
            .collect();
        for inputs in &shapes {
            let g = cache.take_or_build(inputs, counting(&builds)).unwrap();
            cache.put(*inputs, g);
        }
        assert_eq!(builds.get(), u32::saturating_from(shapes.len()));
        // The most recent shapes are kept; the first was replaced.
        for inputs in shapes.iter().skip(1).rev() {
            let g = cache.take_or_build(inputs, counting(&builds)).unwrap();
            cache.put(*inputs, g);
        }
        assert_eq!(builds.get(), u32::saturating_from(shapes.len()));
        let g = cache.take_or_build(&shapes[0], counting(&builds)).unwrap();
        cache.put(shapes[0], g);
        assert_eq!(builds.get(), u32::saturating_from(shapes.len()) + 1);
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
