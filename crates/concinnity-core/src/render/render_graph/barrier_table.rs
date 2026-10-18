//! The per-frame tables a backend executor resolves the graph's resources
//! into: which GPU object backs each driven resource, and which pooled
//! transients each pass reclaims memory for. Refilled every frame from live
//! state into storage kept across frames, so a frame allocates nothing once
//! the tables have grown to the graph.

use alloc::vec::Vec;

use super::{CompiledGraph, CompiledResource};

/// Each graph resource's barrier target, indexed by its position in
/// `CompiledGraph::resources` (`None` for one the backend does not drive),
/// plus an arena for a target that names several objects under one label.
#[derive(Debug)]
pub struct BarrierTable<T, H> {
    targets: Vec<Option<T>>,
    arena: Vec<H>,
}

impl<T, H> Default for BarrierTable<T, H> {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            arena: Vec::new(),
        }
    }
}

impl<T, H> BarrierTable<T, H> {
    /// Resolve every resource of `graph` through `resolve`, which may append
    /// the objects a set target names to the arena it is handed.
    pub fn refill(
        &mut self,
        graph: &CompiledGraph,
        mut resolve: impl FnMut(&CompiledResource, &mut Vec<H>) -> Option<T>,
    ) {
        self.targets.clear();
        self.arena.clear();
        for res in &graph.resources {
            let target = resolve(res, &mut self.arena);
            self.targets.push(target);
        }
    }

    /// The target of resource `index`, when the backend drives it.
    pub fn target(&self, index: usize) -> Option<&T> {
        self.targets.get(index)?.as_ref()
    }

    /// The arena range `first..first + count`.
    pub fn set(&self, first: usize, count: usize) -> &[H] {
        self.arena.get(first..first + count).unwrap_or_default()
    }

    /// Whether each resource is driven, in resource order.
    pub fn driven(&self) -> impl Iterator<Item = bool> + '_ {
        self.targets.iter().map(Option::is_some)
    }

    /// Every driven resource's index and target.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.targets
            .iter()
            .enumerate()
            .filter_map(|(i, t)| Some((i, t.as_ref()?)))
    }
}

/// The pooled transients each pass reclaims shared memory for, at the pass
/// that first uses them, sorted by pass.
#[derive(Debug)]
pub struct AliasRows<A> {
    passes: Vec<usize>,
    entries: Vec<A>,
}

impl<A> Default for AliasRows<A> {
    fn default() -> Self {
        Self {
            passes: Vec::new(),
            entries: Vec::new(),
        }
    }
}

impl<A> AliasRows<A> {
    /// Resolve the aliased resources of `graph`: `resolve` returns the entry for
    /// resource `index` when it reuses an earlier transient's memory.
    pub fn refill(
        &mut self,
        graph: &CompiledGraph,
        mut resolve: impl FnMut(usize, &CompiledResource) -> Option<A>,
    ) {
        self.passes.clear();
        self.entries.clear();
        for (index, res) in graph.resources.iter().enumerate() {
            let first = res.lifetime.first;
            if first >= graph.passes.len() {
                continue;
            }
            if let Some(entry) = resolve(index, res) {
                let at = self.passes.partition_point(|&p| p <= first);
                self.passes.insert(at, first);
                self.entries.insert(at, entry);
            }
        }
    }

    /// The entries pass `pass` owes.
    pub fn for_pass(&self, pass: usize) -> &[A] {
        let start = self.passes.partition_point(|&p| p < pass);
        let end = self.passes.partition_point(|&p| p <= pass);
        &self.entries[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::render_graph::{FrameGraphInputs, build_frame_graph};

    fn graph() -> CompiledGraph {
        let inputs = FrameGraphInputs {
            shadow_enabled: true,
            shadow_map_size: 1024,
            bloom_enabled: true,
            ssao_enabled: true,
            gbuffer_prepass_enabled: true,
            ..FrameGraphInputs::all_off()
        };
        build_frame_graph(&inputs).unwrap()
    }

    #[test]
    fn the_table_holds_one_entry_per_resource_and_a_shared_arena() {
        let graph = graph();
        let mut table: BarrierTable<(usize, usize), u32> = BarrierTable::default();
        table.refill(&graph, |res, arena| match res.label {
            "shadow_map" => {
                let first = arena.len();
                arena.extend([7, 8]);
                Some((first, 2))
            }
            _ => None,
        });
        assert_eq!(table.driven().count(), graph.resources.len());
        let (index, &(first, count)) = table.iter().next().unwrap();
        assert_eq!(graph.resources[index].label, "shadow_map");
        assert_eq!(table.set(first, count), &[7, 8]);
        assert_eq!(table.iter().count(), 1);
        assert!(table.target(graph.resources.len()).is_none());
        table.refill(&graph, |_, _| None);
        assert_eq!(table.iter().count(), 0);
        assert!(
            table.set(0, 2).is_empty(),
            "the arena refills with the table"
        );
    }

    #[test]
    fn alias_rows_land_at_each_resource_first_use() {
        let graph = graph();
        let mut rows: AliasRows<usize> = AliasRows::default();
        rows.refill(&graph, |index, _| Some(index));
        for (pass, _) in graph.passes.iter().enumerate() {
            for &index in rows.for_pass(pass) {
                assert_eq!(graph.resources[index].lifetime.first, pass);
            }
        }
        let total: usize = (0..graph.passes.len())
            .map(|p| rows.for_pass(p).len())
            .sum();
        let used = graph
            .resources
            .iter()
            .filter(|r| r.lifetime.first < graph.passes.len())
            .count();
        assert_eq!(total, used);
    }
}
