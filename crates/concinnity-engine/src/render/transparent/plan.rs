//! One frame's transparent-pass decisions: whether it traces, which trace
//! variant, this frame's see-through mesh draws, and the back-to-front draw
//! order over every producer. The buffers are reused frame to frame, so a
//! steady frame allocates nothing.

use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;
use core::cmp::Ordering;

use crate::render::transparent::producer::TransparentProducers;
use crate::render::transparent::{Producer, sort_distance};

/// The transparent pass's per-frame plan. `M` is the backend's see-through
/// mesh draw; the backend fills [`Self::meshes_mut`] after [`Self::begin`] and
/// before [`Self::order_draws`].
pub struct TransparentFramePlan<M> {
    rt_live: bool,
    textured: bool,
    meshes: Vec<M>,
    order: Vec<(Producer, usize)>,
    // Sort key per `order` entry: camera distance and position in the
    // unsorted list, which keeps equal distances in declaration order.
    keys: Vec<(f32, u32)>,
}

impl<M> Default for TransparentFramePlan<M> {
    fn default() -> Self {
        Self {
            rt_live: false,
            textured: false,
            meshes: Vec::new(),
            order: Vec::new(),
            keys: Vec::new(),
        }
    }
}

impl<M> TransparentFramePlan<M> {
    /// Start a frame: per-pixel tracing is `rt_live`, and the hit shading reads
    /// the bindless pool when `textured` as well. Clears last frame's draws.
    pub fn begin(&mut self, rt_live: bool, textured: bool) {
        self.rt_live = rt_live;
        self.textured = rt_live && textured;
        self.meshes.clear();
        self.order.clear();
        self.keys.clear();
    }

    /// Whether the pass traces this frame.
    pub fn rt_live(&self) -> bool {
        self.rt_live
    }

    /// Whether the trace shades hits from the bindless pool.
    pub fn textured(&self) -> bool {
        self.textured
    }

    /// This frame's see-through mesh draws, which only run while the pass
    /// traces.
    pub fn meshes_mut(&mut self) -> &mut Vec<M> {
        &mut self.meshes
    }

    /// This frame's see-through mesh draws, indexed by the
    /// [`Producer::GlassMesh`] entries of [`Self::order`].
    pub fn meshes(&self) -> &[M] {
        &self.meshes
    }

    /// Order every visible record of `producers` and every mesh draw (centered
    /// where `mesh_center` says) farthest from `cam` first. The producers
    /// interleave so a pane standing in a pool composites in the right order;
    /// a non-finite distance sorts nearest.
    pub fn order_draws<P, R>(
        &mut self,
        producers: &TransparentProducers<P, R>,
        mesh_center: impl Fn(&M) -> [f32; 3],
        cam: [f32; 3],
    ) {
        self.order.clear();
        self.keys.clear();
        for kind in [Producer::Glass, Producer::Water] {
            let Some(p) = producers.producer(kind) else {
                continue;
            };
            for (i, r) in p.records.iter().enumerate().filter(|(_, r)| r.visible) {
                self.keys.push((
                    sort_distance(r.center, cam),
                    u32::saturating_from(self.order.len()),
                ));
                self.order.push((kind, i));
            }
        }
        for (i, m) in self.meshes.iter().enumerate() {
            self.keys.push((
                sort_distance(mesh_center(m), cam),
                u32::saturating_from(self.order.len()),
            ));
            self.order.push((Producer::GlassMesh, i));
        }
        let finite = |d: f32| if d.is_finite() { d } else { f32::NEG_INFINITY };
        self.keys.sort_unstable_by(|a, b| {
            finite(b.0)
                .partial_cmp(&finite(a.0))
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        // Permute `order` into the sorted key order. Each key carries the
        // entry's original position, so a cycle walk applies it in place.
        for slot in 0..self.keys.len() {
            let mut from = self.keys[slot].1 as usize;
            while from < slot {
                from = self.keys[from].1 as usize;
            }
            self.order.swap(slot, from);
        }
    }

    /// The frame's draws, farthest first.
    pub fn order(&self) -> &[(Producer, usize)] {
        &self.order
    }

    /// Whether a glass surface (pane or mesh) draws this frame.
    fn draws_glass(&self) -> bool {
        self.order
            .iter()
            .any(|&(kind, _)| matches!(kind, Producer::Glass | Producer::GlassMesh))
    }

    /// Whether the glass reflection pre-pass runs: the pass traces, the reduced
    /// layers exist (`layers`), and a glass surface draws.
    pub fn reflection_prepass(&self, layers: bool) -> bool {
        self.rt_live && layers && self.draws_glass()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::transparent::back_to_front_order;
    use crate::render::transparent::producer::{
        ShapePipelines, TransparentProducer, TransparentRecord,
    };
    use alloc::vec;

    fn producer(records: &[([f32; 3], bool)]) -> TransparentProducer<(), ()> {
        TransparentProducer {
            shape: ShapePipelines {
                base: (),
                flat_rt: None,
                textured_rt: None,
            },
            reflection_flat: None,
            reflection_textured: None,
            sea: None,
            records: records
                .iter()
                .map(|&(center, visible)| TransparentRecord {
                    gpu: (),
                    visible,
                    center,
                    planar_slot: None,
                })
                .collect(),
            water: Vec::new(),
        }
    }

    fn plan_for(
        glass: &[([f32; 3], bool)],
        water: &[([f32; 3], bool)],
        meshes: &[[f32; 3]],
        cam: [f32; 3],
    ) -> TransparentFramePlan<[f32; 3]> {
        let producers = TransparentProducers {
            glass: Some(producer(glass)),
            water: Some(producer(water)),
        };
        let mut plan = TransparentFramePlan::default();
        plan.begin(true, false);
        plan.meshes_mut().extend_from_slice(meshes);
        plan.order_draws(&producers, |m| *m, cam);
        plan
    }

    fn order_of(
        glass: &[([f32; 3], bool)],
        water: &[([f32; 3], bool)],
        meshes: &[[f32; 3]],
        cam: [f32; 3],
    ) -> Vec<(Producer, usize)> {
        plan_for(glass, water, meshes, cam).order().to_vec()
    }

    #[test]
    fn ordered_draws_excludes_hidden_and_sorts_back_to_front() {
        // Pane 1 is hidden; 0 (dist 5) and 2 (dist 3) are visible. Farthest
        // first => [0, 2]; the hidden pane never appears.
        let glass = [
            ([0.0, 0.0, 5.0], true),
            ([0.0, 0.0, 9.0], false),
            ([0.0, 0.0, 3.0], true),
        ];
        let order = order_of(&glass, &[], &[], [0.0, 0.0, 0.0]);
        assert_eq!(order, vec![(Producer::Glass, 0), (Producer::Glass, 2)]);
    }

    #[test]
    fn ordered_draws_interleaves_the_two_producers() {
        // A pane standing in a pool has to composite in distance order, not in
        // producer order: the far pane draws first, then the water, then the near
        // pane.
        let glass = [([0.0, 0.0, 9.0], true), ([0.0, 0.0, 1.0], true)];
        let water = [([0.0, 0.0, 5.0], true), ([0.0, 0.0, 7.0], false)];
        let order = order_of(&glass, &water, &[], [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![
                (Producer::Glass, 0),
                (Producer::Water, 0),
                (Producer::Glass, 1),
            ]
        );
    }

    #[test]
    fn ordered_draws_interleaves_mesh_draws_with_the_static_producers() {
        // A see-through mesh sorts against panes and water by the same camera
        // distance, so it is not simply appended after them. Every mesh entry the
        // encoder passes is already visible, which is why the slice carries
        // centers alone.
        let glass = [([0.0, 0.0, 9.0], true)];
        let water = [([0.0, 0.0, 3.0], true)];
        let meshes = [[0.0, 0.0, 6.0], [0.0, 0.0, 1.0]];
        let order = order_of(&glass, &water, &meshes, [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![
                (Producer::Glass, 0),
                (Producer::GlassMesh, 0),
                (Producer::Water, 0),
                (Producer::GlassMesh, 1),
            ]
        );
    }

    #[test]
    fn ordered_draws_orders_meshes_alone_back_to_front() {
        // A world whose only transparent content is see-through meshes: the pass
        // still runs, and they still sort farthest first.
        let meshes = [[0.0, 0.0, 2.0], [0.0, 0.0, 8.0]];
        let order = order_of(&[], &[], &meshes, [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![(Producer::GlassMesh, 1), (Producer::GlassMesh, 0)]
        );
    }

    #[test]
    fn ordered_draws_is_empty_with_no_visible_records() {
        let glass = [([0.0, 0.0, 5.0], false)];
        let water = [([0.0, 0.0, 3.0], false)];
        assert!(order_of(&glass, &water, &[], [0.0, 0.0, 0.0]).is_empty());
    }

    #[test]
    fn nan_distances_sort_nearest() {
        let glass = [
            ([0.0, 0.0, 5.0], true),
            ([0.0, 0.0, 9.0], false),
            ([0.0, 0.0, 2.0], true),
            ([0.0, 0.0, 5.0], true),
        ];
        let water = [([0.0, 0.0, 7.0], true), ([0.0, 0.0, 5.0], true)];
        let meshes = [[0.0, 0.0, 3.0], [f32::NAN, 0.0, 0.0], [0.0, 0.0, 8.0]];
        let order = order_of(&glass, &water, &meshes, [0.0; 3]);
        assert_eq!(
            order,
            vec![
                (Producer::GlassMesh, 2),
                (Producer::Water, 0),
                (Producer::Glass, 0),
                (Producer::Glass, 3),
                (Producer::Water, 1),
                (Producer::GlassMesh, 0),
                (Producer::Glass, 2),
                (Producer::GlassMesh, 1),
            ]
        );
    }

    #[test]
    fn many_equal_distances_keep_declaration_order() {
        let glass: Vec<_> = (0..40).map(|_| ([1.0, 0.0, 0.0], true)).collect();
        let plan = plan_for(&glass, &[], &[], [0.0; 3]);
        let indices: Vec<usize> = plan.order().iter().map(|&(_, i)| i).collect();
        assert_eq!(indices, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn a_long_mixed_list_sorts_like_back_to_front_order() {
        let glass: Vec<_> = (0..37)
            .map(|i| ([((i * 7919) % 13) as f32, 0.0, 0.0], i % 5 != 0))
            .collect();
        let plan = plan_for(&glass, &[], &[], [0.0; 3]);
        let live: Vec<usize> = (0..37).filter(|i| i % 5 != 0).collect();
        let dists: Vec<f32> = live.iter().map(|&i| glass[i].0[0]).collect();
        let expected: Vec<usize> = back_to_front_order(&dists)
            .into_iter()
            .map(|o| live[o])
            .collect();
        let got: Vec<usize> = plan.order().iter().map(|&(_, i)| i).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn begin_clears_the_last_frame() {
        let mut plan = plan_for(&[([0.0; 3], true)], &[], &[[1.0; 3]], [0.0; 3]);
        assert_eq!(plan.order().len(), 2);
        plan.begin(false, true);
        assert!(plan.order().is_empty());
        assert!(plan.meshes_mut().is_empty());
        assert!(!plan.textured(), "textured needs the trace");
    }

    #[test]
    fn the_reflection_prepass_needs_the_trace_layers_and_glass() {
        let water_only = plan_for(&[], &[([0.0; 3], true)], &[], [0.0; 3]);
        assert!(!water_only.draws_glass());
        assert!(!water_only.reflection_prepass(true));
        let mesh = plan_for(&[], &[], &[[0.0; 3]], [0.0; 3]);
        assert!(mesh.reflection_prepass(true));
        assert!(!mesh.reflection_prepass(false));
        let mut off = plan_for(&[([0.0; 3], true)], &[], &[], [0.0; 3]);
        off.begin(false, false);
        assert!(!off.reflection_prepass(true));
        assert_eq!(mesh.meshes(), [[0.0; 3]]);
        assert_eq!(mesh.order(), vec![(Producer::GlassMesh, 0)]);
    }
}
