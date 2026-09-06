// Per-instance LOD for the folded instanced cull records: which index slice
// each instance of an `InstancedProp` draws from this frame.

use crate::gfx::render_types::InstancedCluster;

/// Whether any cluster declares LOD alternates. Callers cache this at load and
/// skip [`for_each_instance_lod`] entirely when it is false: every instance of
/// an alternate-less cluster draws the cluster's base slice for the life of the
/// world, so its records never need rewriting.
pub fn any_cluster_has_lod(clusters: &[InstancedCluster]) -> bool {
    clusters.iter().any(|c| !c.lod_alternates.is_empty())
}

/// Visit `(record, index_offset, index_count)` for every instance of every
/// cluster that declares LOD alternates, where `record` is the instance's
/// position in the cull's instance tail.
///
/// The tail is in cluster-then-instance order, the ordering
/// [`instance_object_records`](crate::gfx::render_types::instance_object_records)
/// packs its records in, so `record` addresses the same instance in the object
/// and draw-args buffers. Clusters with no alternates are skipped but still
/// advance `record`.
///
/// Distance is measured from `cam_pos` to the instance's model translation,
/// matching [`InstancedCluster::lod_buckets`] so the spot caster body and the
/// GPU-driven passes put an instance at the same level.
pub fn for_each_instance_lod(
    clusters: &[InstancedCluster],
    cam_pos: [f32; 3],
    mut visit: impl FnMut(usize, usize, usize),
) {
    let mut record = 0usize;
    for cluster in clusters {
        if cluster.lod_alternates.is_empty() {
            record += cluster.instances.len();
            continue;
        }
        for model in &cluster.instances {
            let d = super::instance_camera_distance(*model, cam_pos);
            let (index_offset, index_count) = super::pick_lod_slice(
                (cluster.index_offset, cluster.index_count),
                &cluster.lod_alternates,
                d,
            );
            visit(record, index_offset, index_count);
            record += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::{LodSlice, MaterialUniforms, NO_NORMAL_MAP_SLOT};
    use alloc::vec;
    use alloc::vec::Vec;

    fn cluster(instance_x: &[f32], alternates: Vec<LodSlice>) -> InstancedCluster {
        InstancedCluster {
            vertex_offset: 0,
            vertex_count: 4,
            index_offset: 100,
            index_count: 12,
            texture_slot: 0,
            normal_map_slot: NO_NORMAL_MAP_SLOT,
            material: MaterialUniforms::DEFAULT,
            cluster_bb_min: [0.0; 3],
            cluster_bb_max: [1.0; 3],
            local_bb_min: [0.0; 3],
            local_bb_max: [1.0; 3],
            cull_distance: 0.0,
            instances: instance_x
                .iter()
                .map(|x| {
                    let mut m = [[0.0f32; 4]; 4];
                    m[0][0] = 1.0;
                    m[1][1] = 1.0;
                    m[2][2] = 1.0;
                    m[3] = [*x, 0.0, 0.0, 1.0];
                    m
                })
                .collect(),
            lod_alternates: alternates,
        }
    }

    fn slice(switch_distance: f32, index_offset: usize) -> LodSlice {
        LodSlice {
            index_offset,
            index_count: 6,
            switch_distance,
        }
    }

    fn visited(clusters: &[InstancedCluster], cam_pos: [f32; 3]) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for_each_instance_lod(clusters, cam_pos, |record, offset, count| {
            out.push((record, offset, count));
        });
        out
    }

    #[test]
    fn a_cluster_without_alternates_is_skipped_whole() {
        let clusters = [cluster(&[0.0, 50.0], Vec::new())];
        assert!(!any_cluster_has_lod(&clusters));
        assert!(visited(&clusters, [0.0; 3]).is_empty());
    }

    #[test]
    fn instances_either_side_of_a_threshold_take_different_slices() {
        let clusters = [cluster(&[5.0, 25.0], vec![slice(10.0, 200)])];
        assert!(any_cluster_has_lod(&clusters));
        assert_eq!(
            visited(&clusters, [0.0; 3]),
            [(0, 100, 12), (1, 200, 6)],
            "the near instance keeps the base slice, the far one takes the alternate"
        );
    }

    #[test]
    fn the_threshold_itself_selects_the_alternate() {
        let clusters = [cluster(&[10.0], vec![slice(10.0, 200)])];
        assert_eq!(visited(&clusters, [0.0; 3]), [(0, 200, 6)]);
    }

    #[test]
    fn records_stay_in_cluster_then_instance_order_across_a_skipped_cluster() {
        let clusters = [
            cluster(&[0.0, 1.0], Vec::new()),
            cluster(&[2.0, 40.0], vec![slice(10.0, 200)]),
        ];
        // The two records of the alternate-less cluster still consume 0 and 1.
        assert_eq!(visited(&clusters, [0.0; 3]), [(2, 100, 12), (3, 200, 6)]);
    }

    #[test]
    fn an_empty_cluster_does_not_shift_the_clusters_after_it() {
        let clusters = [
            cluster(&[], vec![slice(10.0, 200)]),
            cluster(&[40.0], vec![slice(10.0, 300)]),
        ];
        assert_eq!(visited(&clusters, [0.0; 3]), [(0, 300, 6)]);
    }

    #[test]
    fn distance_is_measured_to_the_instance_translation() {
        let clusters = [cluster(&[40.0], vec![slice(10.0, 200)])];
        // Camera parked on the instance: the base slice, despite the world-space
        // translation being well past the threshold.
        assert_eq!(visited(&clusters, [40.0, 0.0, 0.0]), [(0, 100, 12)]);
    }

    #[test]
    fn a_ladder_picks_the_highest_alternate_at_or_below_the_distance() {
        let alts = vec![slice(10.0, 200), slice(20.0, 300)];
        let clusters = [cluster(&[5.0, 15.0, 25.0], alts)];
        assert_eq!(
            visited(&clusters, [0.0; 3]),
            [(0, 100, 12), (1, 200, 6), (2, 300, 6)]
        );
    }
}
