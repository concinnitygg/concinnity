use super::*;
use crate::gfx::render_types::MaterialUniforms;
use crate::render::rt_geom::RT_SKINNED_FLAG;
use crate::test_support::{draw_object, skinned_draw_object};

// A BLAS stands in as a label, so a test can tell which one an instance names.
type Blas = u32;

// What one laid-out instance referenced: the head BLAS label, a fresh slot, or a
// skinned BLAS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Head(usize, Blas),
    Fresh(usize),
    Skinned(usize),
}

type Book = AccelBook<Blas, (u32, Target, [f32; 4])>;

fn make(model: Mat4, id: u32, blas: InstanceBlas<'_, Blas>) -> (u32, Target, [f32; 4]) {
    let target = match blas {
        InstanceBlas::Head { index, blas } => Target::Head(index, *blas),
        InstanceBlas::Fresh { index } => Target::Fresh(index),
        InstanceBlas::Skinned { n } => Target::Skinned(n),
    };
    (id, target, model[3])
}

fn at(x: f32) -> Mat4 {
    let mut m = crate::transform::IDENTITY;
    m[3][0] = x;
    m
}

// Draw object `tag`, placed at x = `tag` over its own index slice.
fn object(tag: usize) -> DrawObject {
    let mut o = draw_object();
    o.index_offset = tag * 100;
    o.model = at(tag as f32);
    o
}

fn cluster(instances: &[f32]) -> InstancedCluster {
    InstancedCluster {
        vertex_offset: 0,
        vertex_count: 3,
        index_offset: 900,
        index_count: 3,
        texture_slot: 1,
        normal_map_slot: 1,
        material: MaterialUniforms::DEFAULT,
        cluster_bb_min: [-1.0; 3],
        cluster_bb_max: [1.0; 3],
        local_bb_min: [-1.0; 3],
        local_bb_max: [1.0; 3],
        cull_distance: 0.0,
        instances: instances.iter().map(|&x| at(x)).collect(),
        lod_alternates: Vec::new(),
    }
}

// Objects 0..n, with object 1 evicted, plus one two-instance cluster. BLAS are
// labeled 10, 11, ... in build order.
fn seeded(n: usize) -> (Vec<DrawObject>, Book) {
    let mut draw: Vec<DrawObject> = (0..n).map(object).collect();
    if let Some(o) = draw.get_mut(1) {
        o.resident = false;
    }
    let clusters = [cluster(&[100.0, 101.0])];
    let seed = SeedSet::new(&draw, &clusters, false);
    let blas = (0..(seed.objects.len() + seed.clusters.len()) as u32)
        .map(|b| 10 + b)
        .collect();
    let book = Book::new(&seed, blas, &draw, 4).expect("one BLAS per seed entry");
    (draw, book)
}

#[test]
fn a_seed_skips_what_cannot_be_traced() {
    let mut draw = vec![object(0), object(1), object(2)];
    draw[1].resident = false;
    draw[2].index_count = 2;
    let clusters = [cluster(&[]), cluster(&[1.0])];
    let seed = SeedSet::new(&draw, &clusters, false);
    assert_eq!(seed.objects, [0]);
    assert_eq!(seed.clusters.len(), 1);
    assert!(SeedSet::new(&[], &[], false).is_empty());
}

#[test]
fn a_seed_builds_whenever_its_book_would_not_start_out_spent() {
    let draw = [object(0)];
    let clusters = [cluster(&[1.0])];
    for (objects, clusters) in [
        (&draw[..0], &clusters[..0]),
        (&draw[..], &clusters[..0]),
        (&draw[..0], &clusters[..]),
    ] {
        let seed = SeedSet::new(objects, clusters, false);
        let blas = (0..(seed.objects.len() + seed.clusters.len()) as u32).collect();
        let book = Book::new(&seed, blas, objects, 0).expect("one BLAS per seed entry");
        for skinned_present in [false, true] {
            assert_eq!(
                seed.builds_nothing(skinned_present),
                book.is_spent(skinned_present)
            );
        }
    }
    assert!(SeedSet::new(&[], &[], false).builds_nothing(false));
    assert!(!SeedSet::new(&[], &[], false).builds_nothing(true));
}

#[test]
fn a_book_needs_one_blas_per_seed_entry() {
    let draw = vec![object(0)];
    let seed = SeedSet::new(&draw, &[], false);
    assert!(Book::new(&seed, vec![], &draw, 0).is_err());
}

#[test]
fn instances_follow_draws_then_clusters_then_skinned() {
    let (draw, mut book) = seeded(3);
    let skinned = [skinned_draw_object(), skinned_draw_object()];
    book.select_skinned(Some(&skinned));
    book.fill_instances(&draw, Some(&skinned), make);
    let ids: Vec<u32> = book.instances().iter().map(|i| i.0).collect();
    assert_eq!(ids, [0, 1, 2, 3, 4, 5]);
    let targets: Vec<Target> = book.instances().iter().map(|i| i.1).collect();
    assert_eq!(
        targets,
        [
            Target::Head(0, 10),
            Target::Head(1, 11),
            // Both cluster instances share the cluster's one BLAS.
            Target::Head(2, 12),
            Target::Head(2, 12),
            Target::Skinned(0),
            Target::Skinned(1),
        ]
    );
    let xs: Vec<f32> = book.instances().iter().map(|i| i.2[0]).collect();
    assert_eq!(xs, [0.0, 2.0, 100.0, 101.0, 0.0, 0.0]);

    // One geometry entry per instance, in the same order.
    let table = book.geom_table();
    assert_eq!(table.len(), 6);
    assert_eq!(table[1].index_offset, 200);
    assert_eq!(table[2].index_offset, 900);
    assert_ne!(table[4].normal_index & RT_SKINNED_FLAG, 0);
    assert_eq!(table[0].normal_index & RT_SKINNED_FLAG, 0);
}

#[test]
fn a_static_layout_leaves_the_skinned_objects_out() {
    let (draw, mut book) = seeded(3);
    let skinned = [skinned_draw_object()];
    book.select_skinned(Some(&skinned));
    book.fill_instances(&draw, None, make);
    assert_eq!(book.instances().len(), 4);
}

#[test]
fn only_visible_skinned_objects_with_triangles_are_selected() {
    let (_, mut book) = seeded(1);
    let mut skinned = [
        skinned_draw_object(),
        skinned_draw_object(),
        skinned_draw_object(),
    ];
    skinned[0].visible = false;
    skinned[2].index_count = 0;
    skinned[1].vertex_base = 40;
    book.select_skinned(Some(&skinned));
    assert_eq!(book.visible_skinned(), [1]);
    assert_eq!(book.skinned_vertex_extent(&skinned), 50);
    book.fill_skinned_shapes(&skinned, 50);
    let shapes = book.skinned_shapes();
    assert_eq!(shapes.len(), 1);
    assert_eq!(shapes[0].vertex_extent, 50);
    book.select_skinned(None);
    assert!(book.visible_skinned().is_empty());
    assert_eq!(book.skinned_vertex_extent(&skinned), 0);
}

#[test]
fn a_still_scene_keeps_its_tlas_and_a_moved_one_rebuilds() {
    let (mut draw, mut book) = seeded(3);
    let mode = RtDynamicMode::Auto;
    let plan = book.plan(mode, false, None).expect("dynamic");
    assert_eq!(book.next_step(mode, &plan, &draw), RtStep::Keep);
    draw[2].model = at(9.0);
    assert_eq!(book.next_step(mode, &plan, &draw), RtStep::Tlas);
    assert_eq!(book.commit_static(), None);
    assert_eq!(book.next_step(mode, &plan, &draw), RtStep::Keep);
}

#[test]
fn an_evicted_object_waits_for_the_refresh() {
    let (mut draw, mut book) = seeded(3);
    let mode = RtDynamicMode::Tlas;
    let plan = book.plan(mode, false, None).expect("dynamic");
    draw[2].resident = false;
    assert_eq!(book.next_step(mode, &plan, &draw), RtStep::Keep);
}

#[test]
fn the_skinned_tail_is_dropped_once_skinned_geometry_goes() {
    let (draw, mut book) = seeded(3);
    let mode = RtDynamicMode::Auto;
    let skinned = [skinned_draw_object()];
    let plan = book.plan(mode, false, Some(&skinned)).expect("dynamic");
    assert_eq!(book.next_step(mode, &plan, &draw), RtStep::Skinned);
    assert!(book.replace_tail([77]).is_empty());
    book.commit_skinned();
    assert!(book.has_skinned());
    assert_eq!(book.blas(), [10, 11, 12, 77]);

    // Nothing moved, but the TLAS still references the skinned BLAS.
    let plan = book.plan(mode, false, None).expect("dynamic");
    assert_eq!(book.next_step(mode, &plan, &draw), RtStep::Tlas);
    assert_eq!(book.commit_static(), Some(vec![77]));
    assert!(!book.has_skinned());
    assert_eq!(book.blas(), [10, 11, 12]);
}

#[test]
fn a_refresh_reuses_unchanged_blas_and_orphans_the_rest() {
    let (mut draw, mut book) = seeded(4);
    // Objects 0, 2, 3 participate. Bring 1 back, rewrite 2 in place, drop 3.
    draw[1].resident = true;
    draw[2].geometry_generation += 1;
    draw[3].resident = false;
    let refresh = book.plan_refresh(&draw, false, RefreshMode::Reuse);
    assert_eq!(refresh.indices(), [0, 1, 2]);
    assert_eq!(refresh.fresh_slots().collect::<Vec<_>>(), [(1, 1), (2, 2)]);
    assert!(!book.refresh_leaves_nothing(&refresh));

    book.fill_refresh_instances(&refresh, &draw, make);
    let targets: Vec<Target> = book.instances().iter().map(|i| i.1).collect();
    assert_eq!(
        targets,
        [
            Target::Head(0, 10),
            Target::Fresh(1),
            Target::Fresh(2),
            // The cluster BLAS keeps its label and moves to the new head end.
            Target::Head(3, 13),
            Target::Head(3, 13),
        ]
    );

    let orphans = book.commit_refresh(refresh, vec![None, Some(20), Some(21)], &draw);
    assert_eq!(orphans, [11, 12]);
    assert_eq!(book.blas(), [10, 20, 21, 13]);
    assert_eq!(book.static_blas_count(), 4);
    assert_eq!(book.object_indices(), [0, 1, 2]);

    // The refreshed transforms are the ones the TLAS was built from.
    let mode = RtDynamicMode::Auto;
    let plan = book.plan(mode, false, None).expect("dynamic");
    assert_eq!(book.next_step(mode, &plan, &draw), RtStep::Keep);
}

#[test]
fn a_refresh_keeps_the_tail() {
    let (mut draw, mut book) = seeded(3);
    book.replace_tail([77, 78]);
    draw[1].resident = true;
    let refresh = book.plan_refresh(&draw, false, RefreshMode::Reuse);
    let fresh = vec![None, Some(20), None];
    let head: Vec<Blas> = book
        .refreshed_head(&refresh, &fresh)
        .into_iter()
        .copied()
        .collect();
    assert_eq!(head, [10, 20, 11, 12]);
    book.commit_refresh(refresh, fresh, &draw);
    assert_eq!(book.blas(), [10, 20, 11, 12, 77, 78]);
    assert!(book.has_skinned());
    assert_eq!(book.head(), [10, 20, 11, 12]);
}

#[test]
fn rebuild_all_builds_every_draw_blas() {
    let (draw, book) = seeded(3);
    let refresh = book.plan_refresh(&draw, false, RefreshMode::RebuildAll);
    assert_eq!(refresh.fresh_slots().count(), 2);
    let mut book = book;
    let orphans = book.commit_refresh(refresh, vec![Some(30), Some(31)], &draw);
    assert_eq!(orphans, [10, 11]);
    assert_eq!(book.blas(), [30, 31, 12]);
}

#[test]
fn a_mismatched_commit_changes_nothing_and_hands_the_fresh_blas_back() {
    let (mut draw, mut book) = seeded(3);
    draw[1].resident = true;
    let refresh = book.plan_refresh(&draw, false, RefreshMode::Reuse);
    // Slot 1 needs a fresh BLAS and none was built.
    assert!(book.check_refresh(&refresh, &[None, None, None]).is_err());
    assert!(
        book.check_refresh(&refresh, &[None, Some(20), None])
            .is_ok()
    );
    let back = book.commit_refresh(refresh, vec![Some(40), None, None], &draw);
    assert_eq!(back, [40]);
    assert_eq!(book.blas(), [10, 11, 12]);
    assert_eq!(book.object_indices(), [0, 2]);
}

#[test]
fn two_slots_cannot_claim_one_reused_blas() {
    let (draw, book) = seeded(3);
    // Object 0 moved to slot 1 too: a forged plan reusing BLAS 0 twice.
    let mut refresh = book.plan_refresh(&draw, false, RefreshMode::Reuse);
    refresh.plan.reuse = vec![Some(0), Some(0)];
    assert!(book.check_refresh(&refresh, &[None, None]).is_err());
}

#[test]
fn the_refreshed_head_mixes_kept_and_fresh_blas_in_slot_order() {
    let (mut draw, book) = seeded(4);
    // Objects 0, 2, 3 participate; bring 1 back and rewrite 3 in place.
    draw[1].resident = true;
    draw[3].geometry_generation += 1;
    let refresh = book.plan_refresh(&draw, false, RefreshMode::Reuse);
    let fresh = vec![None, Some(20), None, Some(21)];
    let head: Vec<Blas> = book
        .refreshed_head(&refresh, &fresh)
        .into_iter()
        .copied()
        .collect();
    // Kept 0, fresh 1, kept 2, fresh 3, then the cluster.
    assert_eq!(head, [10, 20, 11, 21, 13]);
}

#[test]
fn a_failed_refresh_is_planned_again() {
    let (_, mut book) = seeded(1);
    let mode = RtDynamicMode::Auto;
    assert_eq!(book.plan(mode, false, None).expect("dynamic").refresh, None);
    book.owe_refresh();
    assert_eq!(
        book.plan(mode, false, None).expect("dynamic").refresh,
        Some(RefreshMode::Reuse)
    );
    // Owed once: the refresh it planned settles the debt.
    assert_eq!(book.plan(mode, false, None).expect("dynamic").refresh, None);
}

#[test]
fn parked_orphans_wait_for_the_next_publish() {
    let (_, mut book) = seeded(1);
    book.park(vec![7, 8]);
    book.park(vec![9]);
    assert_eq!(book.take_parked(), [7, 8, 9]);
    assert!(book.take_parked().is_empty());
    book.park(vec![5]);
    // Teardown reaches parked BLAS too.
    assert!(book.drain_blas().any(|b| b == 5));
}

// The live TLAS stays live until one built after it publishes, so the BLAS a
// trace declares resident include the orphans parked meanwhile.
#[test]
fn the_traced_set_covers_every_blas_the_live_tlas_references() {
    let (mut draw, mut book) = seeded(3);
    book.replace_tail([77]);
    let live: Vec<Blas> = book.blas().to_vec();
    draw[2].resident = false;
    let refresh = book.plan_refresh(&draw, false, RefreshMode::Reuse);
    let orphans = book.commit_refresh(refresh, vec![None], &draw);
    assert_eq!(orphans, [11]);
    book.park(orphans);
    let traced: Vec<Blas> = book.traced_blas().copied().collect();
    assert!(
        live.iter().all(|b| traced.contains(b)),
        "{live:?} {traced:?}"
    );
    // Once a TLAS over the refreshed BLAS publishes, only those stay traced.
    book.take_parked();
    let traced: Vec<Blas> = book.traced_blas().copied().collect();
    assert_eq!(traced, book.blas());
}

#[test]
fn a_refresh_can_leave_nothing_to_trace() {
    let mut draw = vec![object(0)];
    let seed = SeedSet::new(&draw, &[], false);
    let mut book = Book::new(&seed, vec![10], &draw, 0).expect("one BLAS");
    draw[0].resident = false;
    let refresh = book.plan_refresh(&draw, false, RefreshMode::Reuse);
    assert!(book.refresh_leaves_nothing(&refresh));
    let orphans = book.commit_refresh(refresh, vec![], &draw);
    assert_eq!(orphans, [10]);
    assert!(book.is_empty());
}

#[test]
fn an_emptied_book_is_spent_only_once_skinned_geometry_cannot_rejoin() {
    let mut draw = vec![object(0)];
    let seed = SeedSet::new(&draw, &[], false);
    let mut book = Book::new(&seed, vec![10], &draw, 0).expect("one BLAS");
    assert!(!book.is_spent(false));

    book.replace_tail([77]);
    draw[0].resident = false;
    let refresh = book.plan_refresh(&draw, false, RefreshMode::Reuse);
    book.commit_refresh(refresh, vec![], &draw);
    assert!(book.is_empty());
    // The live TLAS still publishes the skinned tail.
    assert!(!book.is_spent(false));

    assert_eq!(book.release_skinned(), Some(vec![77]));
    assert!(!book.is_spent(true));
    assert!(book.is_spent(false));
}

#[test]
fn the_clock_only_moves_forward() {
    let (_, mut book) = seeded(1);
    assert_eq!(book.tick(), 1);
    assert_eq!(book.tick(), 2);
}

#[test]
fn a_missing_object_keeps_the_others_on_their_own_blas() {
    let (draw, mut book) = seeded(3);
    // The first participating index fell off the draw list: the second object
    // still traces the second BLAS, not the first.
    book.object_indices = vec![5, 2];
    book.fill_instances(&draw, None, make);
    let targets: Vec<Target> = book.instances().iter().map(|i| i.1).collect();
    assert_eq!(targets[0], Target::Head(1, 11));
}
