use super::*;
use crate::gfx::render_types::{BucketSpans, bucket_regions};

fn status(outcome: u32, bucket: u32) -> u32 {
    outcome | (bucket << CullStatus::BUCKET_SHIFT)
}

// Three buckets over 300 records, the last 20 skinned: bucket 1 takes records
// 40..140 and the runtime reserve 200..280, bucket 2 only 100..180.
fn regions() -> [GpuBucketRegion; MAX_SHADER_BUCKETS] {
    bucket_regions(&[
        BucketSpans {
            records: 0..300,
            runtime: 0..0,
        },
        BucketSpans {
            records: 40..140,
            runtime: 200..280,
        },
        BucketSpans {
            records: 100..180,
            runtime: 0..0,
        },
    ])
}

fn layout(regions: &[GpuBucketRegion; MAX_SHADER_BUCKETS]) -> StreamLayout<'_> {
    StreamLayout {
        regions,
        bucket_count: 3,
        skinned_base: 280,
        records: 300,
    }
}

// A scattered mix of outcomes and buckets, records routed outside their
// bucket's region among them.
fn words() -> Vec<u32> {
    (0..300u32)
        .map(|r| {
            let outcome = match r % 7 {
                0 | 3 => CullStatus::CULLED,
                5 => CullStatus::REDRAW,
                _ => CullStatus::DRAWN,
            };
            status(outcome, (r * 5 / 3) % 3)
        })
        .collect()
}

// The kernels' algorithm run on the CPU: count, scan, then place each kept
// record at its group's place plus its rank among the group's records of its
// stream.
fn scattered(words: &[u32], drawn: u32, layout: &StreamLayout<'_>) -> Vec<Vec<u32>> {
    let mut counts = group_counts(words, drawn, layout);
    let totals = scan_group_counts(&mut counts, layout);
    let groups = cull_groups(layout.records);
    let mut lists: Vec<Vec<Option<u32>>> = totals
        .iter()
        .map(|&t| alloc::vec![None; t as usize])
        .collect();
    for group in 0..groups {
        let first = group * CULL_GROUP_RECORDS;
        let members: Vec<(u32, u32)> = (first..(first + CULL_GROUP_RECORDS).min(layout.records))
            .filter(|&r| CullStatus::outcome(words[r as usize]) == drawn)
            .filter_map(|r| {
                layout
                    .stream(r, CullStatus::bucket(words[r as usize]))
                    .map(|s| (r, s))
            })
            .collect();
        for (i, &(record, stream)) in members.iter().enumerate() {
            let rank =
                u32::try_from(members[..i].iter().filter(|m| m.1 == stream).count()).expect("rank");
            let place = counts[group_count_index(stream, group, groups)] + rank;
            if let Some(slot) = lists[stream as usize].get_mut(place as usize) {
                *slot = Some(record);
            }
        }
    }
    lists
        .into_iter()
        .map(|l| {
            l.into_iter()
                .map(|r| r.expect("every place filled"))
                .collect()
        })
        .collect()
}

#[test]
fn every_list_keeps_record_order_and_only_its_streams_draws() {
    let regions = regions();
    let layout = layout(&regions);
    let words = words();
    let lists = compact(&words, CullStatus::DRAWN, &layout);
    for (stream, list) in lists.iter().enumerate() {
        assert!(
            list.windows(2).all(|w| w[0] < w[1]),
            "stream {stream} in order"
        );
        for &r in list {
            let word = words[r as usize];
            assert_eq!(CullStatus::outcome(word), CullStatus::DRAWN);
            assert_eq!(
                layout.stream(r, CullStatus::bucket(word)),
                Some(u32::try_from(stream).expect("stream"))
            );
        }
    }
    let kept: usize = lists.iter().map(Vec::len).sum();
    let expected = (0..300u32)
        .filter(|&r| CullStatus::outcome(words[r as usize]) == CullStatus::DRAWN)
        .filter(|&r| {
            layout
                .stream(r, CullStatus::bucket(words[r as usize]))
                .is_some()
        })
        .count();
    assert_eq!(kept, expected);
}

#[test]
fn the_kernels_algorithm_builds_the_spec_lists() {
    let regions = regions();
    let words = words();
    for layout in [
        layout(&regions),
        StreamLayout {
            bucket_count: 1,
            ..layout(&regions)
        },
        StreamLayout {
            records: 1,
            ..layout(&regions)
        },
    ] {
        for drawn in [CullStatus::DRAWN, CullStatus::REDRAW] {
            assert_eq!(
                scattered(&words, drawn, &layout),
                compact(&words, drawn, &layout),
                "{} buckets over {} records, outcome {drawn}",
                layout.bucket_count,
                layout.records
            );
        }
    }
}

#[test]
fn the_skinned_tail_is_bucket_zeros_own_stream() {
    let regions = regions();
    let layout = layout(&regions);
    assert_eq!(layout.stream(279, 0), Some(0));
    assert_eq!(layout.stream(280, 0), Some(SKINNED_STREAM));
    assert_eq!(
        layout.slots(0),
        StreamSlots {
            region: 0,
            first: 0,
            capacity: 280
        }
    );
    assert_eq!(
        layout.slots(SKINNED_STREAM),
        StreamSlots {
            region: 0,
            first: 280,
            capacity: 20
        }
    );
}

#[test]
fn a_record_outside_its_buckets_region_is_dropped() {
    let regions = regions();
    let layout = layout(&regions);
    assert_eq!(layout.stream(39, 1), None);
    assert_eq!(layout.stream(40, 1), Some(1));
    assert_eq!(layout.stream(150, 1), None);
    assert_eq!(layout.stream(250, 1), Some(1));
    assert_eq!(layout.stream(250, 2), None);
    assert_eq!(layout.stream(250, 9), None);
}

#[test]
fn a_single_stream_takes_every_record_whatever_its_bucket() {
    let regions = regions();
    let layout = StreamLayout {
        bucket_count: 1,
        ..layout(&regions)
    };
    assert_eq!(layout.stream(10, 2), Some(0));
    assert_eq!(layout.stream(290, 1), Some(SKINNED_STREAM));
    assert_eq!(layout.streams().collect::<Vec<_>>(), [0, SKINNED_STREAM]);
    assert_eq!(layout.slots(1), StreamSlots::default());
}

#[test]
fn a_streams_slots_are_its_regions() {
    let regions = regions();
    let layout = layout(&regions);
    assert_eq!(
        layout.slots(1),
        StreamSlots {
            region: 300,
            first: 0,
            capacity: 180
        }
    );
    assert_eq!(
        layout.slots(2),
        StreamSlots {
            region: 480,
            first: 0,
            capacity: 80
        }
    );
    assert_eq!(layout.slots(3), StreamSlots::default());
    assert_eq!(
        layout.streams().collect::<Vec<_>>(),
        [0, 1, 2, SKINNED_STREAM]
    );
}

#[test]
fn no_stream_outgrows_its_capacity() {
    let regions = regions();
    let layout = layout(&regions);
    let all = alloc::vec![status(CullStatus::DRAWN, 1); 300];
    let mut counts = group_counts(&all, CullStatus::DRAWN, &layout);
    let totals = scan_group_counts(&mut counts, &layout);
    for stream in layout.streams() {
        assert!(totals[stream as usize] <= layout.slots(stream).capacity);
    }
    assert_eq!(totals[1], 180);
    assert_eq!(
        totals[SKINNED_STREAM as usize], 0,
        "skinned records route to bucket 1"
    );
}

#[test]
fn a_region_cull_never_covers_more_than_its_capacity() {
    assert_eq!(culled_records(300, 512), 300);
    assert_eq!(culled_records(600, 512), 512);
    assert_eq!(culled_records(0, 512), 0);
    // The clamped cull's scratch and lists fit the region they were sized for.
    let records = culled_records(600, 512);
    assert_eq!(group_count_words(records), group_count_words(512));
}

#[test]
fn scratch_indexing_is_stream_major() {
    assert_eq!(cull_groups(0), 0);
    assert_eq!(cull_groups(64), 1);
    assert_eq!(cull_groups(65), 2);
    assert_eq!(group_count_words(65), 2 * DRAW_STREAMS);
    assert_eq!(group_count_index(0, 1, 2), 1);
    assert_eq!(group_count_index(1, 0, 2), 2);
    assert_eq!(
        group_count_index(SKINNED_STREAM, 1, 2),
        group_count_words(65) - 1
    );
}

#[test]
fn count_slots_sit_aligned_past_the_commands() {
    assert_eq!(count_slot_offset(0), 0);
    assert_eq!(count_slot_offset(1), COUNT_SLOT_ALIGN);
    assert_eq!(count_slot_offset(24 * 300), 7424);
    assert_eq!(COUNT_SLOT_BYTES, 4 * DRAW_STREAMS as u64);
}

#[test]
fn execution_range_layout() {
    assert_eq!(size_of::<GpuExecutionRange>(), 8);
    assert_eq!(core::mem::offset_of!(GpuExecutionRange, location), 0);
    assert_eq!(core::mem::offset_of!(GpuExecutionRange, length), 4);
}

// The constants the shaders lock to these.
#[test]
fn shader_constants_match() {
    let src = crate::render::shader::embedded::CULL_COMMON;
    let uint = |name| crate::render::shader::consts::uint(src, name);
    assert_eq!(uint("CULL_GROUP_RECORDS"), CULL_GROUP_RECORDS as usize);
    assert_eq!(uint("DRAW_STREAMS"), DRAW_STREAMS);
    assert_eq!(uint("SKINNED_STREAM"), SKINNED_STREAM as usize);
}

// Matches `CompactParams` in cull_compact.hlsl: eight u32s, two 16-byte rows.
#[test]
fn compact_params_layout() {
    use core::mem::offset_of;
    assert_eq!(size_of::<GpuCompactParams>(), 32);
    assert_eq!(offset_of!(GpuCompactParams, object_count), 0);
    assert_eq!(offset_of!(GpuCompactParams, bucket_count), 4);
    assert_eq!(offset_of!(GpuCompactParams, skinned_base), 8);
    assert_eq!(offset_of!(GpuCompactParams, drawn), 12);
    assert_eq!(offset_of!(GpuCompactParams, coarse_offset), 16);
    assert_eq!(offset_of!(GpuCompactParams, coarse_count), 20);
    assert_eq!(offset_of!(GpuCompactParams, full_count), 24);
    assert_eq!(offset_of!(GpuCompactParams, _pad), 28);
}

#[test]
fn a_pack_draws_planet_tiles_coarse_only_when_given_a_grid() {
    let plain = GpuCompactParams {
        object_count: 4,
        bucket_count: 1,
        skinned_base: 4,
        drawn: 1,
        ..GpuCompactParams::default()
    };
    assert_eq!(plain.with_capture_tiles(None), plain);
    let tiles = crate::render::scene_state::CaptureTiles {
        offset: 1536,
        count: 96,
        full_count: 1536,
    };
    let coarse = plain.with_capture_tiles(Some(tiles));
    assert_eq!(
        (coarse.coarse_offset, coarse.coarse_count, coarse.full_count),
        (1536, 96, 1536)
    );
    assert_eq!(coarse.object_count, 4);
}
