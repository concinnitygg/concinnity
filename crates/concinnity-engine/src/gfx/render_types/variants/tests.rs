use alloc::vec::Vec;
use core::mem::offset_of;

use concinnity_core::math::{cos, sqrt};

use super::*;
use crate::gfx::render_types::{FarFieldTint, draw_args_flags};
use crate::render::shader::consts;

// The default's family on a planet, from bucket 2: canopy, canopy lit from the
// planet's sky, ground, ground lit from the sky.
const CANOPY: u32 = 2;
const CANOPY_SKY: u32 = 3;
const GROUND: u32 = 4;
const GROUND_SKY: u32 = 5;

fn planet_route() -> GpuVariantRoute {
    GpuVariantRoute::new(
        [GROUND, CANOPY, GROUND_SKY, CANOPY_SKY],
        VARIANT_FAR | VARIANT_DAYLIGHT,
    )
}

const RADIUS: f32 = 50_000.0;

// A planet of `RADIUS` centered below the origin, seen from `camera_reach` up
// the sky's height ramp, the sky reaching in past 0.5 deg.
fn sky(camera_reach: f32) -> DaylightBlock {
    DaylightBlock {
        center: [0.0, -RADIUS, 0.0, 1.0],
        camera_up: [0.0, 1.0, 0.0, camera_reach],
        reach: [
            cos(0.5_f32.to_radians()),
            cos(3.0_f32.to_radians()),
            RADIUS,
            5_000.0,
        ],
        ..DaylightBlock::OFF
    }
}

// The blades hand over at 60 m, on a planet seen from the ground.
fn block() -> GpuCullVariants {
    let mut block = GpuCullVariants::default();
    block.routes[0] = planet_route();
    block.set_view(60.0, &sky(0.0));
    block
}

fn args(far_field: FarFieldTint) -> GpuDrawArgs {
    let route = if far_field.shades_ground() {
        Some(0)
    } else {
        None
    };
    GpuDrawArgs {
        index_count: 3,
        index_offset: 0,
        base_vertex: 0,
        flags: draw_args_flags(true, true, true)
            | super::super::draw_args_route_bits(route, far_field.grows_grass()),
    }
}

fn grass() -> GpuDrawArgs {
    args(FarFieldTint::of(1 << 8))
}

// A box `half` meters each way around the origin, 1 m tall.
fn tile(half: f32) -> ([f32; 3], [f32; 3]) {
    ([-half, 0.0, -half], [half, 1.0, half])
}

const EYE: [f32; 3] = [0.0, 2.0, 0.0];

#[test]
fn a_route_packs_a_bucket_per_feature_combination() {
    let r = planet_route();
    assert_eq!(r.bucket(0), GROUND);
    assert_eq!(r.bucket(VARIANT_FAR), CANOPY);
    assert_eq!(r.bucket(VARIANT_DAYLIGHT), GROUND_SKY);
    assert_eq!(r.bucket(VARIANT_FAR | VARIANT_DAYLIGHT), CANOPY_SKY);
    assert_eq!(GpuVariantRoute::new([300, 1, 2, 3], 0).bucket(0), 0xFF);
}

// A draw whose bounds reach past where the blades hand over takes the canopy;
// one wholly within it, the ground. Unbounded counts as reaching.
#[test]
fn a_far_band_in_reach_takes_the_canopy() {
    let b = block();
    assert_eq!(b.bucket(&grass(), tile(50.0), EYE), CANOPY);
    assert_eq!(b.bucket(&grass(), tile(20.0), EYE), GROUND);
    // A draw the cull does not frustum-test reaches everywhere, whatever its
    // box says.
    let mut always = grass();
    always.flags &= !DrawArgsFlags::CULLABLE;
    assert_eq!(
        b.bucket(&always, ([f32::NAN; 3], [f32::NAN; 3]), EYE),
        CANOPY_SKY
    );
    assert_eq!(b.bucket(&always, tile(5.0), EYE), CANOPY_SKY);
}

// The hand-over follows the frame; with none yet every draw growing grass takes
// the canopy.
#[test]
fn routing_follows_the_frames_hand_over() {
    let mut b = block();
    b.far[0] = 120.0;
    assert_eq!(b.bucket(&grass(), tile(40.0), EYE), GROUND);
    b.far[0] = 20.0;
    assert_eq!(b.bucket(&grass(), tile(40.0), EYE), CANOPY);
    b.far[0] = 0.0;
    assert_eq!(b.bucket(&grass(), tile(1.0), EYE), CANOPY);
}

// A planet's ground growing no grass has no band: the ground, wherever it lies.
#[test]
fn a_grassless_planet_ground_takes_the_ground() {
    use crate::render::environment::grass::CanopyRange;
    let bare = FarFieldTint::of(
        CanopyRange {
            first: 0,
            count: 0,
            planet: true,
        }
        .packed(),
    );
    assert!(bare.shades_ground() && !bare.grows_grass());
    let mut b = block();
    b.far[0] = 0.0;
    assert_eq!(b.bucket(&args(bare), tile(50.0), EYE), GROUND);
}

// A record without a route keeps its base bucket, clamped to nothing here.
#[test]
fn a_record_without_a_route_keeps_its_bucket() {
    let mut a = args(FarFieldTint::NONE);
    a.flags |= super::super::draw_args_bucket_bits(7);
    assert_eq!(block().bucket(&a, tile(50.0), EYE), 7);
}

// Points over a box, corners included.
fn box_points(bb: ([f32; 3], [f32; 3])) -> Vec<[f32; 3]> {
    let steps = 8;
    let at = |i: usize, t: usize| bb.0[i] + (bb.1[i] - bb.0[i]) * (t as f32 / steps as f32);
    let mut points = Vec::new();
    for x in 0..=steps {
        for y in 0..=steps {
            for z in 0..=steps {
                points.push([at(0, x), at(1, y), at(2, z)]);
            }
        }
    }
    points
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
}

// An eye walking away from a 40 m tile and back: the tile takes the ground
// only while no point of it is past the hand-over, so the canopy's weight is 0
// over every pixel the ground draws and the switch shows no seam; it switches
// once each way.
#[test]
fn routing_follows_the_eye_across_the_hand_over_without_a_seam() {
    let b = block();
    let bb = tile(20.0);
    let points = box_points(bb);
    let walk: Vec<f32> = (0..=120).chain((0..120).rev()).map(|x| x as f32).collect();
    let mut routes = Vec::new();
    for x in walk {
        let eye = [x, 1.7, 0.0];
        let bucket = b.bucket(&grass(), bb, eye);
        let farthest = points
            .iter()
            .map(|p| distance(eye, *p))
            .fold(0.0_f32, f32::max);
        match bucket {
            GROUND => assert!(farthest <= b.far[0], "x {x}: ground, a pixel at {farthest}"),
            CANOPY => assert!(
                farthest > b.far[0] - 1e-3,
                "x {x}: canopy within {farthest}"
            ),
            other => panic!("x {x}: routed to {other}"),
        }
        routes.push(bucket);
    }
    routes.dedup();
    assert_eq!(routes, [GROUND, CANOPY, GROUND]);
}

// `daylight_reach` in daylight.hlsl: how much a point's own sky stands in for
// the camera's, before the ground's weight.
fn sky_reach(sky: &DaylightBlock, p: [f32; 3]) -> f32 {
    let d = [
        p[0] - sky.center[0],
        p[1] - sky.center[1],
        p[2] - sky.center[2],
    ];
    let len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    let c = (d[0] * sky.camera_up[0] + d[1] * sky.camera_up[1] + d[2] * sky.camera_up[2]) / len;
    let far = 1.0 - concinnity_core::math::smoothstep(sky.reach[1], sky.reach[0], c);
    far.max(sky.camera_up[3])
}

// Near the camera on the ground the sky reaches nothing, so the box takes the
// variant without it; the sky reaching any point of it takes the one with it.
#[test]
fn the_sky_picks_the_variant_with_it_only_where_it_reaches() {
    let b = block();
    // 0.5 deg round a 50 km planet is 436 m.
    assert_eq!(b.bucket(&grass(), tile(300.0), EYE), CANOPY);
    assert_eq!(b.bucket(&grass(), tile(10.0), EYE), GROUND);
    assert_eq!(b.bucket(&grass(), tile(500.0), EYE), CANOPY_SKY);
    // A high camera: the sky reaches everything.
    let mut high = block();
    high.set_view(60.0, &sky(0.01));
    assert_eq!(high.bucket(&grass(), tile(10.0), EYE), GROUND_SKY);
    // No planet sky in this view (a probe face, a world without one).
    let mut off = block();
    off.set_view(60.0, &DaylightBlock::OFF);
    assert_eq!(off.bucket(&grass(), tile(500.0), EYE), CANOPY);
}

// Wherever the test keeps the variant without the sky, the sky reaches no
// point of the box: every sampled point of boxes swept out past the cone, and
// of thin boxes leaning along it, has a reach of exactly 0.
#[test]
fn the_corner_test_never_drops_a_reached_point() {
    let b = block();
    let s = sky(0.0);
    let mut kept = 0;
    let mut dropped = 0;
    for step in 0..120 {
        let reach = 380.0 + step as f32;
        for (lo, hi) in [
            ([reach - 60.0, -5.0, -20.0], [reach, 3.0, 20.0]),
            ([-reach, -2.0, -reach], [-reach + 15.0, 40.0, -reach + 15.0]),
            ([-10.0, -30.0, reach - 2.0], [10.0, -20.0, reach]),
        ] {
            let mut a = grass();
            a.flags &= !DrawArgsFlags::FAR_BAND;
            let bucket = b.bucket(&a, (lo, hi), EYE);
            if bucket == GROUND {
                kept += 1;
                for p in box_points((lo, hi)) {
                    assert_eq!(sky_reach(&s, p), 0.0, "{lo:?}..{hi:?} at {p:?}");
                }
            } else {
                assert_eq!(bucket, GROUND_SKY);
                dropped += 1;
            }
        }
    }
    assert!(kept > 0 && dropped > 0, "{kept} kept, {dropped} dropped");
}

// A bucket out of the view's live set is one no record picks: aloft only the
// sky's variants, without a sky only those without it, near the ground all.
#[test]
fn the_live_buckets_hold_every_pick_of_the_view() {
    let all_eight = 0xFF;
    let mut b = block();
    b.routes[1] = GpuVariantRoute::new([6, 6, 6, 6], 0);
    let plain =
        !((1 << CANOPY) | (1 << CANOPY_SKY) | (1 << GROUND) | (1 << GROUND_SKY)) & all_eight;
    assert_eq!(b.live_buckets(8), all_eight);
    b.set_view(60.0, &sky(0.2));
    assert_eq!(
        b.live_buckets(8),
        plain | (1 << CANOPY_SKY) | (1 << GROUND_SKY)
    );
    b.set_view(60.0, &DaylightBlock::OFF);
    assert_eq!(b.live_buckets(8), plain | (1 << CANOPY) | (1 << GROUND));
    assert_eq!(b.live_buckets(3), 0b111, "within the world's buckets");
    // Every pick of every view lies in its live set.
    for view in [sky(0.0), sky(0.2), DaylightBlock::OFF] {
        b.set_view(60.0, &view);
        let live = b.live_buckets(8);
        for half in [5.0, 50.0, 300.0, 600.0] {
            for args in [grass(), args(FarFieldTint::NONE)] {
                let bucket = b.bucket(&args, tile(half), EYE);
                assert_ne!(live & (1 << bucket), 0, "bucket {bucket}");
            }
        }
    }
}

#[test]
fn a_region_maps_its_spans_back_to_back() {
    let r = GpuBucketRegion {
        first: 10,
        count: 5,
        runtime_first: 100,
        runtime_count: 3,
        offset: 40,
        _pad: [0; 3],
    };
    assert_eq!(r.len(), 8);
    assert_eq!(r.command(9), None);
    assert_eq!(r.command(10), Some(40));
    assert_eq!(r.command(14), Some(44));
    assert_eq!(r.command(15), None);
    assert_eq!(r.command(100), Some(45));
    assert_eq!(r.command(102), Some(47));
    assert_eq!(r.command(103), None);
    let regions = bucket_regions(&[
        BucketSpans {
            records: 0..20,
            runtime: 0..0,
        },
        BucketSpans {
            records: 4..6,
            runtime: 12..15,
        },
    ]);
    assert_eq!((regions[0].offset, regions[0].len()), (0, 20));
    assert_eq!((regions[1].offset, regions[1].len()), (20, 5));
    assert!(regions[2].is_empty());
    assert_eq!(region_commands(&regions, 2), 25);
}

// A runtime span growing with the draws streamed in moves every later region
// along, and starts past each bucket's build-time records.
#[test]
fn covering_a_growing_runtime_span_repacks_the_regions() {
    let mut block = GpuCullVariants {
        regions: bucket_regions(&[
            BucketSpans {
                records: 0..30,
                runtime: 0..0,
            },
            BucketSpans {
                records: 4..12,
                runtime: 0..0,
            },
            BucketSpans {
                records: 0..0,
                runtime: 0..0,
            },
        ]),
        ..GpuCullVariants::default()
    };
    block.cover_runtime(3, 30, 10..20);
    let r = &block.regions;
    assert_eq!(
        (r[0].offset, r[0].len()),
        (0, 30),
        "the default spans every record"
    );
    assert_eq!((r[1].runtime_first, r[1].runtime_count), (12, 8));
    assert_eq!((r[1].offset, r[1].len()), (30, 16));
    assert_eq!((r[2].runtime_first, r[2].runtime_count), (10, 10));
    assert_eq!(r[2].offset, 46);
    assert!(r[3].is_empty(), "past the world's buckets");
    block.cover_runtime(3, 24, 10..10);
    assert_eq!(block.regions[0].len(), 24);
    assert_eq!(block.regions[2].len(), 0);
}

#[test]
fn cull_variants_layout_matches_shaders() {
    assert_eq!(size_of::<GpuBucketRegion>(), 32);
    assert_eq!(offset_of!(GpuBucketRegion, first), 0);
    assert_eq!(offset_of!(GpuBucketRegion, count), 4);
    assert_eq!(offset_of!(GpuBucketRegion, runtime_first), 8);
    assert_eq!(offset_of!(GpuBucketRegion, runtime_count), 12);
    assert_eq!(offset_of!(GpuBucketRegion, offset), 16);
    assert_eq!(offset_of!(GpuBucketRegion, _pad), 20);
    assert_eq!(size_of::<GpuVariantRoute>(), 16);
    assert_eq!(offset_of!(GpuVariantRoute, buckets), 0);
    assert_eq!(offset_of!(GpuVariantRoute, features), 4);
    assert_eq!(offset_of!(GpuVariantRoute, _pad), 8);
    assert_eq!(offset_of!(GpuCullVariants, far), 0);
    assert_eq!(offset_of!(GpuCullVariants, day_center), 16);
    assert_eq!(offset_of!(GpuCullVariants, day_camera_up), 32);
    assert_eq!(offset_of!(GpuCullVariants, day_reach), 48);
    assert_eq!(offset_of!(GpuCullVariants, routes), 64);
    assert_eq!(
        offset_of!(GpuCullVariants, regions),
        64 + 16 * MAX_VARIANT_ROUTES
    );
    assert_eq!(
        size_of::<GpuCullVariants>(),
        64 + 16 * MAX_VARIANT_ROUTES + 32 * MAX_SHADER_BUCKETS
    );
    // Inline on Metal, which binds at most 4 KB of bytes a slot.
    assert!(size_of::<GpuCullVariants>() <= 4096);
}

#[test]
fn variant_constants_match_the_cull_shader() {
    let src = crate::render::shader::embedded::CULL_COMMON;
    for (name, value) in [
        ("DRAW_FAR_BAND", DrawArgsFlags::FAR_BAND as usize),
        ("DRAW_ROUTE_SHIFT", DrawArgsFlags::ROUTE_SHIFT as usize),
        ("DRAW_ROUTE_MASK", DrawArgsFlags::ROUTE_MASK as usize),
        ("MAX_SHADER_BUCKETS", MAX_SHADER_BUCKETS),
        ("MAX_VARIANT_ROUTES", MAX_VARIANT_ROUTES),
        ("VARIANT_FAR", VARIANT_FAR as usize),
        ("VARIANT_DAYLIGHT", VARIANT_DAYLIGHT as usize),
    ] {
        assert_eq!(consts::uint(src, name), value, "{name}");
    }
    assert_eq!(
        consts::float(src, "DAYLIGHT_COS_MARGIN"),
        DAYLIGHT_COS_MARGIN
    );
    // The kernel reads the block as 16-byte rows.
    assert_eq!(
        consts::uint(src, "CULL_VARIANT_ROUTE_ROW"),
        offset_of!(GpuCullVariants, routes) / 16
    );
    assert_eq!(
        consts::uint(src, "CULL_VARIANT_REGION_ROW"),
        offset_of!(GpuCullVariants, regions) / 16
    );
    assert_eq!(size_of::<GpuBucketRegion>(), 2 * 16);
}
