// Shared fixtures for the crate's unit tests, and the timing harness its
// in-module benchmarks report through.
//
// The draw records are wide enough that constructing one inline dominates a
// test; they are built here once so a test states only the fields it is about.
//
// The name interner here stands in for the production one, which keeps a
// per-thread table and so lives in the std-linked crate above this one. Asset
// tests that deserialize a named reference need names to resolve to dense
// declaration-ordered ids; this reproduces exactly that, and nothing else the
// production interner does.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use std::println;
use std::time::Instant;

use crate::ecs::asset_id::AssetId;
use crate::gfx::render_types::{DrawObject, MaterialUniforms, SkinnedDrawObject};
use crate::gfx::transform::IDENTITY;

// One measured pass runs at least this long before its time is trusted.
const TARGET_NS: u128 = 200_000_000;
const MAX_ITERS: u64 = 1 << 20;

// Time `body` over a calibrated iteration count and report its per-item cost
// beside the allocations one item causes. `items` is how many units of work one
// call performs, so a number is comparable across fixture sizes.
//
// The allocation pass is separate, so the counter reads sit outside the timed
// window. Those counters are process-global, which is why every benchmark here
// asks for `--test-threads=1`: one running beside another reads the other's
// allocations as its own.
pub(crate) fn bench<R>(name: &str, items: u64, mut body: impl FnMut() -> R) {
    let mut iters: u64 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            core::hint::black_box(body());
        }
        if start.elapsed().as_nanos() >= TARGET_NS || iters >= MAX_ITERS {
            break;
        }
        iters = iters.saturating_mul(4).min(MAX_ITERS);
    }

    let start = Instant::now();
    for _ in 0..iters {
        core::hint::black_box(body());
    }
    let elapsed = start.elapsed();

    let before = concinnity_memory::alloc_count();
    for _ in 0..iters {
        core::hint::black_box(body());
    }
    let after = concinnity_memory::alloc_count();

    let units = (iters * items.max(1)) as f64;
    let per_item_ns = elapsed.as_secs_f64() * 1e9 / units;
    match (before, after) {
        (Some(before), Some(after)) => {
            let allocs = (after - before) as f64 / units;
            println!("  {name:<40} {per_item_ns:>10.2} ns/item {allocs:>10.3} allocs/item");
        }
        _ => println!("  {name:<40} {per_item_ns:>10.2} ns/item"),
    }
}

// Per-thread, like the production interner: the harness runs tests in parallel
// and each one resets before interning, so a shared table would let them
// clobber each other.
std::thread_local! {
    static NAMES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

fn resolve(name: &str) -> u32 {
    NAMES.with(|names| {
        let mut names = names.borrow_mut();
        match names.iter().position(|n| n == name) {
            Some(id) => id as u32,
            None => {
                names.push(name.to_string());
                (names.len() - 1) as u32
            }
        }
    })
}

// Empty this thread's interner and install it behind the schema crate's
// resolver seam. Call first in any test that interns or deserializes a named
// reference.
pub(crate) fn reset_interner() {
    concinnity_asset::set_name_resolver(resolve);
    NAMES.with(|names| names.borrow_mut().clear());
}

// Resolve `name` to its id, assigning the next one if it is new.
pub(crate) fn intern(name: &str) -> AssetId {
    AssetId(resolve(name))
}

// Pre-intern a batch of names in order, so ids are dense and follow the order
// a world would declare them in.
pub(crate) fn intern_all(names: &[&str]) {
    for n in names {
        resolve(n);
    }
}

// A populated static draw record with distinct values in every slot, so a test
// that reads one back can tell which field it got.
pub(crate) fn draw_object() -> DrawObject {
    DrawObject {
        vertex_offset: 0,
        vertex_count: 8,
        index_offset: 12,
        index_count: 36,
        base_vertex: 4,
        geometry_generation: 0,
        shader_bucket: 0,
        model: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [5.0, 6.0, 7.0, 1.0],
        ],
        texture_slot: 9,
        normal_map_slot: 2,
        material: MaterialUniforms {
            roughness: 0.3,
            metallic: 0.7,
            macro_variation: 0.5,
            terrain_blend: 0.0,
            tint: [0.1, 0.2, 0.3],
            _pad2: 0.0,
            emissive: [0.4, 0.5, 0.6],
            secondary_blend_sharpness: 0.5,
            albedo_secondary_index: 0,
            normal_secondary_index: 0,
            emissive_map_index: 0,
            orm_map_index: 0,
            alpha_cutoff: 0.0,
            opacity: 1.0,
            transparent: 0,
            see_through: 0,
        },
        visible: true,
        resident: true,
        bb_min: [-1.0, -2.0, -3.0],
        bb_max: [1.0, 2.0, 3.0],
        cull_distance: 42.0,
        lod_alternates: Vec::new(),
    }
}

// A skinned draw record at the origin with a unit bind-pose box.
pub(crate) fn skinned_draw_object() -> SkinnedDrawObject {
    SkinnedDrawObject {
        vertex_base: 0,
        vertex_count: 10,
        index_offset: 0,
        index_count: 30,
        model: IDENTITY,
        texture_slot: 5,
        normal_map_slot: 2,
        material: MaterialUniforms::DEFAULT,
        visible: true,
        joint_count: 4,
        local_bb_min: [-1.0, -1.0, -1.0],
        local_bb_max: [1.0, 1.0, 1.0],
        lod_alternates: Vec::new(),
    }
}
