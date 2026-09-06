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

// How hard a benchmark pass drives its body.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pace {
    // Calibrate an iteration count, time the body, and report its per-item cost.
    Timed,
    // Run the body exactly once and report nothing. What a non-ignored test
    // drives, so a fixture that stopped building or started panicking fails on
    // an ordinary run instead of waiting for someone to pass `--ignored`.
    Once,
}

// Time `body` over a calibrated iteration count and report its per-item cost
// beside the allocations one item causes. `items` is how many units of work one
// call performs, so a number is comparable across fixture sizes.
//
// The allocation pass is separate, so the counter reads sit outside the timed
// window. This crate's test binary installs the tracking allocator, so the
// counters are always live here. They are process-global, which is why every
// benchmark asks for `--test-threads=1`: one running beside another reads the
// other's allocations as its own.
pub(crate) fn bench<R>(pace: Pace, name: &str, items: u64, body: impl FnMut() -> R) {
    bench_over(TARGET_NS, pace, name, items, body);
}

// `bench` against an explicit measurement window, so a test can drive the
// calibration loop without waiting out a real one.
fn bench_over<R>(target_ns: u128, pace: Pace, name: &str, items: u64, mut body: impl FnMut() -> R) {
    if pace == Pace::Once {
        core::hint::black_box(body());
        return;
    }

    let mut iters: u64 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            core::hint::black_box(body());
        }
        if start.elapsed().as_nanos() >= target_ns || iters >= MAX_ITERS {
            break;
        }
        iters = iters.saturating_mul(4).min(MAX_ITERS);
    }

    let start = Instant::now();
    for _ in 0..iters {
        core::hint::black_box(body());
    }
    let elapsed = start.elapsed();

    let before = crate::memory::alloc_count().unwrap_or(0);
    for _ in 0..iters {
        core::hint::black_box(body());
    }
    let allocated = crate::memory::alloc_count()
        .unwrap_or(0)
        .saturating_sub(before);

    let units = (iters * items.max(1)) as f64;
    let per_item_ns = elapsed.as_secs_f64() * 1e9 / units;
    let allocs = allocated as f64 / units;
    println!("  {name:<40} {per_item_ns:>10.2} ns/item {allocs:>10.3} allocs/item");
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

// Empty this thread's interner and install it behind the resolver seam. Call first in any test that interns or deserializes a named
// reference.
pub(crate) fn reset_interner() {
    crate::ecs::resolver::set_name_resolver(resolve);
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
            alpha_cutoff: 0.25,
            opacity: 1.0,
            tint: [0.1, 0.2, 0.3],
            _pad0: 0.0,
            emissive: [0.4, 0.5, 0.6],
            _pad1: 0.0,
            emissive_map_index: 0,
            orm_map_index: 0,
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

use serde::de::{self, Deserializer, Visitor};

// A name resolves to its byte length: a deterministic stand-in for the build's
// declaration-ordered resource tables. Names prefixed `unknown_` resolve to
// nothing, standing in for a reference no resource of that kind declares.
pub(crate) fn len_handle_resolver(name: &str) -> Option<u32> {
    if name.starts_with("unknown_") {
        None
    } else {
        Some(name.len() as u32)
    }
}

// A name resolves to its byte length, standing in for the build-time interner
// the handle resolvers fall back to.
pub(crate) fn len_name_resolver(name: &str) -> u32 {
    name.len() as u32
}

// Installs every seam with the stand-ins above.
pub(crate) fn install_resolvers() {
    crate::ecs::resolver::set_name_resolver(len_name_resolver);
    crate::ecs::resolver::set_texture_handle_resolver(len_handle_resolver);
    crate::ecs::resolver::set_audio_clip_handle_resolver(len_handle_resolver);
    crate::ecs::resolver::set_font_handle_resolver(len_handle_resolver);
    crate::ecs::resolver::set_mesh_handle_resolver(len_handle_resolver);
    crate::ecs::resolver::set_material_handle_resolver(len_handle_resolver);
    crate::ecs::resolver::set_skinned_mesh_handle_resolver(len_handle_resolver);
    crate::ecs::resolver::set_shader_handle_resolver(len_handle_resolver);
}

// Reports `None` from `deserialize_any`, the way an option-aware self-describing
// format does. serde_json only ever reports a `null` unit, so the optional
// reference helpers' `visit_none` arm needs this stand-in.
pub(crate) struct NoneDeserializer;

impl<'de> Deserializer<'de> for NoneDeserializer {
    type Error = de::value::Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_none()
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU32, Ordering};

    // A single-run pass drives the body exactly once and reports nothing.
    #[test]
    fn a_single_run_pass_drives_the_body_once() {
        let runs = AtomicU32::new(0);
        bench(Pace::Once, "test/once", 4, || {
            runs.fetch_add(1, Ordering::Relaxed)
        });
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    // A measured pass calibrates an iteration count against its window, then
    // runs that count twice more: once timed, once counting allocations. The
    // window here is a microsecond rather than the real fifth of a second, so
    // the loop is driven rather than waited out.
    #[test]
    fn a_measured_pass_calibrates_then_times_and_counts() {
        let runs = AtomicU32::new(0);
        bench_over(1_000, Pace::Timed, "test/timed", 1, || {
            runs.fetch_add(1, Ordering::Relaxed)
        });
        assert!(
            runs.load(Ordering::Relaxed) >= 3,
            "a measured pass runs the body far more than a single-run one"
        );
    }

    // A body slow enough to fill the window on its first iteration never
    // multiplies the count, so the calibration cannot run away.
    #[test]
    fn a_body_that_fills_the_window_stays_at_one_iteration() {
        let runs = AtomicU32::new(0);
        bench_over(0, Pace::Timed, "test/wide", 0, || {
            runs.fetch_add(1, Ordering::Relaxed)
        });
        assert_eq!(
            runs.load(Ordering::Relaxed),
            3,
            "one calibration pass, one timed, one counted"
        );
    }
}
