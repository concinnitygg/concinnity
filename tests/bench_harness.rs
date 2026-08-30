//! Tests over the bench suite's shared harness. The suite itself is a
//! `harness = false` bench target, which never executes `#[test]` functions,
//! so the module is compiled here under the test harness instead.

#[path = "../benches/suite/support.rs"]
mod support;

use support::{Bench, MIN_SAMPLES, Record, Rng, TIME_BUDGET_NS};
use support::{fmt_bytes, fmt_count, fmt_time, json_report, percentile, samples_for};

fn for_test() -> Bench {
    for_test_with_target(10_000)
}

fn for_test_with_target(target_sample_ns: u128) -> Bench {
    Bench {
        filters: Vec::new(),
        json_path: None,
        samples: 3,
        target_sample_ns,
        records: Vec::new(),
    }
}

#[test]
fn rng_shuffle_is_a_deterministic_permutation() {
    let mut a: Vec<u32> = (0..64).collect();
    let mut b: Vec<u32> = (0..64).collect();
    Rng::new(7).shuffle(&mut a);
    Rng::new(7).shuffle(&mut b);
    assert_eq!(a, b, "same seed must produce the same order");
    assert_ne!(
        a,
        (0..64).collect::<Vec<u32>>(),
        "shuffle must move something"
    );
    let mut sorted = a.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        (0..64).collect::<Vec<u32>>(),
        "shuffle must be a permutation"
    );
}

#[test]
fn zero_seed_still_generates() {
    let mut rng = Rng::new(0);
    assert_ne!(rng.next_u64(), rng.next_u64());
}

#[test]
fn sample_count_scales_with_body_cost() {
    // A cheap body gets the full sample count.
    assert_eq!(samples_for(2_000_000, 25), 25);
    // A one-second body sheds samples down to the budget.
    assert_eq!(samples_for(1_000_000_000, 25), 3.max(MIN_SAMPLES));
    // Never below the floor, even for a body that eats the whole budget.
    assert_eq!(samples_for(10 * TIME_BUDGET_NS, 25), MIN_SAMPLES);
    assert_eq!(samples_for(0, 25), 25);
    // A max below the floor wins (the test harness runs with tiny counts).
    assert_eq!(samples_for(2_000_000, 3), 3);
    assert_eq!(samples_for(10 * TIME_BUDGET_NS, 3), 3);
}

#[test]
fn percentile_interpolates_between_ranks() {
    let sorted = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(percentile(&sorted, 0.0), 1.0);
    assert_eq!(percentile(&sorted, 50.0), 2.5);
    assert_eq!(percentile(&sorted, 100.0), 4.0);
    assert!((percentile(&sorted, 95.0) - 3.85).abs() < 1e-9);
    assert_eq!(percentile(&[], 50.0), 0.0);
    assert_eq!(percentile(&[7.0], 95.0), 7.0);
}

#[test]
fn tiny_counts_are_not_shown_as_zero() {
    assert_eq!(fmt_count(0.0), "0.000");
    assert_eq!(fmt_count(1.0), "1.000");
    assert_eq!(fmt_count(0.000_244), "<0.001");
    assert_eq!(fmt_count(-0.000_244), "<0.001");
}

#[test]
fn time_and_byte_formats_scale_units() {
    assert_eq!(fmt_time(1.6), "1.6 ns");
    assert_eq!(fmt_time(1_500.0), "1.50 us");
    assert_eq!(fmt_time(2_500_000.0), "2.50 ms");
    assert_eq!(fmt_bytes(0.0), "0 B");
    assert_eq!(fmt_bytes(64.0), "+64 B");
    assert_eq!(fmt_bytes(-2048.0), "-2.0 KiB");
    assert_eq!(fmt_bytes(3.0 * 1024.0 * 1024.0), "+3.0 MiB");
}

#[test]
fn args_split_filters_from_flags() {
    let bench = Bench::from_args(
        ["--bench", "join", "--json", "out.json", "spawn"]
            .into_iter()
            .map(String::from),
    );
    assert_eq!(bench.filters, ["join", "spawn"]);
    assert_eq!(bench.json_path.as_deref(), Some("out.json"));
    assert!(bench.matches("ecs/join2/10k"));
    assert!(bench.matches("ecs/spawn_prop/10k"));
    assert!(!bench.matches("ecs/scan_transforms/10k"));
    assert!(Bench::from_args(std::iter::empty()).matches("anything"));
}

#[test]
fn from_env_reads_the_process_arguments() {
    // The test binary's own arguments; the splitting rules are covered by
    // `args_split_filters_from_flags`.
    let _ = Bench::from_env();
}

// Runs a real measurement against a body with a known allocation profile.
// The counters are process-global and tests run in parallel, so only lower
// bounds hold. Timings are checked for shape only: a body this cheap can
// read as zero at the platform clock's resolution.
#[test]
fn a_measured_body_reports_its_allocations() {
    let mut bench = for_test();
    bench.run("test/alloc_one_vec", 4, || {
        core::hint::black_box(Vec::<u8>::with_capacity(256));
    });
    let record = &bench.records[0];
    assert_eq!(record.items, 4);
    assert!(record.iters >= 1);
    assert!(record.median_ns.is_finite() && record.median_ns >= 0.0);
    assert!(record.p95_ns >= record.median_ns);
    assert!(
        record.allocs_per_iter >= 1.0,
        "one Vec per call must show at least one allocation, got {}",
        record.allocs_per_iter
    );
}

// The clock-resolution case the timings above must tolerate. A zero sample
// target ends calibration at one iteration, and a body cheaper than one
// clock tick then measures as zero.
#[test]
fn a_body_cheaper_than_a_clock_tick_still_reports() {
    let mut bench = for_test_with_target(0);
    bench.run("test/one_iteration", 1, || {
        core::hint::black_box(Vec::<u8>::with_capacity(256));
    });
    let record = &bench.records[0];
    assert_eq!(record.iters, 1);
    assert!(record.median_ns.is_finite() && record.median_ns >= 0.0);
    assert!(record.p95_ns >= record.median_ns);
    assert!(record.allocs_per_iter >= 1.0);
}

#[test]
fn filtered_out_benchmarks_do_not_run() {
    let mut bench = for_test();
    bench.filters.push("nothing-matches-this".to_string());
    let mut calls = 0u32;
    bench.run("test/skipped", 1, || calls += 1);
    assert_eq!(calls, 0);
    assert!(bench.records.is_empty());
}

#[test]
fn finish_writes_the_requested_json_report() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("report.json");
    let mut bench = for_test();
    bench.json_path = Some(path.display().to_string());
    bench.run("test/report_one", 1, || core::hint::black_box(1u32));
    bench.finish();
    let json = std::fs::read_to_string(&path).expect("finish writes the report");
    assert!(json.contains("\"name\":\"test/report_one\""));
}

#[test]
fn json_report_holds_one_object_per_record() {
    let records = vec![
        Record {
            name: "a/b".to_string(),
            items: 2,
            iters: 8,
            median_ns: 1.5,
            p95_ns: 2.0,
            allocs_per_iter: 1.0,
            frees_per_iter: 1.0,
            heap_bytes_per_iter: 0.0,
            device_bytes_per_iter: 0.0,
        },
        Record {
            name: "c\"d".to_string(),
            items: 1,
            iters: 1,
            median_ns: 3.0,
            p95_ns: 4.0,
            allocs_per_iter: 0.0,
            frees_per_iter: 0.0,
            heap_bytes_per_iter: -8.0,
            device_bytes_per_iter: 16.0,
        },
    ];
    let json = json_report(&records);
    assert!(json.starts_with("[\n"));
    assert!(json.ends_with("]\n"));
    assert!(json.contains("\"name\":\"a/b\""));
    assert!(json.contains("\"median_ns\":1.5"));
    assert!(
        json.contains("\"name\":\"c\\\"d\""),
        "quotes must be escaped"
    );
    assert!(json.contains("\"device_bytes_per_iter\":16}"));
    assert_eq!(json.matches("{\"name\"").count(), 2);
}
