// concinnity-bench/src/lib.rs
//
// The harness behind the workspace's `cargo bench` targets. It times a body
// over calibrated iteration counts, then re-runs it between snapshots of the
// engine's own memory instruments: the tracked heap for allocation counts and
// live-byte deltas, and the tagged ledger for host/device bytes. No external
// bench framework, so every number comes from accounting the engine ships with.
//
// A benchmark body is the measured unit; `items` says how many units of work
// one call performs (entities scanned, lookups made) so the report can state
// per-item costs. Bodies must be deterministic and self-contained: whatever a
// body builds it should also tear down, or the heap column will show the drift.

use std::time::Instant;

use concinnity_memory::Realm;

// One declaration for every bench target in this package: the per-iteration
// allocation counts are only real because the harness runs on the tracking
// allocator.
concinnity_memory::install_global_allocator!();

const MAX_ITERS: u64 = 1 << 22;
const DEFAULT_SAMPLES: usize = 25;
const MIN_SAMPLES: usize = 5;
const DEFAULT_TARGET_SAMPLE_NS: u128 = 2_000_000;
// Cap on one benchmark's timing passes, so a body costing seconds per call
// (a whole-world cook) sheds samples instead of stalling the run.
const TIME_BUDGET_NS: u128 = 3_000_000_000;

/// Deterministic xorshift RNG so bench fixtures lay out identically across
/// runs and machines.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // Xorshift has a zero fixed point; substitute a fixed odd constant.
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

/// One finished measurement: per-iteration timing percentiles plus the heap
/// and ledger traffic one iteration causes.
pub struct Record {
    pub name: String,
    pub items: u64,
    pub iters: u64,
    pub median_ns: f64,
    pub p95_ns: f64,
    pub allocs_per_iter: f64,
    pub frees_per_iter: f64,
    pub heap_bytes_per_iter: f64,
    pub device_bytes_per_iter: f64,
}

/// Collects and reports benchmark measurements for one bench target.
pub struct Bench {
    filters: Vec<String>,
    json_path: Option<String>,
    samples: usize,
    target_sample_ns: u128,
    records: Vec<Record>,
}

impl Bench {
    /// Build a harness from the CLI arguments `cargo bench` forwards: bare
    /// words select benchmarks by substring, `--json <path>` writes the
    /// records as JSON, and other flags (cargo's own `--bench`) are ignored.
    pub fn from_env() -> Bench {
        Bench::from_args(std::env::args().skip(1))
    }

    fn from_args(args: impl Iterator<Item = String>) -> Bench {
        let mut filters = Vec::new();
        let mut json_path = None;
        let mut args = args;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json" => json_path = args.next(),
                flag if flag.starts_with('-') => {}
                word => filters.push(word.to_string()),
            }
        }
        Bench {
            filters,
            json_path,
            samples: DEFAULT_SAMPLES,
            target_sample_ns: DEFAULT_TARGET_SAMPLE_NS,
            records: Vec::new(),
        }
    }

    #[cfg(test)]
    fn for_test() -> Bench {
        Bench {
            filters: Vec::new(),
            json_path: None,
            samples: 3,
            target_sample_ns: 10_000,
            records: Vec::new(),
        }
    }

    fn matches(&self, name: &str) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|f| name.contains(f.as_str()))
    }

    /// Measure `body`, attributing each iteration's cost across `items` units
    /// of work. `items` only scales the report; the body always runs whole.
    pub fn run<R>(&mut self, name: &str, items: u64, mut body: impl FnMut() -> R) {
        if !self.matches(name) {
            return;
        }
        let (iters, sample_ns) = self.calibrate(&mut body);
        let samples_to_take = samples_for(sample_ns, self.samples);

        let mut samples = Vec::with_capacity(samples_to_take);
        for _ in 0..samples_to_take {
            let start = Instant::now();
            for _ in 0..iters {
                core::hint::black_box(body());
            }
            samples.push(start.elapsed().as_nanos() as f64 / iters as f64);
        }
        samples.sort_by(|a, b| a.total_cmp(b));

        // A separate pass for memory, so snapshot reads sit outside the timed
        // windows and the timing loop sits outside the counted window.
        let ledger = concinnity_memory::ledger();
        let device_before = ledger.snapshot().realm_bytes(Realm::Device) as i64;
        let heap_before = concinnity_memory::stats().unwrap_or_default();
        for _ in 0..iters {
            core::hint::black_box(body());
        }
        let heap_after = concinnity_memory::stats().unwrap_or_default();
        let device_after = ledger.snapshot().realm_bytes(Realm::Device) as i64;

        let per_iter =
            |after: u64, before: u64| (after as i64 - before as i64) as f64 / iters as f64;
        let record = Record {
            name: name.to_string(),
            items: items.max(1),
            iters,
            median_ns: percentile(&samples, 50.0),
            p95_ns: percentile(&samples, 95.0),
            allocs_per_iter: per_iter(heap_after.alloc_count, heap_before.alloc_count),
            frees_per_iter: per_iter(heap_after.free_count, heap_before.free_count),
            heap_bytes_per_iter: per_iter(heap_after.live_bytes, heap_before.live_bytes),
            device_bytes_per_iter: (device_after - device_before) as f64 / iters as f64,
        };
        if self.records.is_empty() {
            print_header();
        }
        print_record(&record);
        self.records.push(record);
    }

    // Grow the per-sample iteration count until one sample meets the time
    // target, warming the body up along the way. Returns the count and the
    // final round's duration, the cost estimate the sample budget divides.
    fn calibrate<R>(&self, body: &mut impl FnMut() -> R) -> (u64, u128) {
        let mut iters: u64 = 1;
        loop {
            let start = Instant::now();
            for _ in 0..iters {
                core::hint::black_box(body());
            }
            let elapsed = start.elapsed().as_nanos();
            if elapsed >= self.target_sample_ns || iters >= MAX_ITERS {
                return (iters, elapsed);
            }
            let scale = self.target_sample_ns.div_ceil(elapsed.max(1)).max(2) as u64;
            iters = iters.saturating_mul(scale).min(MAX_ITERS);
        }
    }

    /// Finish the run: write the JSON report when one was requested.
    pub fn finish(self) {
        if self.records.is_empty() {
            println!("no benchmarks matched the filter");
            return;
        }
        if let Some(path) = &self.json_path {
            std::fs::write(path, json_report(&self.records))
                .unwrap_or_else(|e| panic!("writing {path}: {e}"));
            println!("\nwrote {path}");
        }
    }
}

// How many timing samples a benchmark gets: as many as fit the time budget,
// bounded to [MIN_SAMPLES, max_samples].
fn samples_for(sample_ns: u128, max_samples: usize) -> usize {
    let fit = (TIME_BUDGET_NS / sample_ns.max(1)) as usize;
    fit.clamp(MIN_SAMPLES.min(max_samples), max_samples)
}

// Linear-interpolation percentile over an ascending-sorted sample set.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    match sorted {
        [] => 0.0,
        [only] => *only,
        _ => {
            let rank = p / 100.0 * (sorted.len() - 1) as f64;
            let lo = rank.floor() as usize;
            let hi = rank.ceil() as usize;
            sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
        }
    }
}

fn fmt_time(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} us", ns / 1_000.0)
    } else {
        format!("{:.2} ms", ns / 1_000_000.0)
    }
}

fn fmt_bytes(bytes: f64) -> String {
    const KIB: f64 = 1024.0;
    if bytes == 0.0 {
        return "0 B".to_string();
    }
    let sign = if bytes < 0.0 { "-" } else { "+" };
    let magnitude = bytes.abs();
    if magnitude < KIB {
        format!("{sign}{magnitude:.0} B")
    } else if magnitude < KIB * KIB {
        format!("{sign}{:.1} KiB", magnitude / KIB)
    } else {
        format!("{sign}{:.1} MiB", magnitude / (KIB * KIB))
    }
}

fn print_header() {
    println!(
        "{:<40} {:>12} {:>12} {:>13} {:>12} {:>12}",
        "benchmark", "time/item", "p95/item", "allocs/item", "heap/item", "vram/item"
    );
}

fn print_record(r: &Record) {
    let items = r.items as f64;
    println!(
        "{:<40} {:>12} {:>12} {:>13} {:>12} {:>12}",
        r.name,
        fmt_time(r.median_ns / items),
        fmt_time(r.p95_ns / items),
        fmt_count(r.allocs_per_iter / items),
        fmt_bytes(r.heap_bytes_per_iter / items),
        fmt_bytes(r.device_bytes_per_iter / items),
    );
}

// Rounding a small-but-real count to 0.000 would read as "allocation-free",
// which is exactly the claim this column exists to check.
fn fmt_count(v: f64) -> String {
    if v != 0.0 && v.abs() < 0.001 {
        "<0.001".to_string()
    } else {
        format!("{v:.3}")
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_report(records: &[Record]) -> String {
    let mut out = String::from("[\n");
    for (i, r) in records.iter().enumerate() {
        let comma = if i + 1 < records.len() { "," } else { "" };
        out.push_str(&format!(
            "  {{\"name\":\"{}\",\"items\":{},\"iters\":{},\"median_ns\":{},\"p95_ns\":{},\
             \"allocs_per_iter\":{},\"frees_per_iter\":{},\"heap_bytes_per_iter\":{},\
             \"device_bytes_per_iter\":{}}}{comma}\n",
            json_escape(&r.name),
            r.items,
            r.iters,
            r.median_ns,
            r.p95_ns,
            r.allocs_per_iter,
            r.frees_per_iter,
            r.heap_bytes_per_iter,
            r.device_bytes_per_iter,
        ));
    }
    out.push_str("]\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Runs a real measurement against a body with a known allocation profile.
    // The counters are process-global and tests run in parallel, so only lower
    // bounds hold.
    #[test]
    fn a_measured_body_reports_its_allocations() {
        let mut bench = Bench::for_test();
        bench.run("test/alloc_one_vec", 4, || {
            core::hint::black_box(Vec::<u8>::with_capacity(256));
        });
        let record = &bench.records[0];
        assert_eq!(record.items, 4);
        assert!(record.iters >= 1);
        assert!(record.median_ns > 0.0);
        assert!(record.p95_ns >= record.median_ns);
        assert!(
            record.allocs_per_iter >= 1.0,
            "one Vec per call must show at least one allocation, got {}",
            record.allocs_per_iter
        );
    }

    #[test]
    fn filtered_out_benchmarks_do_not_run() {
        let mut bench = Bench::for_test();
        bench.filters.push("nothing-matches-this".to_string());
        let mut calls = 0u32;
        bench.run("test/skipped", 1, || calls += 1);
        assert_eq!(calls, 0);
        assert!(bench.records.is_empty());
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
}
