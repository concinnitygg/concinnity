//! The harness behind the bench suite. It times a body
//! over calibrated iteration counts, then re-runs it between snapshots of the
//! engine's own memory instruments: the tracked heap for allocation counts and
//! live-byte deltas, and the tagged ledger for host/device bytes. No external
//! bench framework, so every number comes from accounting the engine ships with.
//!
//! A benchmark body is the measured unit; `items` says how many units of work
//! one call performs (entities scanned, lookups made) so the report can state
//! per-item costs. Bodies must be deterministic and self-contained: whatever a
//! body builds it should also tear down, or the heap column will show the drift.

use std::time::Instant;

use concinnity_core::memory::Realm;

// One declaration for the whole suite: the per-iteration
// allocation counts are only real because the harness runs on the tracking
// allocator.
concinnity_core::install_global_allocator!();

const MAX_ITERS: u64 = 1 << 22;
const DEFAULT_SAMPLES: usize = 25;
pub(crate) const MIN_SAMPLES: usize = 5;
const DEFAULT_TARGET_SAMPLE_NS: u128 = 2_000_000;
// Cap on one benchmark's timing passes, so a body costing seconds per call
// (a whole-world cook) sheds samples instead of stalling the run.
pub(crate) const TIME_BUDGET_NS: u128 = 3_000_000_000;

/// Deterministic xorshift RNG so bench fixtures lay out identically across
/// runs and machines.
pub(crate) struct Rng(u64);

impl Rng {
    /// A generator seeded with `seed`; zero is replaced by a fixed constant.
    pub(crate) fn new(seed: u64) -> Rng {
        // Xorshift has a zero fixed point; substitute a fixed odd constant.
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// The next 64-bit value.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value in `0..n`.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Shuffle `items` in place.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

/// One finished measurement: per-iteration timing percentiles plus the heap
/// and ledger traffic one iteration causes.
pub(crate) struct Record {
    /// The benchmark's name.
    pub(crate) name: String,
    /// Items processed per iteration.
    pub(crate) items: u64,
    /// Iterations timed.
    pub(crate) iters: u64,
    /// Median per-iteration time in nanoseconds.
    pub(crate) median_ns: f64,
    /// 95th-percentile per-iteration time in nanoseconds.
    pub(crate) p95_ns: f64,
    /// Allocations per iteration.
    pub(crate) allocs_per_iter: f64,
    /// Frees per iteration.
    pub(crate) frees_per_iter: f64,
    /// Heap bytes allocated per iteration.
    pub(crate) heap_bytes_per_iter: f64,
    /// Device bytes reported per iteration.
    pub(crate) device_bytes_per_iter: f64,
}

/// Collects and reports benchmark measurements for one bench target.
pub(crate) struct Bench {
    pub(crate) filters: Vec<String>,
    pub(crate) json_path: Option<String>,
    pub(crate) samples: usize,
    pub(crate) target_sample_ns: u128,
    pub(crate) records: Vec<Record>,
}

impl Bench {
    /// Build a harness from the CLI arguments `cargo bench` forwards: bare
    /// words select benchmarks by substring, `--json <path>` writes the
    /// records as JSON, and other flags (cargo's own `--bench`) are ignored.
    pub(crate) fn from_env() -> Bench {
        Bench::from_args(std::env::args().skip(1))
    }

    pub(crate) fn from_args(args: impl Iterator<Item = String>) -> Bench {
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

    pub(crate) fn matches(&self, name: &str) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|f| name.contains(f.as_str()))
    }

    /// Measure `body`, attributing each iteration's cost across `items` units
    /// of work. `items` only scales the report; the body always runs whole.
    pub(crate) fn run<R>(&mut self, name: &str, items: u64, mut body: impl FnMut() -> R) {
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
        let ledger = concinnity_core::memory::ledger();
        let device_before = ledger.snapshot().realm_bytes(Realm::Device) as i64;
        let heap_before = concinnity_core::memory::stats().unwrap_or_default();
        for _ in 0..iters {
            core::hint::black_box(body());
        }
        let heap_after = concinnity_core::memory::stats().unwrap_or_default();
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
    pub(crate) fn finish(self) {
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
pub(crate) fn samples_for(sample_ns: u128, max_samples: usize) -> usize {
    let fit = (TIME_BUDGET_NS / sample_ns.max(1)) as usize;
    fit.clamp(MIN_SAMPLES.min(max_samples), max_samples)
}

// Linear-interpolation percentile over an ascending-sorted sample set.
pub(crate) fn percentile(sorted: &[f64], p: f64) -> f64 {
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

pub(crate) fn fmt_time(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} us", ns / 1_000.0)
    } else {
        format!("{:.2} ms", ns / 1_000_000.0)
    }
}

pub(crate) fn fmt_bytes(bytes: f64) -> String {
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
pub(crate) fn fmt_count(v: f64) -> String {
    if v != 0.0 && v.abs() < 0.001 {
        "<0.001".to_string()
    } else {
        format!("{v:.3}")
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

pub(crate) fn json_report(records: &[Record]) -> String {
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
