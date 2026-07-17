// src/editor/health.rs
//
// The Health panel's model: what the engine knows about its own resource use,
// sampled off the world and shaped into meters. The layout half is
// `health_panel.rs`; nothing here draws.
//
// Every meter reads the same way -- what our accounting explains, inside what
// the process actually uses, inside what the machine has:
//
//   RAM   Heap (tracked allocator) < process RSS < physical RAM
//   VRAM  streaming pools          < device allocated < GPU budget
//   CPU   engine systems           < process CPU      < core count
//
// The gap between the first two is the point of the panel: it is everything our
// accounting does not explain. Read it per row, because each row's gap means
// something different. For RAM it is allocations Rust never made (GPU driver,
// mapped assets, stacks, the binary). For VRAM it is device memory outside the
// streaming pools (render targets, shader resources, the swapchain).
//
// CPU is the one row whose nesting is approximate. System timings are wall-clock
// spans around each system's step on the main thread, not CPU time across every
// thread, so a system parked on a GPU fence bills wall time it never spent
// computing. Tracked can exceed used in a GPU-bound frame; that reads as "the
// main thread is blocked, not busy", which is worth seeing rather than hiding.
//
// Sampling is throttled: RSS and process CPU are syscalls, and the pool walk
// sums every resident item. The per-frame system timings are the one thing
// accumulated every tick, because a rate needs the whole window.

use crate::app::syscpu::CpuSampler;
use crate::ecs::World;
use concinnity_memory::MemStats;
use std::time::{Duration, Instant};

// Matches the StatHud's chip cadence: often enough to feel live, rare enough
// that the syscalls and the pool walk do not show up in a frame.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(500);

// What a meter measures, and so how its numbers read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unit {
    Bytes,
    Cores,
}

// One resource's meter: three nested quantities sharing a scale. `None` is
// "unavailable" and renders as a dash, never as a zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Meter {
    // The resource, e.g. "RAM".
    pub caption: &'static str,
    // Names the three numbers in order, e.g. "Heap / Process / Physical".
    pub parts: &'static str,
    // What the engine's own accounting explains.
    pub tracked: Option<f64>,
    // What the whole process uses.
    pub used: Option<f64>,
    // Capacity: the scale both fills are drawn against.
    pub total: Option<f64>,
    pub unit: Unit,
}

impl Meter {
    pub(crate) fn tracked_frac(&self) -> f32 {
        frac(self.tracked, self.total)
    }

    pub(crate) fn used_frac(&self) -> f32 {
        frac(self.used, self.total)
    }

    // The row's three numbers, each at its own scale. A shared scale would be
    // tidier but unreadable: a 30 MB heap against a 32 GB machine rounds to
    // "0.0". The bar already carries the comparison, so the text can just be
    // legible.
    pub(crate) fn value_text(&self) -> String {
        let render = match self.unit {
            Unit::Bytes => bytes_text,
            Unit::Cores => cores_text,
        };
        let suffix = match self.unit {
            Unit::Bytes => "",
            Unit::Cores => " cores",
        };
        format!(
            "{} / {} / {}{suffix}",
            render(self.tracked),
            render(self.used),
            render(self.total),
        )
    }
}

// How much of the track a quantity fills. Clamped: a tracked value above its
// capacity is real (an over-budget pool) but must not draw past the bar.
fn frac(value: Option<f64>, total: Option<f64>) -> f32 {
    match (value, total) {
        (Some(v), Some(t)) if t > 0.0 => (v / t).clamp(0.0, 1.0) as f32,
        _ => 0.0,
    }
}

fn bytes_text(value: Option<f64>) -> String {
    match value {
        Some(v) => {
            let (div, suffix) = byte_scale(v);
            format!("{:.1} {suffix}", v / div)
        }
        None => "--".to_string(),
    }
}

fn cores_text(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{v:.1}"),
        None => "--".to_string(),
    }
}

// Binary scale, labelled the way the StatHud's chips already label it.
fn byte_scale(largest: f64) -> (f64, &'static str) {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * KB;
    const GB: f64 = 1024.0 * MB;
    if largest >= GB {
        (GB, "GB")
    } else if largest >= MB {
        (MB, "MB")
    } else if largest >= KB {
        (KB, "KB")
    } else {
        (1.0, "B")
    }
}

// The sampled numbers behind the meters, refreshed on the throttled tick and
// held between samples so the panel reads steady rather than flickering.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct HealthSnapshot {
    pub heap: Option<MemStats>,
    pub rss: Option<u64>,
    pub total_ram: Option<u64>,
    pub pool_bytes: Option<u64>,
    pub vram_bytes: Option<u64>,
    pub vram_budget: Option<u64>,
    pub systems_cores: Option<f32>,
    pub process_cores: Option<f32>,
    pub total_cores: Option<usize>,
}

// The three meters, in panel order. Pure, so the whole model is testable
// without a world or a clock.
pub(crate) fn meters(snap: &HealthSnapshot) -> [Meter; 3] {
    [
        Meter {
            caption: "RAM",
            parts: "Heap / Process / Physical",
            tracked: snap.heap.map(|h| h.live_bytes as f64),
            used: snap.rss.map(|v| v as f64),
            total: snap.total_ram.map(|v| v as f64),
            unit: Unit::Bytes,
        },
        Meter {
            caption: "VRAM",
            parts: "Pools / Device / Budget",
            tracked: snap.pool_bytes.map(|v| v as f64),
            used: snap.vram_bytes.map(|v| v as f64),
            total: snap.vram_budget.map(|v| v as f64),
            unit: Unit::Bytes,
        },
        Meter {
            caption: "CPU",
            parts: "Systems / Process / Cores",
            tracked: snap.systems_cores.map(|v| v as f64),
            used: snap.process_cores.map(|v| v as f64),
            total: snap.total_cores.map(|v| v as f64),
            unit: Unit::Cores,
        },
    ]
}

// The heap's detail line, under the meters. `None` when no binary installed the
// tracking allocator, which is the shipped runtime's normal state.
//
// Live allocations rather than the raw alloc / free totals: those only ever
// climb (millions within a minute), so they say nothing at a glance and run off
// the panel. The live count is the one that holds steady unless something leaks.
pub(crate) fn churn_text(snap: &HealthSnapshot) -> Option<String> {
    let heap = snap.heap?;
    let live = heap.alloc_count.saturating_sub(heap.free_count);
    Some(format!(
        "peak {}  |  {live} live allocs",
        bytes_text(Some(heap.peak_bytes as f64)),
    ))
}

// The panel's sampler: owns the throttle, the CPU rate, and the latest snapshot.
#[derive(Debug, Default)]
pub(crate) struct HealthState {
    cpu: CpuSampler,
    // Main-thread system microseconds accumulated since the window opened.
    systems_us: u64,
    window_start: Option<Instant>,
    snapshot: HealthSnapshot,
}

impl HealthState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn snapshot(&self) -> &HealthSnapshot {
        &self.snapshot
    }

    // Called every tick regardless of whether the panel is open, so the window's
    // system-time accumulation stays continuous and opening the panel shows a
    // real rate immediately rather than a first-window artifact.
    pub(crate) fn sample(&mut self, world: &World) {
        self.sample_at(world, Instant::now());
    }

    // The throttle, split from the clock so a test can drive the window.
    fn sample_at(&mut self, world: &World, now: Instant) {
        self.systems_us += world
            .profile()
            .system_timings()
            .iter()
            .map(|&(_, micros)| micros as u64)
            .sum::<u64>();

        let start = *self.window_start.get_or_insert(now);
        let elapsed = now.saturating_duration_since(start);
        if elapsed < SAMPLE_INTERVAL {
            return;
        }

        let window_us = elapsed.as_micros() as f64;
        let systems_cores = (window_us > 0.0).then(|| (self.systems_us as f64 / window_us) as f32);
        self.systems_us = 0;
        self.window_start = Some(now);

        let render = &world.profile().render;
        self.snapshot = HealthSnapshot {
            heap: concinnity_memory::stats(),
            rss: crate::app::sysmem::process_resident_bytes(),
            total_ram: world.memory_budget().and_then(|b| b.total_ram_bytes),
            pool_bytes: pool_bytes(world),
            // A backend that cannot report its allocation reports zero; that is
            // "unknown", not "nothing allocated".
            vram_bytes: (render.vram_bytes > 0).then_some(render.vram_bytes),
            vram_budget: world
                .gpu_profile()
                .map(|p| p.memory_budget_bytes)
                .filter(|&b| b > 0),
            systems_cores,
            // A failed read holds the last known rate rather than blanking the row.
            process_cores: self.cpu.sample().or(self.snapshot.process_cores),
            total_cores: world.thread_budget().map(|b| b.total_cores),
        };
    }
}

// Resident bytes across every streaming pool: what the engine's VRAM accounting
// explains. `None` when nothing is streaming, so the row reads "--" rather than
// claiming the engine tracks zero bytes.
fn pool_bytes(world: &World) -> Option<u64> {
    let stats = world.streaming_stats()?;
    [stats.texture_bytes, stats.mesh_bytes, stats.chunk_bytes]
        .into_iter()
        .flatten()
        .map(|(resident, _budget)| resident)
        .fold(None, |acc: Option<u64>, resident| {
            Some(acc.unwrap_or(0) + resident)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    fn ram_snapshot() -> HealthSnapshot {
        HealthSnapshot {
            heap: Some(MemStats {
                live_bytes: GB / 2,
                peak_bytes: GB,
                alloc_count: 10,
                free_count: 4,
            }),
            rss: Some(2 * GB),
            total_ram: Some(32 * GB),
            ..Default::default()
        }
    }

    #[test]
    fn fractions_scale_each_fill_against_the_capacity() {
        let ram = meters(&ram_snapshot())[0];
        assert!((ram.tracked_frac() - 0.5 / 32.0).abs() < 1e-6);
        assert!((ram.used_frac() - 2.0 / 32.0).abs() < 1e-6);
    }

    // Each number carries its own unit. A shared scale would round the heap away
    // entirely, which is the whole reason this row is formatted per value.
    #[test]
    fn value_text_scales_each_number_independently() {
        assert_eq!(
            meters(&ram_snapshot())[0].value_text(),
            "512.0 MB / 2.0 GB / 32.0 GB"
        );
    }

    // A small quantity beside a huge capacity stays legible instead of rounding
    // to "0.0" -- the failure that a shared scale produced on real data.
    #[test]
    fn a_small_value_beside_a_huge_capacity_stays_legible() {
        let snap = HealthSnapshot {
            heap: Some(MemStats {
                live_bytes: 30 * 1024 * 1024,
                ..Default::default()
            }),
            rss: Some(3 * GB),
            total_ram: Some(32 * GB),
            ..Default::default()
        };
        assert_eq!(
            meters(&snap)[0].value_text(),
            "30.0 MB / 3.0 GB / 32.0 GB",
            "a 30 MB heap must not round away against a 32 GB machine"
        );
    }

    // An absent number is a dash. A zero here would read as a real measurement
    // of nothing, which is exactly the lie the panel must not tell.
    #[test]
    fn missing_values_read_as_dashes_and_draw_no_fill() {
        let vram = meters(&HealthSnapshot::default())[1];
        assert_eq!(vram.value_text(), "-- / -- / --");
        assert_eq!(vram.tracked_frac(), 0.0);
        assert_eq!(vram.used_frac(), 0.0);
    }

    // A known capacity with unknown usage still scales, so the row shows what it
    // does know.
    #[test]
    fn a_known_capacity_scales_even_with_unknown_usage() {
        let snap = HealthSnapshot {
            vram_budget: Some(8 * GB),
            ..Default::default()
        };
        assert_eq!(meters(&snap)[1].value_text(), "-- / -- / 8.0 GB");
    }

    // Over-capacity is real (a pool past its byte budget) but must not draw off
    // the end of the track.
    #[test]
    fn an_over_capacity_fill_clamps_to_the_track() {
        let snap = HealthSnapshot {
            pool_bytes: Some(12 * GB),
            vram_budget: Some(8 * GB),
            ..Default::default()
        };
        assert_eq!(meters(&snap)[1].tracked_frac(), 1.0);
    }

    // A zero capacity cannot divide; the fill is empty rather than NaN.
    #[test]
    fn a_zero_capacity_yields_an_empty_fill() {
        let snap = HealthSnapshot {
            rss: Some(GB),
            total_ram: Some(0),
            ..Default::default()
        };
        let ram = meters(&snap)[0];
        assert_eq!(ram.used_frac(), 0.0);
        assert!(ram.used_frac().is_finite());
    }

    // Cores are not bytes: the row reports them as a count against the machine's
    // cores, with no byte scaling applied.
    #[test]
    fn cpu_reads_in_cores_against_the_core_count() {
        let snap = HealthSnapshot {
            systems_cores: Some(0.4),
            process_cores: Some(3.2),
            total_cores: Some(10),
            ..Default::default()
        };
        let cpu = meters(&snap)[2];
        assert_eq!(cpu.unit, Unit::Cores);
        assert_eq!(cpu.value_text(), "0.4 / 3.2 / 10.0 cores");
        assert!((cpu.used_frac() - 0.32).abs() < 1e-6);
    }

    // The CPU row's nesting is approximate by construction: a GPU-blocked main
    // thread bills wall time it never computed. The meter reports that rather
    // than clamping tracked to used and hiding it.
    #[test]
    fn cpu_tracked_may_exceed_used_when_the_main_thread_blocks() {
        let snap = HealthSnapshot {
            systems_cores: Some(0.9),
            process_cores: Some(0.5),
            total_cores: Some(8),
            ..Default::default()
        };
        let cpu = meters(&snap)[2];
        assert!(cpu.tracked_frac() > cpu.used_frac());
    }

    #[test]
    fn byte_scale_picks_the_unit_from_the_largest_value() {
        assert_eq!(byte_scale(512.0).1, "B");
        assert_eq!(byte_scale(2048.0).1, "KB");
        assert_eq!(byte_scale(5.0 * 1024.0 * 1024.0).1, "MB");
        assert_eq!(byte_scale(3.0 * GB as f64).1, "GB");
    }

    // Live allocations, not the ever-climbing cumulative totals: 10 made minus 4
    // freed is 6 outstanding.
    #[test]
    fn churn_text_reports_peak_and_live_allocations() {
        let text = churn_text(&ram_snapshot()).expect("the snapshot carries heap stats");
        assert_eq!(text, "peak 1.0 GB  |  6 live allocs");
    }

    // No tracking allocator installed means no heap line at all, rather than a
    // row of zeroes.
    #[test]
    fn churn_text_is_absent_without_the_tracking_allocator() {
        assert_eq!(churn_text(&HealthSnapshot::default()), None);
    }
}
