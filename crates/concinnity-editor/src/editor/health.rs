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
// Under the meters, the tag breakdown names the part of each realm that *is*
// explained: what every subsystem reports holding, host rows then device ones.
// It is a floor on real usage, never a total, since it holds only what someone
// reported.
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

use crate::app::mem_drift::MemoryDrift;
use crate::app::syscpu::CpuSampler;
use crate::ecs::World;
use concinnity_memory::{LedgerSnapshot, MemStats, MemTag, Realm, SizeClass};
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
    // What the engine's subsystems report holding, by tag and realm. The named
    // part of the two byte meters above.
    pub tags: LedgerSnapshot,
    // The allocation size class holding the most live blocks. `None` unless the
    // binary was built with the allocator's `detail` feature.
    pub hot_class: Option<SizeClass>,
    // Where the process's memory has moved since the session settled. `None`
    // until a baseline exists, which is most of a short editor session.
    pub drift: Option<MemoryDrift>,
}

// The most breakdown rows there can be: every tag, in both realms.
pub(crate) const MAX_TAG_ROWS: usize = MemTag::COUNT * Realm::COUNT;

// One line of the tag breakdown: who is holding memory, where, and how much of
// its budget that is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TagRow {
    pub name: &'static str,
    pub realm: &'static str,
    // Bytes held, followed by the budget when the tag has one.
    pub value: String,
    pub over_budget: bool,
}

// The breakdown under the meters: every tag something reports into, host rows
// first. Unreported tags are absent rather than listed at zero -- the ledger
// holds only what a subsystem published, so a missing row means "nobody
// accounts for this", not "this is empty".
pub(crate) fn tag_rows(snap: &HealthSnapshot) -> Vec<TagRow> {
    Realm::ALL
        .into_iter()
        .flat_map(|realm| snap.tags.reported(realm))
        .map(|usage| TagRow {
            name: usage.tag.name(),
            realm: usage.realm.name(),
            value: match usage.budget {
                Some(budget) => format!(
                    "{} / {}",
                    bytes_text(Some(usage.bytes as f64)),
                    bytes_text(Some(budget as f64))
                ),
                None => bytes_text(Some(usage.bytes as f64)),
            },
            over_budget: usage.over_budget(),
        })
        .collect()
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

// The drift line: how far the process's memory has moved since the session
// settled, split into the part the tracked heap explains and the part it does
// not. `None` until a baseline exists.
//
// The split is the whole line. The meters above show where memory stands, and
// standing still says nothing about which way it is heading; a heap that grew
// while the rest held steady is ours to fix, and a resident set that grew while
// the heap held steady is not. Whole units, because a drift figure is a
// magnitude and a tenth of a megabyte of it is noise.
pub(crate) fn drift_text(snap: &HealthSnapshot) -> Option<String> {
    let drift = snap.drift?;
    Some(format!(
        "{}  heap {}  outside {}",
        window_text(drift.window_secs),
        whole_signed_bytes(drift.heap_growth_bytes),
        whole_signed_bytes(drift.outside_heap_growth_bytes),
    ))
}

// Signed bytes at whole-unit precision, e.g. "+412 MB" / "-3 GB".
fn whole_signed_bytes(value: i64) -> String {
    let (div, suffix) = byte_scale(value.unsigned_abs() as f64);
    let sign = if value < 0 { '-' } else { '+' };
    format!("{sign}{:.0} {suffix}", value.unsigned_abs() as f64 / div)
}

// The drift window at the coarsest useful precision: minutes, then hours, then
// days. A drift figure without its window is unreadable -- 400 MB over three
// minutes and over three hours are different problems -- and a window to the
// second is more than the figure beside it deserves. Days matter because a
// long-running server is exactly what this line is for, and a session measured
// in weeks reads as "41d" rather than "1000h".
fn window_text(secs: u64) -> String {
    let minutes = secs / 60;
    let hours = minutes / 60;
    if hours >= 48 {
        format!("{}d", hours / 24)
    } else if minutes >= 60 {
        format!("{hours}h")
    } else {
        format!("{minutes}m")
    }
}

// Where the heap's live blocks concentrate, under the meters' churn line.
// `None` in a build without the allocator's `detail` feature, which is every
// shipped one.
pub(crate) fn hot_class_text(snap: &HealthSnapshot) -> Option<String> {
    let class = snap.hot_class?;
    Some(format!(
        "hot class {}  |  {} live blocks",
        bytes_text(Some(class.min_bytes as f64)),
        class.live_blocks,
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
            total_ram: concinnity_engine::ecs::memory_budget(world).and_then(|b| b.total_ram_bytes),
            pool_bytes: pool_bytes(world),
            // A backend that cannot report its allocation reports zero; that is
            // "unknown", not "nothing allocated".
            vram_bytes: (render.vram_bytes > 0).then_some(render.vram_bytes),
            vram_budget: concinnity_engine::ecs::gpu_profile(world)
                .map(|p| p.memory_budget_bytes)
                .filter(|&b| b > 0),
            systems_cores,
            // A failed read holds the last known rate rather than blanking the row.
            process_cores: self.cpu.sample().or(self.snapshot.process_cores),
            total_cores: concinnity_engine::ecs::thread_budget(world).map(|b| b.total_cores),
            tags: concinnity_memory::ledger().snapshot(),
            hot_class: concinnity_memory::size_classes().and_then(|c| c.busiest()),
            drift: concinnity_engine::ecs::memory_drift(world),
        };
    }
}

// Resident bytes across every streaming pool: what the engine's VRAM accounting
// explains. `None` when nothing is streaming, so the row reads "--" rather than
// claiming the engine tracks zero bytes.
fn pool_bytes(world: &World) -> Option<u64> {
    let stats = concinnity_engine::ecs::streaming_stats(world)?;
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

    // A ledger nothing has reported into, which is a world that streams
    // nothing: the breakdown is empty rather than eighteen zeroes.
    #[test]
    fn an_unreported_ledger_lists_no_rows() {
        assert!(tag_rows(&HealthSnapshot::default()).is_empty());
    }

    fn tagged_snapshot() -> HealthSnapshot {
        let ledger = concinnity_memory::Ledger::new();
        ledger.set(MemTag::Textures, Realm::Device, 3 * GB / 2);
        ledger.set_budget(MemTag::Textures, Realm::Device, Some(2 * GB));
        ledger.set(MemTag::Meshes, Realm::Device, GB / 4);
        ledger.set(MemTag::Scratch, Realm::Host, 64 * 1024);
        HealthSnapshot {
            tags: ledger.snapshot(),
            ..Default::default()
        }
    }

    // Host rows first, then device: RAM and VRAM read as separate accounts
    // under one vocabulary.
    #[test]
    fn the_breakdown_lists_host_rows_before_device_rows() {
        let rows = tag_rows(&tagged_snapshot());
        let names: Vec<&str> = rows.iter().map(|r| r.name).collect();
        let realms: Vec<&str> = rows.iter().map(|r| r.realm).collect();
        assert_eq!(names, ["Scratch", "Textures", "Meshes"]);
        assert_eq!(realms, ["RAM", "VRAM", "VRAM"]);
    }

    // A budgeted tag shows what it holds against its ceiling; an unbudgeted one
    // shows the bytes alone rather than inventing a scale.
    #[test]
    fn a_budgeted_row_reads_against_its_ceiling() {
        let rows = tag_rows(&tagged_snapshot());
        assert_eq!(rows[1].value, "1.5 GB / 2.0 GB");
        assert!(!rows[1].over_budget);
        assert_eq!(rows[2].value, "256.0 MB");
    }

    #[test]
    fn a_row_past_its_budget_is_flagged() {
        let ledger = concinnity_memory::Ledger::new();
        ledger.set(MemTag::Chunks, Realm::Device, 3 * GB);
        ledger.set_budget(MemTag::Chunks, Realm::Device, Some(GB));
        let snap = HealthSnapshot {
            tags: ledger.snapshot(),
            ..Default::default()
        };

        let rows = tag_rows(&snap);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].over_budget);
    }

    // Every possible row has somewhere to be drawn, which is what keeps the
    // panel from silently dropping a reporter.
    #[test]
    fn the_breakdown_never_outgrows_the_panels_row_budget() {
        let ledger = concinnity_memory::Ledger::new();
        for tag in MemTag::ALL {
            for realm in Realm::ALL {
                ledger.set(tag, realm, 1);
            }
        }
        let snap = HealthSnapshot {
            tags: ledger.snapshot(),
            ..Default::default()
        };
        assert_eq!(tag_rows(&snap).len(), MAX_TAG_ROWS);
    }

    // The reading the line exists for: the heap held steady while the resident
    // set climbed, so the growth is not a leak and hunting one would waste the
    // session.
    #[test]
    fn the_drift_line_separates_heap_growth_from_growth_outside_it() {
        let snap = HealthSnapshot {
            drift: Some(MemoryDrift {
                heap_growth_bytes: 2 * 1024 * 1024,
                outside_heap_growth_bytes: 412 * 1024 * 1024,
                window_secs: 2 * 3600,
                verdict: crate::app::mem_drift::DriftVerdict::OutsideHeap,
            }),
            ..Default::default()
        };
        assert_eq!(
            drift_text(&snap).expect("a baseline was captured"),
            "2h  heap +2 MB  outside +412 MB"
        );
    }

    // Memory handed back reads as negative rather than as growth, which is the
    // difference between a session recovering and one in trouble.
    #[test]
    fn the_drift_line_signs_a_shrinking_term() {
        let snap = HealthSnapshot {
            drift: Some(MemoryDrift {
                heap_growth_bytes: -(3 * 1024 * 1024 * 1024),
                outside_heap_growth_bytes: 0,
                window_secs: 45 * 60,
                verdict: crate::app::mem_drift::DriftVerdict::Settled,
            }),
            ..Default::default()
        };
        assert_eq!(
            drift_text(&snap).expect("a baseline was captured"),
            "45m  heap -3 GB  outside +0 B"
        );
    }

    // A session measured in weeks is the one this line is built for, so its
    // window has to stay readable rather than running to four digits of hours.
    #[test]
    fn a_multi_day_window_reads_in_days() {
        assert_eq!(window_text(0), "0m");
        assert_eq!(window_text(59 * 60), "59m");
        assert_eq!(window_text(3 * 3600), "3h");
        assert_eq!(window_text(47 * 3600), "47h");
        assert_eq!(window_text(41 * 24 * 3600), "41d");
    }

    // No baseline yet is no line, rather than a row of zeroes claiming a
    // settled session that was never measured.
    #[test]
    fn the_drift_line_is_absent_before_a_baseline_exists() {
        assert_eq!(drift_text(&HealthSnapshot::default()), None);
    }

    // The size-class line is a development instrument: absent, not zeroed, in a
    // build without it.
    #[test]
    fn the_hot_class_line_is_absent_without_the_detail_feature() {
        assert_eq!(hot_class_text(&HealthSnapshot::default()), None);
    }

    #[test]
    fn the_hot_class_line_names_the_class_and_its_live_blocks() {
        let snap = HealthSnapshot {
            hot_class: Some(SizeClass {
                min_bytes: 128,
                max_bytes: 255,
                allocs: 900,
                live_blocks: 42,
            }),
            ..Default::default()
        };
        assert_eq!(
            hot_class_text(&snap).expect("a class was measured"),
            "hot class 128.0 B  |  42 live blocks"
        );
    }
}
