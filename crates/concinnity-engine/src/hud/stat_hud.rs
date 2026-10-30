//! Default stats-HUD overlay behavior. An internal system (not a declarable
//! asset): `World::start` constructs one from the world's `StatHud` component
//! and it writes live engine stats into that component's `TextLabel` chips.
//!
//! The frame-rate, frame-time, allocation, and GPU-memory chips are gated by the in-game
//! video setting "Display performance stats" (published by the renderer as the
//! `HudPrefs` resource); the other chips show whenever their reading exists.
//! Developer readouts (passes / cursor / camera) live on `DebugHud`.

use crate::ecs::{
    Access, FrameTime, HudHidden, HudPrefs, PipelineContext, Ref, StepResult, System, World,
};
use concinnity_core::math::SaturatingFrom;
use core::fmt::{self, Display, Write as _};

use crate::benchmark::BenchmarkRun;
use crate::budget::{MemoryBudget, ResidentMemory};
use crate::camera::track::CameraTrackStatus;
use crate::components::TextLabel;
use crate::platform::GraphicsApiStatus;
use crate::profile::RenderResolution;
use crate::render::streaming::pools::StreamingRuntime;

use super::rate_window::RateWindow;

// How often the chip text is rebuilt, in seconds. The rates are averaged over
// this window so the numbers are readable rather than flickering.
const EMIT_INTERVAL_SECS: f32 = 0.5;

// The frame rate over a window of `frames` frames lasting `elapsed_secs`.
fn fps_text(frames: u32, elapsed_secs: f32) -> impl Display {
    let fps = if elapsed_secs > 0.0 {
        frames as f32 / elapsed_secs
    } else {
        0.0
    };
    fmt::from_fn(move |out| write!(out, "FPS {fps:.0}"))
}

// A microsecond reading as milliseconds under `name`.
fn ms_text(name: &'static str, micros: u64) -> impl Display {
    fmt::from_fn(move |out| write!(out, "{name} {:.1} ms", micros as f32 / 1000.0))
}

// The longest frame of the window.
fn worst_frame_text(secs: f32) -> impl Display {
    fmt::from_fn(move |out| write!(out, "MAX {:.1} ms", secs * 1000.0))
}

// Heap allocations per frame over the window, and the most any one frame made.
// Blank in a program without the counting allocator.
fn alloc_text(window: Option<AllocWindow>, frames: u32) -> impl Display {
    fmt::from_fn(move |out| match window {
        Some(w) => write!(
            out,
            "ALLOC {}/f max {}",
            w.sum / u64::from(frames.max(1)),
            w.max
        ),
        None => Ok(()),
    })
}

// The window's heap allocations: their sum over its frames and the largest
// single frame's.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct AllocWindow {
    sum: u64,
    max: u32,
}

fn vram_text(bytes: u64) -> impl Display {
    fmt::from_fn(move |out| write!(out, "VRAM {} MB", bytes / (1024 * 1024)))
}

// The process resident set size, against the memory budget when it is known.
// Blank without a reading (an unsupported platform, or the editor's in-memory
// preview that does not query it).
fn ram_text(rss: Option<u64>, budget_mib: Option<u64>) -> impl Display {
    fmt::from_fn(move |out| {
        let Some(rss) = rss else {
            return Ok(());
        };
        let rss_mib = rss / (1024 * 1024);
        match budget_mib {
            Some(budget) => write!(out, "RAM {rss_mib} / {budget} MB"),
            None => write!(out, "RAM {rss_mib} MB"),
        }
    })
}

// Loads in flight across the streaming pools, blank when nothing streams.
fn stream_text(pending: Option<usize>) -> impl Display {
    fmt::from_fn(move |out| match pending {
        Some(n) => write!(out, "STREAM {n} pending"),
        None => Ok(()),
    })
}

// The adapted auto-exposure value, blank when auto-exposure is off.
fn ev_text(ev: Option<f32>) -> impl Display {
    fmt::from_fn(move |out| match ev {
        Some(v) if v.is_finite() => write!(out, "EV {v:+.2}"),
        _ => Ok(()),
    })
}

// The panel's extended-range headroom, blank on SDR.
fn edr_text(max_edr: Option<f32>) -> impl Display {
    fmt::from_fn(move |out| match max_edr {
        Some(v) if v.is_finite() && v > 0.0 => write!(out, "EDR x{v:.1}"),
        _ => Ok(()),
    })
}

fn time_text(secs: f32) -> impl Display {
    fmt::from_fn(move |out| write!(out, "TIME {secs:.1} s"))
}

// The benchmark phase and how far into it the run is, or the warm-up before
// the first phase.
fn phase_text(phase: Option<(&str, f32, f32)>, warmup_left: f32) -> impl Display {
    fmt::from_fn(move |out| match phase {
        Some((name, into, length)) => write!(out, "PHASE {name} {into:.1}/{length:.1} s"),
        None => write!(out, "PHASE warmup {warmup_left:.1} s"),
    })
}

fn api_text(label: &'static str) -> impl Display {
    fmt::from_fn(move |out| write!(out, "API {label}"))
}

// The scene's render size, and the output size and upscaler when it is
// upscaled. Blank before the backend reports a size.
fn resolution_text(res: RenderResolution) -> impl Display {
    fmt::from_fn(move |out| {
        let [w, h] = res.render;
        if w == 0 || h == 0 {
            return Ok(());
        }
        write!(out, "RES {w}x{h}")?;
        match res.upscaler {
            Some(name) => write!(out, " -> {}x{} {name}", res.output[0], res.output[1]),
            None => Ok(()),
        }
    })
}

fn count_text(name: &'static str, count: u32) -> impl Display {
    fmt::from_fn(move |out| write!(out, "{name} {count}"))
}

// Blank when `show` is false, `text` otherwise.
fn shown(show: bool, text: impl Display) -> impl Display {
    fmt::from_fn(move |out| if show { text.fmt(out) } else { Ok(()) })
}

// What the per-frame benchmark chips last showed, in tenths of a second, so
// they are rewritten only when the text would change.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct ClockKey {
    time: u32,
    phase: Option<u32>,
    into: u32,
}

/// The default stats HUD, built from a world's `StatHud` request component.
#[derive(Debug)]
pub(crate) struct StatHudSystem {
    labels: StatHud,
    // The frame-rate averaging window, which also paces the emit.
    window: RateWindow,
    // Sums and the maximum over the frames of the current window, so each
    // chip averages over the same frames as the frame rate.
    cpu_sum_us: u64,
    gpu_sum_us: u64,
    worst_dt: f32,
    allocs: Option<AllocWindow>,
    // Whether the HUD was hidden by a `toggle_hud` action as of last step.
    hidden: bool,
    // The benchmark the run was armed with, read once at init. Its presence
    // turns on the benchmark row.
    benchmark: Option<BenchmarkRun>,
    // What the time and phase chips last showed.
    clock: Option<ClockKey>,
}

impl StatHudSystem {
    /// The system for a world that declares a `StatHud`.
    pub(crate) fn for_world(world: &World) -> Option<Self> {
        world.query::<StatHud>().next().map(Self::new)
    }

    /// Build the HUD from a world's `StatHud` request component.
    pub(crate) fn new(config: &StatHud) -> Self {
        Self {
            labels: config.clone(),
            window: RateWindow::default(),
            cpu_sum_us: 0,
            gpu_sum_us: 0,
            worst_dt: 0.0,
            allocs: None,
            hidden: false,
            benchmark: None,
            clock: None,
        }
    }

    // Write `text` into the TextLabel `label`, if it exists, reusing the
    // label's own string.
    fn write_chip(
        ctx: &mut PipelineContext<'_>,
        label: Option<Ref<TextLabel>>,
        text: impl Display,
    ) {
        crate::ecs::by_asset_id::update::<TextLabel>(ctx, label.map(Ref::id), |l| {
            l.content.clear();
            // Writing into a String cannot fail.
            write!(l.content, "{text}").unwrap_or_default();
        });
    }

    // Blank every chip, which hides it: a blank chip draws neither text nor
    // its background box.
    fn blank_all(&self, ctx: &mut PipelineContext<'_>) {
        for label in self
            .labels
            .top_row()
            .into_iter()
            .chain(self.labels.bench_row())
        {
            Self::write_chip(ctx, label, "");
        }
    }

    // The benchmark row's clock chips, rewritten whenever their tenth of a
    // second changes so a moment seen on screen can be found in the report.
    fn write_clock(&mut self, ctx: &mut PipelineContext<'_>) {
        let Some(status) = ctx.resource::<CameraTrackStatus>().copied() else {
            return;
        };
        let secs = status.elapsed_seconds;
        let run = self.benchmark.as_ref();
        let phase = run.and_then(|run| {
            let at = run.phase_at(secs)?;
            let (start, end) = run.phase_span(at)?;
            let name = run.phase_names().get(at as usize)?;
            Some((
                at,
                name.as_str(),
                (secs - start).max(0.0).min(end - start),
                end - start,
            ))
        });
        let tenths = |s: f32| u32::saturating_from(s * 10.0);
        let key = ClockKey {
            time: tenths(secs),
            phase: phase.map(|(at, ..)| at),
            into: phase.map_or_else(|| tenths(secs), |(_, _, into, _)| tenths(into)),
        };
        if self.clock == Some(key) {
            return;
        }
        self.clock = Some(key);
        Self::write_chip(ctx, self.labels.time_label, time_text(secs));
        if let Some(run) = run {
            let warmup_left = (run.config.warmup_seconds - secs).max(0.0);
            let text = phase_text(
                phase.map(|(_, name, into, len)| (name, into, len)),
                warmup_left,
            );
            Self::write_chip(ctx, self.labels.phase_label, text);
        }
    }

    // Rewrite the window-paced chips from the window that just closed.
    fn emit(&self, ctx: &mut PipelineContext<'_>, frames: u32, elapsed: f32) {
        // Per-chip visibility from the "Display performance stats" video
        // settings, published by SettingsSystem. Absent (no settings state),
        // both are shown.
        let (show_fps, show_vram) = ctx
            .resource::<HudPrefs>()
            .map_or((true, true), |p| (p.show_fps, p.show_vram));
        let render = ctx.profile.render;
        let frames_u64 = u64::from(frames.max(1));
        // Host resident-set size, sampled here so the query runs at most every
        // EMIT_INTERVAL_SECS rather than every frame.
        let rss = ctx
            .resource::<ResidentMemory>()
            .and_then(ResidentMemory::read);
        let budget_mib = ctx.resource::<MemoryBudget>().map(MemoryBudget::budget_mib);
        let pending = ctx
            .resource::<StreamingRuntime>()
            .and_then(|s| s.stats().pending());
        let labels = &self.labels;

        Self::write_chip(
            ctx,
            labels.fps_label,
            shown(show_fps, fps_text(frames, elapsed)),
        );
        Self::write_chip(
            ctx,
            labels.worst_frame_label,
            shown(show_fps, worst_frame_text(self.worst_dt)),
        );
        Self::write_chip(
            ctx,
            labels.cpu_label,
            shown(show_fps, ms_text("CPU", self.cpu_sum_us / frames_u64)),
        );
        Self::write_chip(
            ctx,
            labels.gpu_label,
            shown(show_fps, ms_text("GPU", self.gpu_sum_us / frames_u64)),
        );
        Self::write_chip(
            ctx,
            labels.allocs_label,
            shown(show_fps, alloc_text(self.allocs, frames)),
        );
        Self::write_chip(
            ctx,
            labels.vram_label,
            shown(show_vram, vram_text(render.vram_bytes)),
        );
        Self::write_chip(ctx, labels.ram_label, ram_text(rss, budget_mib));
        Self::write_chip(ctx, labels.streaming_label, stream_text(pending));
        Self::write_chip(ctx, labels.ev_label, ev_text(render.auto_exposure_ev));
        Self::write_chip(ctx, labels.edr_label, edr_text(render.max_edr));

        if self.benchmark.is_none() {
            return;
        }
        if let Some(api) = ctx
            .resource::<GraphicsApiStatus>()
            .map(|s| s.current.label())
        {
            Self::write_chip(ctx, labels.api_label, api_text(api));
        }
        Self::write_chip(
            ctx,
            labels.resolution_label,
            resolution_text(render.resolution),
        );
        Self::write_chip(
            ctx,
            labels.objects_label,
            count_text("OBJECTS", render.objects),
        );
        Self::write_chip(
            ctx,
            labels.draws_label,
            count_text("DRAWS", render.draw_calls),
        );
    }
}

impl System for StatHudSystem {
    fn access(&self) -> Access {
        Access::new()
            .writes_components(crate::component_mask![TextLabel])
            .reads_resources(crate::resource_mask![
                crate::ecs::EntityById,
                HudPrefs,
                HudHidden,
                FrameTime,
                MemoryBudget,
                ResidentMemory,
                CameraTrackStatus,
                GraphicsApiStatus,
                StreamingRuntime,
            ])
    }

    fn init(&mut self, ctx: &mut PipelineContext<'_>) {
        self.benchmark = ctx.resource::<BenchmarkRun>().cloned();
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        let dt = ctx.resource::<FrameTime>().copied().unwrap_or_default().dt;
        self.cpu_sum_us += u64::from(ctx.profile.cpu_work_us());
        self.gpu_sum_us += u64::from(ctx.profile.render.gpu_frame_us);
        self.worst_dt = self.worst_dt.max(dt);
        if let Some(n) = ctx.profile.frame_allocs() {
            let w = self.allocs.get_or_insert_default();
            w.sum += u64::from(n);
            w.max = w.max.max(n);
        }

        let hidden = ctx.resource::<HudHidden>().is_some_and(|h| h.0);
        let revealed = self.hidden && !hidden;
        if hidden && !self.hidden {
            self.blank_all(ctx);
            self.clock = None;
        }
        self.hidden = hidden;
        if !hidden {
            // Every frame rather than on the emit tick: a moment called out on
            // screen should match the report to the tenth of a second.
            self.write_clock(ctx);
        }

        // A HUD shown again fills at once from the window so far rather than
        // waiting for it to close.
        let window = self
            .window
            .tick(dt, EMIT_INTERVAL_SECS)
            .or_else(|| revealed.then(|| self.window.close()));
        if let Some((frames, elapsed)) = window {
            if !hidden {
                self.emit(ctx, frames, elapsed);
            }
            self.cpu_sum_us = 0;
            self.gpu_sum_us = 0;
            self.worst_dt = 0.0;
            self.allocs = None;
        }
        StepResult::Continue
    }
}

/// Requests the default on-screen stats HUD: two rows of
/// [`TextLabel`](#textlabel) chips with live engine stats, refreshed on a fixed
/// interval and packed at the top-left of the window.
///
/// The top row is the run's health, left to right:
/// - `fps_label`: the frame rate, averaged over the refresh interval.
/// - `worst_frame_label`: the longest frame of that interval (`MAX 23.4 ms`),
///   which shows a hitch the average hides.
/// - `cpu_label`: the CPU work per frame: every system's step plus the render
///   side's submission, with any wait on the GPU left out.
/// - `gpu_label`: the GPU time per frame.
/// - `allocs_label`: heap allocations per frame, and the most any one frame
///   of the interval made (`ALLOC 3/f max 40`). Counted across the whole
///   process, background loading included. Blank in a program built without
///   the counting allocator.
/// - `vram_label`: the GPU-memory use. On unified-memory hardware (Apple
///   Silicon) that is the device's share of system memory.
/// - `ram_label`: the process resident set size, against the memory budget
///   when it is known.
/// - `streaming_label`: loads in flight across the streaming pools (textures,
///   meshes, chunks, planet tiles).
/// - `ev_label`: the adapted auto-exposure value.
/// - `edr_label`: the HDR panel's headroom multiplier (`EDR x2.0` on an
///   HDR400 panel).
///
/// Comparing `CPU` with `GPU` tells which side bounds the frame: the larger
/// one is near the frame time and the other has headroom. Both are averaged
/// over the same frames as `FPS`.
///
/// The second row appears in a benchmarked run (one armed with a
/// [`BenchmarkConfig`](#benchmarkconfig)):
/// - `time_label`: how far the [`CameraTrack`](#cameratrack) has run, on the
///   clock the benchmark report times its frames on, so a moment seen on
///   screen (`TIME 47.3 s`) can be found in the report.
/// - `phase_label`: the benchmark phase and how far into it the run is
///   (`PHASE ground 12.4/60.0 s`).
/// - `api_label`: the graphics API the run renders with.
/// - `resolution_label`: the size the scene renders at and, when it is
///   upscaled, the size it is shown at and the upscaler
///   (`RES 1280x720 -> 2560x1440 MetalFX`).
/// - `objects_label`: the renderable objects in the scene.
/// - `draws_label`: the draw calls the CPU issued. Draws the GPU culls and
///   issues for itself are not counted.
///
/// `time_label` also shows in a run without a benchmark whenever a track is
/// running. Chips whose stat is unavailable stay blank, and a blank chip
/// reserves no width. The chips' on-screen position is fixed by the engine
/// rather than the authored coordinates.
///
/// The frame-rate, frame-time, allocation, and GPU-memory chips are shown or
/// hidden from the in-game video settings ("Display performance stats"). A
/// `toggle_hud` action hides and shows the whole HUD.
///
/// Developer-facing readouts (per-pass GPU timings, cursor position, live
/// camera pose) live on the separate [`DebugHud`](#debughud), toggled with F1.
///
/// A world that declares a [`MainMenu`](#mainmenu) receives a `StatHud` from the
/// build when it declares none, since the menu's performance-stats toggles
/// drive the chips, and any label field left unset receives a chip at start.
/// So the example below is only needed to restyle the chips or run a HUD
/// without a menu. Declare an [`EngineDefaults`](#enginedefaults) with
/// `"hud": false` to leave the chips unfilled.
///
/// ```jsonl
/// ["Font",{"$id":"hud_font","size_px":20}]
/// ["TextLabel",{"$id":"fps_chip","font":"hud_font","scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}]
/// ["StatHud",{"fps_label":"fps_chip"}]
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StatHud {
    /// [`TextLabel`](#textlabel) that receives the frame-rate chip text.
    pub fps_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the longest-frame chip text.
    pub worst_frame_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the CPU-time chip text.
    pub cpu_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the GPU-time chip text.
    pub gpu_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the heap-allocation chip text.
    pub allocs_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the GPU-memory chip text.
    pub vram_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the host-memory (RSS) chip text.
    pub ram_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the streaming chip text.
    pub streaming_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the auto-exposure chip text.
    pub ev_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the HDR-headroom chip text.
    pub edr_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the run-time chip text.
    pub time_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the benchmark-phase chip text.
    pub phase_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the graphics-API chip text.
    pub api_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the resolution chip text.
    pub resolution_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the object-count chip text.
    pub objects_label: Option<Ref<TextLabel>>,
    /// [`TextLabel`](#textlabel) that receives the draw-call chip text.
    pub draws_label: Option<Ref<TextLabel>>,
}

impl StatHud {
    /// The top row's labels, left to right.
    pub(crate) fn top_row(&self) -> [Option<Ref<TextLabel>>; 10] {
        [
            self.fps_label,
            self.worst_frame_label,
            self.cpu_label,
            self.gpu_label,
            self.allocs_label,
            self.vram_label,
            self.ram_label,
            self.streaming_label,
            self.ev_label,
            self.edr_label,
        ]
    }

    /// The benchmark row's labels, left to right.
    pub(crate) fn bench_row(&self) -> [Option<Ref<TextLabel>>; 6] {
        [
            self.time_label,
            self.phase_label,
            self.api_label,
            self.resolution_label,
            self.objects_label,
            self.draws_label,
        ]
    }
}

#[cfg(test)]
mod tests;
