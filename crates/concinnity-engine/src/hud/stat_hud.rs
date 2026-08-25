// src/hud/stat_hud.rs
//
// Default stats-HUD overlay behavior. An internal system (not a declarable
// asset): `World::start` constructs one from the world's `StatHud` component
// and it writes live engine stats into that component's `TextLabel` chips.
//
// The frame-rate and GPU-memory chips are gated by the in-game video setting
// "Display performance stats" (published by `GraphicsSystem` as the `HudPrefs`
// resource); the exposure and HDR chips show whenever their feature is active.
// Developer readouts (passes / cursor / camera) live on `DebugHud`.

use crate::components::{StatHud, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{HudPrefs, PipelineContext, StepResult, System};
use std::time::Instant;

// How often the chip text is rebuilt, in seconds. The frame rate is averaged
// over this window so the number is readable rather than flickering.
const EMIT_INTERVAL_SECS: f32 = 0.5;

// Build the frame-rate chip text from a frame count over an elapsed window.
fn fps_text(frames: u32, elapsed_secs: f32) -> String {
    let fps = if elapsed_secs > 0.0 {
        frames as f32 / elapsed_secs
    } else {
        0.0
    };
    format!("FPS {fps:.0}")
}

// Build the GPU-memory chip text from a byte count.
fn vram_text(bytes: u64) -> String {
    format!("VRAM {} MB", bytes / (1024 * 1024))
}

// Build the host-memory chip text from the process resident set size, with the
// memory budget appended as `/ {budget} MB` when the budget resource is
// available. `None` RSS (an unsupported platform, or the editor's in-memory
// preview that does not query it) blanks the chip so the HUD shows no stale
// reading; a known RSS without a budget still reports the bare figure.
fn ram_text(rss: Option<u64>, budget_mib: Option<u64>) -> String {
    let Some(rss) = rss else {
        return String::new();
    };
    let rss_mib = rss / (1024 * 1024);
    match budget_mib {
        Some(budget) => format!("RAM {rss_mib} / {budget} MB"),
        None => format!("RAM {rss_mib} MB"),
    }
}

// Build the exposure-value chip text from the current auto-exposure EMA
// reading, or `None` when auto-exposure is not active for this world.
fn ev_text(ev: Option<f32>) -> String {
    match ev {
        Some(v) if v.is_finite() => format!("EV {v:+.2}"),
        _ => String::new(),
    }
}

// Build the EDR-headroom chip text from the active panel's maximum
// extended-range multiplier. `None` on SDR (the chip then stays empty so
// the HUD strip has no orphan reading on a non-HDR display).
fn edr_text(max_edr: Option<f32>) -> String {
    match max_edr {
        Some(v) if v.is_finite() && v > 0.0 => format!("EDR x{v:.1}"),
        _ => String::new(),
    }
}

// Draws the default stats HUD: an `FPS` chip, a `VRAM` chip, an `EV` chip (when
// auto-exposure is on), and an `EDR` chip (when the renderer is on the HDR
// display path), each written into its own `TextLabel`. Give the labels a
// `background` colour for the boxed look.
//
// The `FPS` and `VRAM` chips are shown or hidden from the in-game video
// settings ("Display performance stats" + per-readout toggles); the `EV` and
// `EDR` chips appear automatically whenever their feature is active. The
// per-system timing breakdown, cursor position, and camera pose are on the
// separate `DebugHud` (F1).
//
// `VRAM` is the render device's current allocation; on Apple Silicon's
// unified memory that is its share of system RAM. Filled on Metal
// (`MTLDevice.currentAllocatedSize`), DirectX
// (`IDXGIAdapter3::QueryVideoMemoryInfo(LOCAL).CurrentUsage`), and Vulkan
// (`VK_EXT_memory_budget`).
//
// `EV` is the adapted exposure value the auto-exposure EMA settled on for
// the most recent frame (the multiplier the post-process stack uses is
// `2^ev`). Filled when the world's `PostProcessConfig` enables auto-exposure;
// the chip stays blank otherwise so a world without auto-exposure has no orphan
// reading.
//
// `EDR` is the active panel's maximum extended-range colour-component
// multiplier (e.g. `EDR x2.0` on an HDR400 panel, `x8.0` on HDR1000).
// Filled on Metal when `PostProcessConfig.hdr_display = true` AND the
// platform reports an EDR headroom above SDR reference white; the chip
// stays blank on SDR or when the HDR request fell back.
//
// ```jsonl
// {"type":"Font","name":"hud_font","args":{"size_px":20}}
// {"type":"TextLabel","name":"fps_chip","args":{"font":"hud_font","x":10,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
// {"type":"TextLabel","name":"vram_chip","args":{"font":"hud_font","x":92,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
// {"type":"TextLabel","name":"ev_chip","args":{"font":"hud_font","x":192,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
// {"type":"TextLabel","name":"edr_chip","args":{"font":"hud_font","x":272,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
// {"type":"StatHud","name":"hud","args":{"fps_label":"fps_chip","vram_label":"vram_chip","ev_label":"ev_chip","edr_label":"edr_chip"}}
// ```
#[derive(Debug)]
pub struct StatHudSystem {
    fps_label: Option<AssetId>,
    vram_label: Option<AssetId>,
    ram_label: Option<AssetId>,
    ev_label: Option<AssetId>,
    edr_label: Option<AssetId>,
    // Start of the current averaging window.
    last_emit: Instant,
    // Frames counted since `last_emit`.
    frames: u32,
    // Most recent GPU-memory sample, bytes.
    vram_bytes: u64,
    // Most recent host resident-set size, bytes; `None` when the platform
    // query is unavailable (the chip is then blanked). Sampled on the throttled
    // emit tick, not per frame, so the syscall runs at most twice a second.
    ram_bytes: Option<u64>,
    // Most recent auto-exposure EV; `None` when auto-exposure is not active
    // for this world (the chip is then blanked).
    ev: Option<f32>,
    // Most recent panel EDR multiplier; `None` on the SDR path (the chip
    // is then blanked).
    max_edr: Option<f32>,
}

impl StatHudSystem {
    // Build the HUD from a world's `StatHud` request component.
    pub fn new(config: StatHud) -> Self {
        Self {
            fps_label: config.fps_label,
            vram_label: config.vram_label,
            ram_label: config.ram_label,
            ev_label: config.ev_label,
            edr_label: config.edr_label,
            last_emit: Instant::now(),
            frames: 0,
            vram_bytes: 0,
            ram_bytes: None,
            ev: None,
            max_edr: None,
        }
    }

    // Write `text` into the TextLabel with the given id, if it exists.
    fn write_chip(ctx: &mut PipelineContext, id: Option<AssetId>, text: String) {
        crate::ecs::by_asset_id::update::<TextLabel>(ctx, id, |l| l.content = text);
    }
}

impl System for StatHudSystem {
    fn access(&self) -> crate::ecs::Access {
        crate::ecs::Access::new()
            .writes_components(crate::component_mask![crate::components::TextLabel])
            .reads_resources(crate::resource_mask![
                crate::ecs::HudPrefs,
                crate::app::budget::MemoryBudget,
            ])
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Per-chip visibility from the "Display performance stats" video
        // settings, published each frame by GraphicsSystem. Absent (a HUD-only
        // unit test with no GraphicsSystem) -> both shown, matching the old
        // default-visible behavior.
        let (show_fps, show_vram) = ctx
            .resource::<HudPrefs>()
            .map_or((true, true), |p| (p.show_fps, p.show_vram));

        self.frames += 1;
        self.vram_bytes = ctx.profile.render.vram_bytes;
        self.ev = ctx.profile.render.auto_exposure_ev;
        self.max_edr = ctx.profile.render.max_edr;

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_emit).as_secs_f32();
        if elapsed >= EMIT_INTERVAL_SECS {
            // Host resident-set size + memory budget, sampled here (on the
            // throttled tick) so the syscall runs at most every EMIT_INTERVAL_SECS
            // rather than every frame. The budget resource is absent in the
            // editor's in-memory preview, so `ram_text` reports the bare RSS then.
            self.ram_bytes = crate::app::sysmem::process_resident_bytes();
            let budget_mib = ctx
                .resource::<crate::app::budget::MemoryBudget>()
                .map(|b| b.budget_mib());
            // A blank chip reads as empty content -> the renderer draws neither
            // text nor the background box, so a disabled chip fully disappears.
            Self::write_chip(
                ctx,
                self.fps_label,
                if show_fps {
                    fps_text(self.frames, elapsed)
                } else {
                    String::new()
                },
            );
            Self::write_chip(
                ctx,
                self.vram_label,
                if show_vram {
                    vram_text(self.vram_bytes)
                } else {
                    String::new()
                },
            );
            Self::write_chip(ctx, self.ram_label, ram_text(self.ram_bytes, budget_mib));
            Self::write_chip(ctx, self.ev_label, ev_text(self.ev));
            Self::write_chip(ctx, self.edr_label, edr_text(self.max_edr));
            self.frames = 0;
            self.last_emit = now;
        }
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_text_averages_frames_over_window() {
        // 60 frames in 1.0 s -> 60 FPS.
        assert_eq!(fps_text(60, 1.0), "FPS 60");
        // 75 frames in 0.5 s -> 150 FPS.
        assert_eq!(fps_text(75, 0.5), "FPS 150");
    }

    #[test]
    fn fps_text_handles_zero_window() {
        assert_eq!(fps_text(0, 0.0), "FPS 0");
    }

    #[test]
    fn vram_text_reports_whole_megabytes() {
        assert_eq!(vram_text(0), "VRAM 0 MB");
        assert_eq!(vram_text(512 * 1024 * 1024), "VRAM 512 MB");
        // Truncates to whole MB.
        assert_eq!(vram_text(1024 * 1024 + 1), "VRAM 1 MB");
    }

    #[test]
    fn ram_text_reports_rss_with_budget_when_known() {
        // Known RSS and budget render as "used / budget MB", both truncated to
        // whole MiB.
        assert_eq!(
            ram_text(Some(512 * 1024 * 1024), Some(16384)),
            "RAM 512 / 16384 MB"
        );
    }

    #[test]
    fn ram_text_reports_bare_rss_without_a_budget() {
        // No budget resource (the editor's in-memory preview): report the RSS
        // alone rather than a dangling "/ MB".
        assert_eq!(ram_text(Some(256 * 1024 * 1024), None), "RAM 256 MB");
    }

    #[test]
    fn ram_text_blanks_when_rss_unavailable() {
        // `None` RSS (an unsupported platform): the chip stays empty rather than
        // showing a stale or zeroed figure.
        assert_eq!(ram_text(None, Some(16384)), "");
        assert_eq!(ram_text(None, None), "");
    }

    #[test]
    fn ev_text_formats_signed_value_with_two_decimals() {
        // Positive bias keeps the leading "+" so the sign is unambiguous on
        // screen.
        assert_eq!(ev_text(Some(1.25)), "EV +1.25");
        assert_eq!(ev_text(Some(-0.5)), "EV -0.50");
        assert_eq!(ev_text(Some(0.0)), "EV +0.00");
    }

    #[test]
    fn ev_text_blanks_when_auto_exposure_off() {
        // `None` means the world did not opt in to auto-exposure; the chip
        // stays empty so the HUD doesn't show a stale or misleading value.
        assert_eq!(ev_text(None), "");
    }

    #[test]
    fn ev_text_blanks_on_non_finite_values() {
        // A backend bug could leave the EV at NaN or infinity; render blank
        // rather than print "NaN" in the HUD.
        assert_eq!(ev_text(Some(f32::NAN)), "");
        assert_eq!(ev_text(Some(f32::INFINITY)), "");
    }

    #[test]
    fn edr_text_formats_multiplier_with_one_decimal() {
        // HDR400-class panels typically report 2.0; HDR1000-class 8.0+.
        // One decimal place keeps the chip narrow and unambiguous.
        assert_eq!(edr_text(Some(2.0)), "EDR x2.0");
        assert_eq!(edr_text(Some(8.5)), "EDR x8.5");
    }

    #[test]
    fn edr_text_blanks_when_sdr() {
        // `None` from the renderer means the SDR path is active: the chip
        // should not show an "EDR x1.0" reading because there is no HDR
        // headroom in play.
        assert_eq!(edr_text(None), "");
    }

    #[test]
    fn edr_text_blanks_on_invalid_values() {
        // Defensive: a backend reporting non-finite or non-positive max_edr
        // should not surface as "EDR xNaN" / "EDR x-1.0" on screen.
        assert_eq!(edr_text(Some(f32::NAN)), "");
        assert_eq!(edr_text(Some(f32::INFINITY)), "");
        assert_eq!(edr_text(Some(0.0)), "");
        assert_eq!(edr_text(Some(-1.0)), "");
    }

    // A StatHud component spawns the internal HUD system.
    #[test]
    fn stat_hud_component_spawns_internal_system() {
        use crate::components::StatHud;
        use crate::ecs::World;

        let mut world = World::new();
        world.add_component(StatHud::default());
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["StatHud"]);
    }

    #[test]
    fn no_stat_hud_no_system() {
        use crate::ecs::World;

        let mut world = World::new();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }

    // A world carrying a StatHud wired to fps + vram + ram chips and their
    // labels.
    fn hud_world() -> crate::ecs::World {
        let mut world = crate::ecs::World::new();
        world.add_component(StatHud {
            fps_label: Some(AssetId(1)),
            vram_label: Some(AssetId(2)),
            ram_label: Some(AssetId(3)),
            ..StatHud::default()
        });
        for id in [1u32, 2, 3] {
            world.add_component(TextLabel {
                asset_id: AssetId(id),
                ..Default::default()
            });
        }
        world
    }

    // Backdate the emit window so the next step crosses EMIT_INTERVAL_SECS
    // without a real sleep (the field is injectable in-file).
    fn force_emit_due(world: &mut crate::ecs::World) {
        use std::time::Duration;
        for system in world.systems_mut() {
            if let crate::ecs::SystemAsset::StatHud(s) = system {
                s.last_emit = Instant::now() - Duration::from_secs(1);
            }
        }
    }

    fn chip(world: &crate::ecs::World, id: u32) -> String {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(id))
            .map(|l| l.content.clone())
            .unwrap_or_default()
    }

    // Once the averaging window elapses, the step body writes the FPS and VRAM
    // chips (both shown when no HudPrefs is published).
    #[test]
    fn emit_window_writes_fps_and_vram_chips() {
        let mut world = hud_world();
        world.start().unwrap();
        force_emit_due(&mut world);
        world.step();
        assert!(chip(&world, 1).starts_with("FPS "), "{}", chip(&world, 1));
        assert_eq!(chip(&world, 2), "VRAM 0 MB");
    }

    // The RAM chip is always-on (not gated by HudPrefs): on a platform that
    // reports RSS it fills with a "RAM ..." reading, and the published
    // MemoryBudget resource appends the "/ budget MB" tail.
    #[test]
    fn emit_window_writes_ram_chip_with_budget() {
        use crate::app::budget::MemoryBudget;

        let mut world = hud_world();
        world.start().unwrap();
        // The budget defaults to a fraction of total RAM, so derive the expected
        // MiB from the same value rather than assuming it equals total RAM.
        let budget = MemoryBudget::compute(Some(16 * 1024 * 1024 * 1024), 0);
        world.insert_resource(budget);
        force_emit_due(&mut world);
        world.step();
        // RSS is available on macOS / Linux / Windows; other targets report
        // `None`, blanking the chip (so the assertion is platform-gated).
        if cfg!(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )) {
            let ram = chip(&world, 3);
            assert!(ram.starts_with("RAM "), "{ram}");
            assert!(
                ram.ends_with(&format!(" / {} MB", budget.budget_mib())),
                "{ram}"
            );
        }
    }

    // The HudPrefs resource (published by GraphicsSystem) gates each chip: with
    // both off, the fps and vram chips blank on the next emit.
    #[test]
    fn hud_prefs_hide_fps_and_vram_chips() {
        use crate::ecs::HudPrefs;

        let mut world = hud_world();
        world.start().unwrap();
        world.insert_resource(HudPrefs {
            show_fps: false,
            show_vram: false,
        });
        force_emit_due(&mut world);
        world.step();
        assert_eq!(chip(&world, 1), "", "fps chip hidden");
        assert_eq!(chip(&world, 2), "", "vram chip hidden");
    }
}
