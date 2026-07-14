// src/hud/debug_hud.rs
//
// Developer debug-HUD overlay behavior. An internal system (not a declarable
// asset): `World::start` constructs one from the world's `DebugHud` component
// and it writes diagnostic readouts into that component's `TextLabel` chips.
//
// The chips are toggled together with F1 (hidden by default) and anchored to
// the top-right of the window by `GraphicsSystem` (it owns the font metrics and
// live window size needed to right-align and stack them).

use crate::assets::{Camera3D, DebugHud, FrameInput, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::profile::PassTiming;

// How many per-pass entries the passes chip lists. Picked to fit comfortably
// in the top-right debug column; passes past this count are dropped from the
// chip (still visible via the debug WS `profile.passes` reply).
const PASSES_CHIP_TOP_N: usize = 6;

// Build the per-pass timing chip text from the active backend's
// `RenderStats.pass_times_us` array. Picks the top
// [`PASSES_CHIP_TOP_N`] entries by GPU microseconds, descending, one per
// line as `name µs`. Returns an empty string when every slot is at the
// default `("", 0)` (the chip then renders nothing, DX/Vulkan keep it
// blank until their per-pass timing pools land).
//
// **Apple-GPU caveat:** the GPU overlaps fragment work across encoders,
// so summing these values exceeds `gpu_frame_us`. Display them as
// per-pass attributions, not as components of a total. The chip's
// "PASSES" header is meant to make that obvious at a glance: there is
// no row labelled "total".
fn passes_text(slots: &[PassTiming]) -> String {
    let mut entries: Vec<(&'static str, u32)> = slots
        .iter()
        .copied()
        .filter(|(name, micros)| !name.is_empty() && *micros > 0)
        .collect();
    if entries.is_empty() {
        return String::new();
    }
    // Sort descending by µs; stable so equal-time passes keep their
    // PassId-order (alphabetical-ish across a typical frame).
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));
    entries.truncate(PASSES_CHIP_TOP_N);
    let mut out = String::from("PASSES");
    for (name, micros) in entries {
        out.push('\n');
        out.push_str(name);
        out.push(' ');
        // Format compactly. < 1 ms reads as raw µs (e.g. "120 us"); ≥ 1 ms
        // reads as a single-decimal millisecond figure ("1.2 ms"). Keeps
        // the column width steady across the typical 50 µs to 5 ms band.
        if micros < 1000 {
            out.push_str(&format!("{micros} us"));
        } else {
            out.push_str(&format!("{:.1} ms", micros as f32 / 1000.0));
        }
    }
    out
}

// Build the cursor-position chip text from the latest window-space cursor
// coordinates (origin top-left). Rounded to whole pixels: sub-pixel jitter is
// not useful on a debug readout. The reading is only meaningful when the
// cursor is not captured (free-look worlds leave it stale); the chip still
// renders so a screenshot carries a reference point.
fn mouse_text(x: f32, y: f32) -> String {
    format!("MOUSE {x:.0}, {y:.0}")
}

// Build the camera-pose chip text from the live `Camera3D` pose, or blank when
// the world has no camera. The values are exactly what the debug `camera-set`
// command consumes, so a screenshot of this chip carries the arguments to
// reproduce the shot.
fn camera_text(pose: Option<([f32; 3], f32, f32)>) -> String {
    match pose {
        Some((p, yaw, pitch)) => {
            format!(
                "CAM {:.2} {:.2} {:.2}\nyaw {yaw:.3} pitch {pitch:.3}",
                p[0], p[1], p[2]
            )
        }
        None => String::new(),
    }
}

// Build the system-budget chip text from the process-level thread + memory
// budgets: the job pool's worker count against the machine's core count, and
// the process resident set against the memory budget. Each half degrades
// independently -- a missing `ThreadBudget` / `MemoryBudget` resource (the
// editor's in-memory preview) or an unavailable RSS query renders `--` rather
// than a stale or zeroed figure -- so the chip stays informative in any context.
fn sys_text(threads: Option<(usize, usize)>, rss: Option<u64>, budget_mib: Option<u64>) -> String {
    let threads_part = match threads {
        Some((job, cores)) => format!("threads {job}/{cores}"),
        None => "threads --".to_string(),
    };
    let mem_part = match (rss, budget_mib) {
        (Some(rss), Some(budget)) => format!("mem {}/{} MB", rss / (1024 * 1024), budget),
        (Some(rss), None) => format!("mem {} MB", rss / (1024 * 1024)),
        (None, _) => "mem -- MB".to_string(),
    };
    format!("{threads_part} | {mem_part}")
}

/// Draws the developer debug HUD: a multi-line `PASSES` chip listing the top
/// render-graph passes of the last frame, a `MOUSE` chip with the cursor
/// position, and a `CAM` chip with the live camera pose. The chips are anchored
/// to the top-right of the window (stacked cursor, passes, camera) and toggled
/// together with **F1**; the HUD starts hidden.
///
/// `PASSES` lists the top six passes in descending GPU-microsecond order
/// (e.g. `main 1.4 ms`, `shadow 380 us`). Filled on Metal when the device
/// exposes `MTLCommonCounterSetTimestamp`; blank on DirectX / Vulkan until
/// their per-pass timing pools land. **The values are per-pass attributions,
/// not components of `gpu_frame_us`**: the Apple GPU overlaps fragment work
/// across encoders, so summing them exceeds the whole-frame timer.
#[derive(Debug)]
pub struct DebugHudSystem {
    passes_label: Option<AssetId>,
    mouse_label: Option<AssetId>,
    camera_label: Option<AssetId>,
    sys_label: Option<AssetId>,
    // Whether the HUD is currently shown. Toggled by F1; hidden by default.
    visible: bool,
    // Most recent per-pass GPU microseconds (a snapshot of
    // `RenderStats.pass_times_us`); empty when the active backend has
    // no per-pass timing wired and the chip therefore stays blank.
    pass_times: Vec<PassTiming>,
    // Most recent cursor position (window pixels) from `FrameInput`.
    mouse_pos: (f32, f32),
    // Most recent live camera pose (position, yaw, pitch); `None` until a
    // `Camera3D` is seen (a world with no camera leaves the chip blank).
    camera_pose: Option<([f32; 3], f32, f32)>,
}

impl DebugHudSystem {
    // Build the debug HUD from a world's `DebugHud` request component.
    pub fn new(config: DebugHud) -> Self {
        Self {
            passes_label: config.passes_label,
            mouse_label: config.mouse_label,
            camera_label: config.camera_label,
            sys_label: config.sys_label,
            visible: false,
            pass_times: Vec::new(),
            mouse_pos: (0.0, 0.0),
            camera_pose: None,
        }
    }

    // Write `text` into the TextLabel with the given id, if it exists.
    fn write_chip(ctx: &mut PipelineContext, id: Option<AssetId>, text: String) {
        let Some(id) = id else {
            return;
        };
        for label in ctx.query_mut::<TextLabel>() {
            if label.asset_id == id {
                label.content = text;
                return;
            }
        }
    }
}

impl System for DebugHudSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // F1 toggles the debug HUD. The per-frame input snapshot is read from
        // the FrameInput resource GraphicsSystem publishes earlier in the frame;
        // it also carries the cursor position for the mouse chip.
        let frame_input = ctx.resource::<FrameInput>();
        let toggled = frame_input.is_some_and(|input| input.hud_toggle);
        if let Some(input) = frame_input {
            self.mouse_pos = (input.mouse_x, input.mouse_y);
        }
        if toggled {
            self.visible = !self.visible;
        }

        if !self.visible {
            // Blank chips read as empty content -> the renderer draws neither
            // text nor the background box, so the HUD fully disappears.
            Self::write_chip(ctx, self.passes_label, String::new());
            Self::write_chip(ctx, self.mouse_label, String::new());
            Self::write_chip(ctx, self.camera_label, String::new());
            Self::write_chip(ctx, self.sys_label, String::new());
            return StepResult::Continue;
        }

        // Snapshot the per-pass slots once per frame so the chip refresh works
        // off a single stable sample instead of racing the latest readback.
        self.pass_times.clear();
        self.pass_times
            .extend_from_slice(&ctx.profile.render.pass_times_us);
        // Live camera pose for the camera chip. Read here (before
        // Camera3DSystem steps) so a screenshot carries a settled reference
        // pose; one component read, cheap to refresh every frame.
        self.camera_pose = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| (c.position, c.yaw, c.pitch));

        // Process-level thread + memory budgets (published by App::start) and
        // the live process RSS. Copied out to owned values before the mutable
        // chip write; each half is optional and renders `--` when absent.
        let threads = ctx
            .resource::<crate::app::budget::ThreadBudget>()
            .map(|t| (t.job_threads, t.total_cores));
        let budget_mib = ctx
            .resource::<crate::app::budget::MemoryBudget>()
            .map(|b| b.budget_mib());
        let rss = crate::app::sysmem::process_resident_bytes();

        Self::write_chip(ctx, self.passes_label, passes_text(&self.pass_times));
        Self::write_chip(
            ctx,
            self.mouse_label,
            mouse_text(self.mouse_pos.0, self.mouse_pos.1),
        );
        Self::write_chip(ctx, self.camera_label, camera_text(self.camera_pose));
        Self::write_chip(ctx, self.sys_label, sys_text(threads, rss, budget_mib));
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_text_blanks_on_all_zero_slots() {
        // Every slot at the default ("", 0) → DX/Vulkan baseline → chip
        // stays empty so the HUD doesn't render an orphan "PASSES" header.
        let slots = vec![("", 0u32); 8];
        assert_eq!(passes_text(&slots), "");
    }

    #[test]
    fn passes_text_lists_top_entries_descending() {
        // Five recorded passes, three of them above the truncation cutoff
        // for a Top-N=6 chip. Order is by µs descending, regardless of the
        // input slot order.
        let slots = vec![
            ("shadow", 380),
            ("", 0),
            ("main", 1400),
            ("ssao_kernel", 120),
            ("composite", 60),
            ("ssr_resolve", 800),
            ("", 0),
        ];
        let out = passes_text(&slots);
        // Header first, then six lines for the recorded entries (well
        // under PASSES_CHIP_TOP_N so no truncation).
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "PASSES");
        assert_eq!(lines[1], "main 1.4 ms");
        assert_eq!(lines[2], "ssr_resolve 800 us");
        assert_eq!(lines[3], "shadow 380 us");
        assert_eq!(lines[4], "ssao_kernel 120 us");
        assert_eq!(lines[5], "composite 60 us");
        assert_eq!(lines.len(), 6);
    }

    #[test]
    fn passes_text_truncates_to_top_n() {
        // Eight non-empty slots, all distinct microsecond values → the
        // chip keeps only the top PASSES_CHIP_TOP_N (= 6) and drops the
        // smallest two.
        let slots: Vec<PassTiming> = vec![
            ("a", 80),
            ("b", 70),
            ("c", 60),
            ("d", 50),
            ("e", 40),
            ("f", 30),
            ("g", 20),
            ("h", 10),
        ];
        let out = passes_text(&slots);
        // Header line + exactly PASSES_CHIP_TOP_N entries.
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 1 + PASSES_CHIP_TOP_N);
        // The two smallest passes are dropped: never appear in the chip.
        assert!(!out.contains("g "));
        assert!(!out.contains("h "));
    }

    #[test]
    fn passes_text_formats_microseconds_below_one_ms() {
        // < 1000 µs reads as a bare integer; ≥ 1000 µs reads as a
        // single-decimal millisecond figure.
        let slots = vec![
            ("a", 999u32),
            ("b", 1000u32),
            ("c", 1499u32),
            ("d", 1500u32),
        ];
        let out = passes_text(&slots);
        assert!(out.contains("a 999 us"));
        assert!(out.contains("b 1.0 ms"));
        assert!(out.contains("c 1.5 ms"));
        assert!(out.contains("d 1.5 ms"));
    }

    #[test]
    fn mouse_text_rounds_to_whole_pixels() {
        assert_eq!(mouse_text(0.0, 0.0), "MOUSE 0, 0");
        // Sub-pixel coordinates round to the nearest whole pixel.
        assert_eq!(mouse_text(640.4, 360.6), "MOUSE 640, 361");
    }

    #[test]
    fn camera_text_formats_pose_in_camera_set_form() {
        // Position to two decimals, yaw/pitch to three, on two lines so the
        // chip reads back as the camera-set arguments.
        let out = camera_text(Some(([3.0, 1.6, 20.0], 1.2, -0.1)));
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "CAM 3.00 1.60 20.00");
        assert_eq!(lines[1], "yaw 1.200 pitch -0.100");
    }

    #[test]
    fn camera_text_blanks_without_camera() {
        // No Camera3D in the world: the chip stays empty rather than showing a
        // stale or zeroed pose.
        assert_eq!(camera_text(None), "");
    }

    #[test]
    fn sys_text_reports_threads_and_memory_when_known() {
        // Both budgets and the RSS present: the full "threads j/c | mem u/b MB"
        // line, memory truncated to whole MiB.
        assert_eq!(
            sys_text(Some((11, 12)), Some(512 * 1024 * 1024), Some(16384)),
            "threads 11/12 | mem 512/16384 MB"
        );
    }

    #[test]
    fn sys_text_degrades_each_half_independently() {
        // No ThreadBudget: the thread half reads `--`; the memory half still
        // renders from the RSS + budget.
        assert_eq!(
            sys_text(None, Some(256 * 1024 * 1024), Some(8192)),
            "threads -- | mem 256/8192 MB"
        );
        // No budget resource: the memory half drops the "/ budget" tail.
        assert_eq!(
            sys_text(Some((3, 4)), Some(256 * 1024 * 1024), None),
            "threads 3/4 | mem 256 MB"
        );
        // No RSS query: the memory half reads `--`.
        assert_eq!(
            sys_text(Some((3, 4)), None, Some(8192)),
            "threads 3/4 | mem -- MB"
        );
        // Nothing available (the editor's in-memory preview on an unsupported
        // target): a fully placeholdered line, never a panic.
        assert_eq!(sys_text(None, None, None), "threads -- | mem -- MB");
    }

    // A DebugHud component spawns the internal debug-HUD system.
    #[test]
    fn debug_hud_component_spawns_internal_system() {
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(DebugHud::default());
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["DebugHud"]);
    }

    // Build a world with a DebugHud wired to four chips, a camera (no
    // controller, so no camera system), and pre-filled chip labels.
    fn hud_world() -> crate::ecs::World {
        let mut world = crate::ecs::World::new_empty();
        world.add_component(DebugHud {
            passes_label: Some(AssetId(1)),
            mouse_label: Some(AssetId(2)),
            camera_label: Some(AssetId(3)),
            sys_label: Some(AssetId(4)),
        });
        for id in [1u32, 2, 3, 4] {
            world.add_component(TextLabel {
                asset_id: AssetId(id),
                content: "stale".to_string(),
                ..Default::default()
            });
        }
        world.add_component(Camera3D {
            fov_y_degrees: 75.0,
            near: 0.05,
            far: 200.0,
            view_matrix: [[0.0; 4]; 4],
            position: [1.0, 2.0, 3.0],
            yaw: 0.5,
            pitch: -0.2,
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller: None,
        });
        world
    }

    fn chip(world: &crate::ecs::World, id: u32) -> String {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(id))
            .map(|l| l.content.clone())
            .unwrap_or_default()
    }

    // Hidden by default: a step with no FrameInput blanks every chip (the
    // hidden branch), clearing the pre-filled content.
    #[test]
    fn hidden_hud_blanks_all_chips() {
        let mut world = hud_world();
        world.start().unwrap();
        world.step();
        assert_eq!(chip(&world, 1), "");
        assert_eq!(chip(&world, 2), "");
        assert_eq!(chip(&world, 3), "");
        assert_eq!(chip(&world, 4), "");
    }

    // The F1 toggle (a FrameInput with hud_toggle) reveals the HUD, so the same
    // step fills the mouse and camera chips from the live state.
    #[test]
    fn toggle_reveals_mouse_and_camera_chips() {
        let mut world = hud_world();
        world.start().unwrap();
        world.insert_resource(FrameInput {
            hud_toggle: true,
            mouse_x: 640.4,
            mouse_y: 360.6,
            ..Default::default()
        });
        world.step();
        assert_eq!(chip(&world, 2), "MOUSE 640, 361");
        assert_eq!(
            chip(&world, 3),
            "CAM 1.00 2.00 3.00\nyaw 0.500 pitch -0.200"
        );
    }

    // With the HUD revealed and both budget resources published, the sys chip
    // renders the thread + memory line off the live budgets and process RSS.
    #[test]
    fn toggle_reveals_sys_chip_from_budgets() {
        use crate::app::budget::{MemoryBudget, ThreadBudget};

        let mut world = hud_world();
        world.start().unwrap();
        world.insert_resource(ThreadBudget {
            total_cores: 12,
            job_threads: 11,
        });
        // The budget defaults to a fraction of total RAM, so derive the expected
        // MiB from the same value rather than assuming it equals total RAM.
        let budget = MemoryBudget::compute(Some(16 * 1024 * 1024 * 1024), 0);
        world.insert_resource(budget);
        world.insert_resource(FrameInput {
            hud_toggle: true,
            ..Default::default()
        });
        world.step();
        let sys = chip(&world, 4);
        assert!(sys.starts_with("threads 11/12 | mem "), "{sys}");
        // RSS is available on macOS / Linux / Windows; elsewhere it reads `--`.
        if cfg!(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )) {
            assert!(
                sys.ends_with(&format!("/{} MB", budget.budget_mib())),
                "{sys}"
            );
        } else {
            assert!(sys.ends_with("mem -- MB"), "{sys}");
        }
    }
}
