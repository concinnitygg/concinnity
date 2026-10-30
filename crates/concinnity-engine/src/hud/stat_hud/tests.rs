use super::*;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Phase, World};
use crate::test_support::start_with;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;

fn start(world: &mut World) {
    let config = world.query::<StatHud>().next().cloned().unwrap();
    start_with(
        world,
        Phase::Late,
        "StatHudSystem",
        StatHudSystem::new(&config),
    );
}

fn label(id: u32) -> Ref<TextLabel> {
    Ref::new(AssetId(id))
}

// A world carrying a StatHud with every chip wired to its own label: `1..=15`
// in field order, then the allocation chip at 16.
fn hud_world() -> World {
    let mut world = World::new();
    world.add_component(StatHud {
        fps_label: Some(label(1)),
        worst_frame_label: Some(label(2)),
        cpu_label: Some(label(3)),
        gpu_label: Some(label(4)),
        vram_label: Some(label(5)),
        ram_label: Some(label(6)),
        streaming_label: Some(label(7)),
        ev_label: Some(label(8)),
        edr_label: Some(label(9)),
        time_label: Some(label(10)),
        phase_label: Some(label(11)),
        api_label: Some(label(12)),
        resolution_label: Some(label(13)),
        objects_label: Some(label(14)),
        draws_label: Some(label(15)),
        allocs_label: Some(label(16)),
    });
    for id in 1..=16 {
        world.push_identified(AssetId(id), TextLabel::default());
    }
    world
}

fn chip(world: &World, id: u32) -> String {
    world
        .get_by_id::<TextLabel>(AssetId(id))
        .map(|l| l.content.clone())
        .unwrap_or_default()
}

fn frame_time(world: &mut World, dt: f32) {
    world.insert_resource(FrameTime { dt, elapsed: 0.0 });
}

#[test]
fn fps_text_averages_frames_over_window() {
    assert_eq!(fps_text(60, 1.0).to_string(), "FPS 60");
    assert_eq!(fps_text(75, 0.5).to_string(), "FPS 150");
    assert_eq!(fps_text(0, 0.0).to_string(), "FPS 0");
}

#[test]
fn time_chips_read_in_milliseconds() {
    assert_eq!(ms_text("CPU", 4_250).to_string(), "CPU 4.2 ms");
    assert_eq!(ms_text("GPU", 400).to_string(), "GPU 0.4 ms");
    assert_eq!(worst_frame_text(0.0234).to_string(), "MAX 23.4 ms");
}

#[test]
fn alloc_text_averages_the_window_and_blanks_without_a_count() {
    let window = AllocWindow { sum: 30, max: 25 };
    assert_eq!(alloc_text(Some(window), 10).to_string(), "ALLOC 3/f max 25");
    assert_eq!(alloc_text(None, 10).to_string(), "");
}

// The test binary counts its heap, so a closed window fills the allocation
// chip from the frames' own totals.
#[test]
fn the_alloc_chip_fills_from_the_counted_heap() {
    let mut world = hud_world();
    start(&mut world);
    frame_time(&mut world, 1.0);
    world.step();
    world.step();
    let text = chip(&world, 16);
    assert!(
        text.starts_with("ALLOC ") && text.contains("/f max "),
        "{text}"
    );
}

#[test]
fn vram_text_reports_whole_megabytes() {
    assert_eq!(vram_text(512 * 1024 * 1024).to_string(), "VRAM 512 MB");
    assert_eq!(vram_text(1024 * 1024 + 1).to_string(), "VRAM 1 MB");
}

#[test]
fn ram_text_reports_rss_against_the_budget_when_known() {
    let mib = 1024 * 1024;
    assert_eq!(
        ram_text(Some(512 * mib), Some(16384)).to_string(),
        "RAM 512 / 16384 MB"
    );
    assert_eq!(ram_text(Some(256 * mib), None).to_string(), "RAM 256 MB");
    assert_eq!(ram_text(None, Some(16384)).to_string(), "");
}

#[test]
fn stream_text_counts_pending_loads_and_blanks_without_streaming() {
    assert_eq!(stream_text(Some(12)).to_string(), "STREAM 12 pending");
    assert_eq!(stream_text(None).to_string(), "");
}

#[test]
fn ev_text_keeps_its_sign_and_blanks_when_off_or_invalid() {
    assert_eq!(ev_text(Some(1.25)).to_string(), "EV +1.25");
    assert_eq!(ev_text(Some(-0.5)).to_string(), "EV -0.50");
    assert_eq!(ev_text(None).to_string(), "");
    assert_eq!(ev_text(Some(f32::NAN)).to_string(), "");
}

#[test]
fn edr_text_blanks_on_sdr_and_invalid_values() {
    assert_eq!(edr_text(Some(2.0)).to_string(), "EDR x2.0");
    assert_eq!(edr_text(None).to_string(), "");
    assert_eq!(edr_text(Some(0.0)).to_string(), "");
    assert_eq!(edr_text(Some(f32::INFINITY)).to_string(), "");
}

#[test]
fn phase_text_names_the_phase_and_its_progress() {
    assert_eq!(
        phase_text(Some(("ground", 12.44, 60.0)), 0.0).to_string(),
        "PHASE ground 12.4/60.0 s"
    );
    assert_eq!(phase_text(None, 1.5).to_string(), "PHASE warmup 1.5 s");
}

#[test]
fn resolution_text_names_the_upscaler_only_when_one_runs() {
    let native = RenderResolution {
        render: [2560, 1440],
        output: [2560, 1440],
        upscaler: None,
    };
    assert_eq!(resolution_text(native).to_string(), "RES 2560x1440");
    let upscaled = RenderResolution {
        render: [1280, 720],
        output: [2560, 1440],
        upscaler: Some("MetalFX"),
    };
    assert_eq!(
        resolution_text(upscaled).to_string(),
        "RES 1280x720 -> 2560x1440 MetalFX"
    );
    assert_eq!(resolution_text(RenderResolution::default()).to_string(), "");
}

// Once the averaging window closes, the top-row chips fill; without a
// benchmark the second row stays blank.
#[test]
fn a_closed_window_fills_the_top_row_only_without_a_benchmark() {
    let mut world = hud_world();
    start(&mut world);
    frame_time(&mut world, 1.0);
    world.step();
    assert_eq!(chip(&world, 1), "FPS 1");
    assert_eq!(chip(&world, 2), "MAX 1000.0 ms");
    assert!(chip(&world, 3).starts_with("CPU "), "{}", chip(&world, 3));
    assert_eq!(chip(&world, 4), "GPU 0.0 ms");
    assert_eq!(chip(&world, 5), "VRAM 0 MB");
    for id in 10..=15 {
        assert_eq!(chip(&world, id), "", "chip {id}");
    }
}

// Frames shorter than the window leave the chips alone until it closes, and
// the timing chips average its frames while MAX keeps the longest.
#[test]
fn the_window_averages_its_frames_and_keeps_the_longest() {
    let mut world = hud_world();
    start(&mut world);
    for (dt, gpu) in [(0.25, 2_000), (0.3, 4_000)] {
        frame_time(&mut world, dt);
        world.context().profile.render.gpu_frame_us = gpu;
        world.step();
        if dt < 0.3 {
            assert_eq!(chip(&world, 1), "", "no emit before the window closes");
        }
    }
    assert_eq!(chip(&world, 4), "GPU 3.0 ms");
    assert_eq!(chip(&world, 2), "MAX 300.0 ms");
}

// The RAM chip fills from the host's resident-set reading, against the
// published memory budget.
#[test]
fn the_ram_chip_reads_the_resident_set_against_the_budget() {
    #[expect(
        clippy::unnecessary_wraps,
        reason = "a `ResidentMemory` probe, which returns `None` where unsupported"
    )]
    fn resident() -> Option<u64> {
        Some(300 * 1024 * 1024)
    }

    let mut world = hud_world();
    start(&mut world);
    let budget = MemoryBudget::compute(Some(16 * 1024 * 1024 * 1024), 0);
    world.insert_resource(budget);
    world.insert_resource(ResidentMemory(resident));
    frame_time(&mut world, 1.0);
    world.step();
    assert_eq!(
        chip(&world, 6),
        format!("RAM 300 / {} MB", budget.budget_mib())
    );
}

// The HudPrefs resource gates the performance chips; the rest stay.
#[test]
fn hud_prefs_hide_the_performance_chips() {
    let mut world = hud_world();
    start(&mut world);
    world.insert_resource(HudPrefs {
        show_fps: false,
        show_vram: false,
    });
    frame_time(&mut world, 1.0);
    world.step();
    for id in 1..=5 {
        assert_eq!(chip(&world, id), "", "chip {id} hidden");
    }
}

// Hiding blanks every chip at once; showing again fills them on the same
// step instead of waiting out the window.
#[test]
fn toggling_the_hud_blanks_and_refills_every_chip() {
    let mut world = hud_world();
    start(&mut world);
    frame_time(&mut world, 1.0);
    world.step();
    assert_eq!(chip(&world, 1), "FPS 1");

    world.insert_resource(HudHidden(true));
    frame_time(&mut world, 0.1);
    world.step();
    assert_eq!(chip(&world, 1), "");
    assert_eq!(chip(&world, 5), "");

    world.insert_resource(HudHidden(false));
    world.step();
    assert_eq!(chip(&world, 1), "FPS 10");
}

// A benchmarked run fills the second row: the time and phase every frame from
// the track's clock, the rest on the window.
#[test]
fn a_benchmarked_run_fills_the_second_row() {
    use crate::components::cook::CameraTrack as CameraTrackArgs;
    use crate::components::{BenchmarkConfig, BenchmarkPhase, CameraTrack};

    let mut world = hud_world();
    world.add_component(CameraTrack::bake(CameraTrackArgs::default()));
    let phase = |name: &str, seconds| BenchmarkPhase {
        name: name.to_string(),
        seconds,
    };
    world.insert_resource(BenchmarkRun::new(BenchmarkConfig {
        warmup_seconds: 4.0,
        phases: vec![phase("settle", 4.0), phase("ground", 60.0)],
        ..Default::default()
    }));
    start(&mut world);
    frame_time(&mut world, 0.1);
    world.step();
    assert_eq!(chip(&world, 10), "", "no track status yet");

    world.insert_resource(CameraTrackStatus {
        elapsed_seconds: 2.5,
        ..Default::default()
    });
    world.step();
    assert_eq!(chip(&world, 10), "TIME 2.5 s");
    assert_eq!(chip(&world, 11), "PHASE warmup 1.5 s");

    world.insert_resource(CameraTrackStatus {
        elapsed_seconds: 20.04,
        ..Default::default()
    });
    world.context().profile.render.objects = 1234;
    world.context().profile.render.draw_calls = 56;
    frame_time(&mut world, 1.0);
    world.step();
    assert_eq!(chip(&world, 10), "TIME 20.0 s");
    assert_eq!(chip(&world, 11), "PHASE ground 12.0/60.0 s");
    assert_eq!(chip(&world, 14), "OBJECTS 1234");
    assert_eq!(chip(&world, 15), "DRAWS 56");
}
