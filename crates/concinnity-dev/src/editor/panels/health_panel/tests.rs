use super::*;
use concinnity_core::memory::{Ledger, MemStats, MemTag, Realm};
use concinnity_engine::components::{UiState, UiValue};
use std::num::NonZeroU64;

const GB: u64 = 1024 * 1024 * 1024;

fn snapshot() -> HealthSnapshot {
    HealthSnapshot {
        heap: Some(MemStats {
            live_bytes: GB,
            peak_bytes: 2 * GB,
            alloc_count: 100,
            free_count: 90,
        }),
        rss: Some(4 * GB),
        total_ram: Some(32 * GB),
        pool_bytes: Some(GB),
        vram_bytes: Some(2 * GB),
        vram_budget: Some(8 * GB),
        systems_cores: Some(0.5),
        process_cores: Some(2.0),
        total_cores: Some(10),
        ..Default::default()
    }
}

// A snapshot whose ledger two pools and an arena have reported into.
fn tagged_snapshot() -> HealthSnapshot {
    let ledger = Ledger::new();
    ledger.set(MemTag::Textures, Realm::Device, 2 * GB);
    ledger.set_budget(MemTag::Textures, Realm::Device, NonZeroU64::new(4 * GB));
    ledger.set(MemTag::Meshes, Realm::Device, GB / 2);
    ledger.set(MemTag::Scratch, Realm::Host, 1024);
    HealthSnapshot {
        tags: ledger.snapshot(),
        ..snapshot()
    }
}

fn built(world: &mut World) -> BuiltPanel {
    let mut ui = UiBuilder::new(world, AssetId(9), None);
    build(&mut ui).unwrap()
}

fn sprite(world: &World, panel: &BuiltPanel, slot: usize) -> Sprite {
    world
        .get_by_id::<Sprite>(panel.slots[slot])
        .unwrap()
        .clone()
}

fn text(world: &World, name: &str) -> String {
    match world.resource::<UiState>().and_then(|s| s.get(name)) {
        Some(UiValue::Text(t)) => t.clone(),
        other => panic!("{name} holds {other:?}"),
    }
}

#[test]
fn fills_are_proportional_and_nested() {
    let mut world = World::new();
    let panel = built(&mut world);
    sync(&snapshot(), 1, &mut world, &panel);
    let used = sprite(&world, &panel, used_slot(0));
    let tracked = sprite(&world, &panel, tracked_slot(0));
    assert!(used.visible && tracked.visible);
    assert!((used.width - BAR_W * 4.0 / 32.0).abs() < 0.01);
    assert!(tracked.width < used.width, "the tracked fill sits inside");
    assert!(text(&world, VALUES[0]).contains('/'));
}

#[test]
fn unknown_quantities_hide_their_fills() {
    let mut world = World::new();
    let panel = built(&mut world);
    sync(&HealthSnapshot::default(), 1, &mut world, &panel);
    assert!(!sprite(&world, &panel, used_slot(0)).visible);
    assert!(!sprite(&world, &panel, tracked_slot(0)).visible);
}

#[test]
fn a_row_near_capacity_turns_the_used_fill_red() {
    let mut world = World::new();
    let panel = built(&mut world);
    let full = HealthSnapshot {
        rss: Some(31 * GB),
        ..snapshot()
    };
    sync(&full, 1, &mut world, &panel);
    assert_eq!(sprite(&world, &panel, used_slot(0)).tint, USED_WARN_TINT);
}

#[test]
fn a_tiny_fill_stays_visible() {
    let mut world = World::new();
    let panel = built(&mut world);
    let tiny = HealthSnapshot {
        heap: Some(MemStats {
            live_bytes: 1024,
            ..MemStats::default()
        }),
        ..snapshot()
    };
    sync(&tiny, 1, &mut world, &panel);
    assert!(sprite(&world, &panel, tracked_slot(0)).width >= MIN_FILL_W);
}

#[test]
fn the_breakdown_shows_a_line_per_reported_tag_and_warns_past_budget() {
    let mut world = World::new();
    let panel = built(&mut world);
    let snap = tagged_snapshot();
    sync(&snap, 1, &mut world, &panel);
    let rows = health::tag_rows(&snap);
    assert_eq!(rows.len(), 3);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(text(&world, &TAGS[i][0]), row.name);
        assert_eq!(text(&world, &TAGS[i][2]), row.value);
    }
    assert_eq!(
        text(&world, &TAGS[3][0]),
        "",
        "an unused line shows nothing"
    );
}

#[test]
fn a_snapshot_is_shown_once() {
    let mut world = World::new();
    let panel = built(&mut world);
    sync(&snapshot(), 1, &mut world, &panel);
    let revision = world.resource::<UiState>().unwrap().revision();
    sync(&snapshot(), 1, &mut world, &panel);
    assert_eq!(world.resource::<UiState>().unwrap().revision(), revision);
}

#[test]
fn the_churn_line_is_empty_without_heap_stats() {
    let mut world = World::new();
    let panel = built(&mut world);
    let snap = HealthSnapshot {
        heap: None,
        ..snapshot()
    };
    sync(&snap, 1, &mut world, &panel);
    assert_eq!(text(&world, CHURN), "");
}
