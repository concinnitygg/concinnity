// src/editor/health_panel.rs
//
// The Health panel's layout: one meter per resource, each a track scaled to
// capacity with two nested fills over it -- the dim one is what the process
// uses, the bright one inside it is what the engine's own accounting explains.
// Under them, one line per tag something reports holding. The model (and what
// those numbers mean) is `health.rs`; this module only places elements.
//
// The panel grows with the breakdown: a world that streams nothing has no rows
// and no space reserved for them, so it never shows an empty region. Every
// caller that needs the footprint (the drag clamp, the hit test) takes the
// snapshot and asks for the size that goes with it.
//
// Read-only: the panel has no controls beyond its close button, so a body press
// is swallowed rather than resolved to an action.

use super::health::{self, HealthSnapshot, Meter, TagRow};
use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Health);

// One meter per resource: RAM, VRAM, CPU.
const ROWS: usize = 3;

const PANEL_BG: AssetId = AssetId(BASE);
const TITLE_LABEL: AssetId = AssetId(BASE + 1);
const CLOSE_BG: AssetId = AssetId(BASE + 2);
const CLOSE_LABEL: AssetId = AssetId(BASE + 3);
const CHURN_LABEL: AssetId = AssetId(BASE + 4);
const HOT_CLASS_LABEL: AssetId = AssetId(BASE + 5);

// Per-row families, one slot per row.
fn track(i: usize) -> AssetId {
    AssetId(BASE + 0x10 + i as u32)
}
fn used_fill(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}
fn tracked_fill(i: usize) -> AssetId {
    AssetId(BASE + 0x30 + i as u32)
}
fn caption_label(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}
fn value_label(i: usize) -> AssetId {
    AssetId(BASE + 0x50 + i as u32)
}
fn parts_label(i: usize) -> AssetId {
    AssetId(BASE + 0x60 + i as u32)
}

// Breakdown families. Wider stride than the meters': there is one slot per tag
// per realm, which is more than a 0x10 family holds.
fn tag_name_label(i: usize) -> AssetId {
    AssetId(BASE + 0x80 + i as u32)
}
fn tag_realm_label(i: usize) -> AssetId {
    AssetId(BASE + 0xA0 + i as u32)
}
fn tag_value_label(i: usize) -> AssetId {
    AssetId(BASE + 0xC0 + i as u32)
}

const PANEL_W: f32 = 340.0;
// Caption + value line, the bar, then the dim line naming the three numbers.
const ROW_H: f32 = 52.0;
const BAR_H: f32 = 10.0;
const PAD_X: f32 = 12.0;
const CHURN_H: f32 = 20.0;
const TAG_ROW_H: f32 = 18.0;
const BOTTOM_PAD: f32 = 6.0;

// Approximate body-font advance. Only the test that holds the breakdown's three
// columns clear of each other needs it; the layout itself places by edge.
#[cfg(test)]
const CHAR_W: f32 = 8.5;
// Left edge of the dim realm column: just past the longest tag name, leaving
// the whole right side of the row to the value. `the_breakdown_columns_cannot_
// overlap` holds both gaps.
const REALM_X: f32 = PAD_X + 78.0;

// Vertical anchors inside a row.
const CAPTION_CY: f32 = 12.0;
const BAR_TOP: f32 = 22.0;
const PARTS_CY: f32 = 42.0;

// The recessed track, and the two fills over it. The bright tracked fill draws
// last so it reads as sitting inside the dim used fill.
const TRACK_TINT: [f32; 4] = [0.06, 0.06, 0.08, 1.0];
const USED_TINT: [f32; 4] = [0.28, 0.31, 0.40, 1.0];
const TRACKED_TINT: [f32; 4] = [0.36, 0.62, 0.92, 1.0];
// A row close to capacity is the whole point of a health readout, so its used
// fill turns warning-red rather than staying neutral.
const USED_WARN_TINT: [f32; 4] = [0.62, 0.30, 0.28, 1.0];
const WARN_FRAC: f32 = 0.9;
// The same warning, for a breakdown row past its budget.
const OVER_BUDGET_LABEL: [f32; 3] = [0.90, 0.52, 0.48];

// A fill this thin is still drawn, so a small-but-real value is visible rather
// than rounding away to an empty track.
const MIN_FILL_W: f32 = 2.0;

// The panel's footprint for what `snap` has to show: the meters and the churn
// line always, plus a line per reported tag and the size-class line when the
// allocator measured one.
pub(crate) fn size(snap: &HealthSnapshot) -> [f32; 2] {
    let lines = health::tag_rows(snap).len() + usize::from(health::hot_class_text(snap).is_some());
    [
        PANEL_W,
        widget::TITLE_H + ROWS as f32 * ROW_H + CHURN_H + lines as f32 * TAG_ROW_H + BOTTOM_PAD,
    ]
}

// Top-right, clear of the Assets panel's left anchor and below the top bar.
pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    [(vp[0] - PANEL_W - 8.0).max(0.0), super::hud::body_top()]
}

pub(crate) fn panel_rect(o: [f32; 2], snap: &HealthSnapshot) -> [f32; 4] {
    widget::outer_rect(o, size(snap))
}

fn row_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    [
        o[0],
        o[1] + widget::TITLE_H + i as f32 * ROW_H,
        PANEL_W,
        ROW_H,
    ]
}

fn bar_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    let r = row_rect(o, i);
    [r[0] + PAD_X, r[1] + BAR_TOP, PANEL_W - 2.0 * PAD_X, BAR_H]
}

// A body press anywhere on the panel is swallowed so it cannot reach the world
// behind it. `None` misses the panel entirely and falls through.
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2], snap: &HealthSnapshot) -> bool {
    point_in(mx, my, panel_rect(o, snap))
}

// Position + show the panel at origin `o`.
pub(crate) fn apply(world: &mut World, snap: &HealthSnapshot, o: [f32; 2], mouse: [f32; 2]) {
    widget::place_panel(world, PANEL_BG, panel_rect(o, snap));
    let title = widget::title_rect(o, PANEL_W);
    widget::place_heading(world, TITLE_LABEL, title, "Health");
    let close_hover = point_in(mouse[0], mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    for (i, meter) in health::meters(snap).iter().enumerate() {
        place_row(world, o, i, meter);
    }

    let mut y = o[1] + widget::TITLE_H + ROWS as f32 * ROW_H;
    match health::churn_text(snap) {
        Some(text) => {
            place_text(
                world,
                CHURN_LABEL,
                [o[0] + PAD_X, y + CHURN_H * 0.5],
                TextAlign::Left,
                theme::LABEL_DIM,
                &text,
            );
        }
        None => widget::set_label_visible(world, CHURN_LABEL, false),
    }
    y += CHURN_H;

    match health::hot_class_text(snap) {
        Some(text) => {
            place_text(
                world,
                HOT_CLASS_LABEL,
                [o[0] + PAD_X, y + TAG_ROW_H * 0.5],
                TextAlign::Left,
                theme::LABEL_DIM,
                &text,
            );
            y += TAG_ROW_H;
        }
        None => widget::set_label_visible(world, HOT_CLASS_LABEL, false),
    }

    let rows = health::tag_rows(snap);
    for (i, row) in rows.iter().enumerate() {
        place_tag_row(world, o, y + i as f32 * TAG_ROW_H, i, row);
    }
    for i in rows.len()..health::MAX_TAG_ROWS {
        for id in [tag_name_label(i), tag_realm_label(i), tag_value_label(i)] {
            widget::set_label_visible(world, id, false);
        }
    }
}

// One breakdown line: what holds the memory, which realm it is in, and how much
// -- against its budget when it has one. A tag past its budget reads in the
// same warning colour a meter near capacity does.
fn place_tag_row(world: &mut World, o: [f32; 2], y: f32, i: usize, row: &TagRow) {
    let cy = y + TAG_ROW_H * 0.5;
    place_text(
        world,
        tag_name_label(i),
        [o[0] + PAD_X, cy],
        TextAlign::Left,
        theme::LABEL,
        row.name,
    );
    place_text(
        world,
        tag_realm_label(i),
        [o[0] + REALM_X, cy],
        TextAlign::Left,
        theme::LABEL_DIM,
        row.realm,
    );
    let color = if row.over_budget {
        OVER_BUDGET_LABEL
    } else {
        theme::LABEL
    };
    place_text(
        world,
        tag_value_label(i),
        [o[0] + PANEL_W - PAD_X, cy],
        TextAlign::Right,
        color,
        &row.value,
    );
}

fn place_row(world: &mut World, o: [f32; 2], i: usize, meter: &Meter) {
    let r = row_rect(o, i);
    place_text(
        world,
        caption_label(i),
        [r[0] + PAD_X, r[1] + CAPTION_CY],
        TextAlign::Left,
        theme::LABEL,
        meter.caption,
    );
    place_text(
        world,
        value_label(i),
        [r[0] + PANEL_W - PAD_X, r[1] + CAPTION_CY],
        TextAlign::Right,
        theme::LABEL,
        &meter.value_text(),
    );
    place_text(
        world,
        parts_label(i),
        [r[0] + PAD_X, r[1] + PARTS_CY],
        TextAlign::Left,
        theme::LABEL_DIM,
        meter.parts,
    );

    let bar = bar_rect(o, i);
    widget::place_rounded(world, track(i), bar, TRACK_TINT, BAR_H * 0.5, true);
    let used = meter.used_frac();
    let used_tint = if used >= WARN_FRAC {
        USED_WARN_TINT
    } else {
        USED_TINT
    };
    place_fill(world, used_fill(i), bar, used, used_tint);
    place_fill(
        world,
        tracked_fill(i),
        bar,
        meter.tracked_frac(),
        TRACKED_TINT,
    );
}

// A proportional fill from the left of `bar`. An unknown or empty quantity hides
// the sprite rather than drawing a zero-width pill.
fn place_fill(world: &mut World, id: AssetId, bar: [f32; 4], frac: f32, tint: [f32; 4]) {
    if frac <= 0.0 {
        widget::set_sprite_visible(world, id, false);
        return;
    }
    let w = (bar[2] * frac).max(MIN_FILL_W);
    // Keep the pill from rounding wider than it is, which would bulge a thin fill.
    let radius = (BAR_H * 0.5).min(w * 0.5);
    widget::place_rounded(world, id, [bar[0], bar[1], w, BAR_H], tint, radius, true);
}

// Place a label centered vertically on `at` ([x, center_y]).
fn place_text(
    world: &mut World,
    id: AssetId,
    at: [f32; 2],
    align: TextAlign,
    color: [f32; 3],
    text: &str,
) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = at[0];
        l.y = at[1] - theme::TEXT_HALF;
        l.align = align;
        l.color = color;
        l.visible = true;
        l.content = text.to_string();
    }
}

pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
}

// Draw order: the panel surface, then each row's track, its used fill, and the
// tracked fill nested on top.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG];
    for i in 0..ROWS {
        ids.push(track(i));
        ids.push(used_fill(i));
        ids.push(tracked_fill(i));
    }
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, CLOSE_LABEL, CHURN_LABEL, HOT_CLASS_LABEL];
    for i in 0..ROWS {
        ids.push(caption_label(i));
        ids.push(value_label(i));
        ids.push(parts_label(i));
    }
    // Every tag can report in either realm, so every line it could take has a
    // slot injected up front; unused ones stay hidden.
    for i in 0..health::MAX_TAG_ROWS {
        ids.push(tag_name_label(i));
        ids.push(tag_realm_label(i));
        ids.push(tag_value_label(i));
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};
    use concinnity_memory::{Ledger, MemStats, MemTag, Realm};

    const GB: u64 = 1024 * 1024 * 1024;

    fn injected_world() -> World {
        let mut world = World::new_empty();
        for id in all_sprite_ids() {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in all_label_ids() {
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

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
        ledger.set_budget(MemTag::Textures, Realm::Device, Some(4 * GB));
        ledger.set(MemTag::Meshes, Realm::Device, GB / 2);
        ledger.set(MemTag::Scratch, Realm::Host, 1024);
        HealthSnapshot {
            tags: ledger.snapshot(),
            ..snapshot()
        }
    }

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .expect("sprite is injected")
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .expect("label is injected")
    }

    #[test]
    fn apply_shows_the_heading_and_every_row() {
        let mut world = injected_world();
        apply(&mut world, &snapshot(), [0.0, 0.0], [0.0, 0.0]);
        assert_eq!(label(&world, TITLE_LABEL).content, "Health");
        for (i, caption) in ["RAM", "VRAM", "CPU"].iter().enumerate() {
            assert_eq!(&label(&world, caption_label(i)).content, caption);
            assert!(label(&world, value_label(i)).visible);
            assert!(sprite(&world, track(i)).visible);
        }
    }

    // The fills are proportional to the track, and the tracked fill nests inside
    // the used fill when the engine's accounting explains part of the usage.
    #[test]
    fn fills_are_proportional_and_nested() {
        let mut world = injected_world();
        let o = [0.0, 0.0];
        apply(&mut world, &snapshot(), o, [0.0, 0.0]);

        let bar = bar_rect(o, 0);
        let used = sprite(&world, used_fill(0));
        let tracked = sprite(&world, tracked_fill(0));
        // RSS 4 of 32 GB, heap 1 of 32 GB.
        assert!((used.width - bar[2] * 4.0 / 32.0).abs() < 0.01);
        assert!((tracked.width - bar[2] / 32.0).abs() < 0.01);
        assert!(
            tracked.width < used.width,
            "tracked should nest inside used"
        );
        assert_eq!(used.x, bar[0], "both fills start at the track's left edge");
        assert_eq!(tracked.x, bar[0]);
    }

    // An unknown quantity draws nothing at all, so an empty track reads as "no
    // data" rather than "zero bytes".
    #[test]
    fn unknown_quantities_hide_their_fills() {
        let mut world = injected_world();
        apply(
            &mut world,
            &HealthSnapshot::default(),
            [0.0, 0.0],
            [0.0, 0.0],
        );
        for i in 0..ROWS {
            assert!(!sprite(&world, used_fill(i)).visible);
            assert!(!sprite(&world, tracked_fill(i)).visible);
            assert!(sprite(&world, track(i)).visible, "the track still shows");
        }
    }

    #[test]
    fn a_row_near_capacity_turns_the_used_fill_red() {
        let mut world = injected_world();
        let mut snap = snapshot();
        snap.rss = Some(31 * GB);
        apply(&mut world, &snap, [0.0, 0.0], [0.0, 0.0]);
        assert_eq!(sprite(&world, used_fill(0)).tint, USED_WARN_TINT);

        snap.rss = Some(4 * GB);
        apply(&mut world, &snap, [0.0, 0.0], [0.0, 0.0]);
        assert_eq!(sprite(&world, used_fill(0)).tint, USED_TINT);
    }

    // A tiny-but-real value stays visible instead of rounding to an empty track.
    #[test]
    fn a_tiny_fill_stays_visible() {
        let mut world = injected_world();
        let snap = HealthSnapshot {
            rss: Some(1024),
            total_ram: Some(32 * GB),
            ..Default::default()
        };
        apply(&mut world, &snap, [0.0, 0.0], [0.0, 0.0]);
        let used = sprite(&world, used_fill(0));
        assert!(used.visible);
        assert!(used.width >= MIN_FILL_W);
        // The pill must not round wider than the fill itself.
        assert!(used.corner_radius <= used.width * 0.5);
    }

    // Without the tracking allocator there is no heap line to draw.
    #[test]
    fn the_churn_line_hides_without_heap_stats() {
        let mut world = injected_world();
        apply(&mut world, &snapshot(), [0.0, 0.0], [0.0, 0.0]);
        assert!(label(&world, CHURN_LABEL).visible);

        let mut snap = snapshot();
        snap.heap = None;
        apply(&mut world, &snap, [0.0, 0.0], [0.0, 0.0]);
        assert!(!label(&world, CHURN_LABEL).visible);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        apply(&mut world, &snapshot(), [0.0, 0.0], [0.0, 0.0]);
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }

    #[test]
    fn hit_test_swallows_panel_presses_and_lets_misses_through() {
        let o = [100.0, 50.0];
        let snap = snapshot();
        assert!(hit_test(
            o[0] + 10.0,
            o[1] + widget::TITLE_H + 10.0,
            o,
            &snap
        ));
        assert!(!hit_test(o[0] - 5.0, o[1], o, &snap));
        assert!(!hit_test(o[0] + PANEL_W + 1.0, o[1] + 10.0, o, &snap));
    }

    // Every row sits inside the panel footprint the drag clamp is given.
    #[test]
    fn rows_fit_within_the_panel_footprint() {
        let o = [0.0, 0.0];
        let snap = snapshot();
        let panel = panel_rect(o, &snap);
        for i in 0..ROWS {
            let bar = bar_rect(o, i);
            assert!(bar[1] + bar[3] <= panel[1] + panel[3]);
            assert!(bar[0] + bar[2] <= panel[0] + panel[2]);
        }
    }

    // One line per reported tag, naming what holds the memory and where.
    #[test]
    fn the_breakdown_draws_a_line_per_reported_tag() {
        let mut world = injected_world();
        apply(&mut world, &tagged_snapshot(), [0.0, 0.0], [0.0, 0.0]);

        let names = ["Scratch", "Textures", "Meshes"];
        let realms = ["RAM", "VRAM", "VRAM"];
        for (i, (name, realm)) in names.iter().zip(realms).enumerate() {
            assert_eq!(&label(&world, tag_name_label(i)).content, name);
            assert_eq!(label(&world, tag_realm_label(i)).content, realm);
            assert!(label(&world, tag_value_label(i)).visible);
        }
        assert_eq!(label(&world, tag_value_label(1)).content, "2.0 GB / 4.0 GB");
    }

    // Rows are stacked, not overdrawn, and every one sits inside the footprint
    // the drag clamp is given.
    #[test]
    fn the_breakdown_stacks_inside_the_panel_footprint() {
        let mut world = injected_world();
        let o = [0.0, 0.0];
        let snap = tagged_snapshot();
        apply(&mut world, &snap, o, [0.0, 0.0]);

        let panel = panel_rect(o, &snap);
        let mut last_y = f32::MIN;
        for i in 0..health::tag_rows(&snap).len() {
            let y = label(&world, tag_name_label(i)).y;
            assert!(y > last_y, "row {i} does not follow the one above it");
            assert!(y + TAG_ROW_H <= panel[1] + panel[3]);
            last_y = y;
        }
    }

    // A world nothing reports into draws no breakdown at all, and the panel is
    // shorter by exactly the rows it does not draw.
    #[test]
    fn an_unreported_ledger_draws_no_rows_and_shortens_the_panel() {
        let mut world = injected_world();
        apply(&mut world, &tagged_snapshot(), [0.0, 0.0], [0.0, 0.0]);
        assert!(label(&world, tag_name_label(0)).visible);

        apply(&mut world, &snapshot(), [0.0, 0.0], [0.0, 0.0]);
        for i in 0..health::MAX_TAG_ROWS {
            assert!(!label(&world, tag_name_label(i)).visible);
            assert!(!label(&world, tag_realm_label(i)).visible);
            assert!(!label(&world, tag_value_label(i)).visible);
        }
        assert_eq!(
            size(&snapshot())[1] + 3.0 * TAG_ROW_H,
            size(&tagged_snapshot())[1]
        );
    }

    // The three columns must clear each other at their widest: the longest tag
    // name in the vocabulary, the longest realm, and a value carrying two
    // fully-rendered byte figures. Text that collided would be unreadable and
    // no test that only checks placement would notice.
    #[test]
    fn the_breakdown_columns_cannot_overlap() {
        let longest_name = MemTag::ALL
            .into_iter()
            .map(|t| t.name().chars().count())
            .max()
            .expect("the vocabulary is not empty");
        let name_end = PAD_X + longest_name as f32 * CHAR_W;
        assert!(
            name_end <= REALM_X,
            "tag names {longest_name} chars wide run into the realm column"
        );

        let longest_realm = Realm::ALL
            .into_iter()
            .map(|r| r.name().chars().count())
            .max()
            .expect("there are two realms");
        let realm_end = REALM_X + longest_realm as f32 * CHAR_W;
        // Both figures at their longest rendering, budget included.
        let widest_value = "1023.9 GB / 1023.9 GB";
        let value_start = PANEL_W - PAD_X - widest_value.len() as f32 * CHAR_W;
        assert!(
            realm_end <= value_start,
            "a full-width value would run into the realm column"
        );
    }

    // A tag past its budget reads in the warning colour, like a meter near
    // capacity.
    #[test]
    fn a_row_past_its_budget_is_coloured_as_a_warning() {
        let ledger = Ledger::new();
        ledger.set(MemTag::Textures, Realm::Device, 8 * GB);
        ledger.set_budget(MemTag::Textures, Realm::Device, Some(4 * GB));
        let snap = HealthSnapshot {
            tags: ledger.snapshot(),
            ..snapshot()
        };

        let mut world = injected_world();
        apply(&mut world, &snap, [0.0, 0.0], [0.0, 0.0]);
        assert_eq!(label(&world, tag_value_label(0)).color, OVER_BUDGET_LABEL);
        assert_eq!(label(&world, tag_name_label(0)).color, theme::LABEL);
    }

    // The size-class line is a development instrument: no line at all in a
    // build without it, and the breakdown moves up to fill the space.
    #[test]
    fn the_hot_class_line_appears_only_when_the_allocator_measured_one() {
        let mut world = injected_world();
        let snap = tagged_snapshot();
        apply(&mut world, &snap, [0.0, 0.0], [0.0, 0.0]);
        assert!(!label(&world, HOT_CLASS_LABEL).visible);
        let without = label(&world, tag_name_label(0)).y;

        let measured = HealthSnapshot {
            hot_class: Some(concinnity_memory::SizeClass {
                min_bytes: 64,
                max_bytes: 127,
                allocs: 10,
                live_blocks: 4,
            }),
            ..snap
        };
        apply(&mut world, &measured, [0.0, 0.0], [0.0, 0.0]);
        assert!(label(&world, HOT_CLASS_LABEL).visible);
        assert_eq!(label(&world, tag_name_label(0)).y - without, TAG_ROW_H);
        assert_eq!(size(&measured)[1] - size(&snap)[1], TAG_ROW_H);
    }
}
