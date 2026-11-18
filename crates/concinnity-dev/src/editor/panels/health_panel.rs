//! The Health panel: one meter per resource, each a track scaled to capacity
//! with two nested fills over it -- the dim one is what the process uses, the
//! bright one inside it is what the engine's own accounting explains. Under
//! them, one line per tag something reports holding. The model (and what those
//! numbers mean) is `health.rs`; this module builds the panel and shows a
//! snapshot in it, only when a new one is taken.
//!
//! The panel grows with the breakdown: a world that streams nothing has no rows
//! and no space reserved for them, so it never shows an empty region.

use std::sync::LazyLock;

use concinnity_engine::components::{
    LayoutChild, LayoutDirection, LayoutSize, Sprite, TextAlign, TextLabel, UiState, UiValue,
};
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{Ref, World};

use super::health::{self, HealthSnapshot};
use super::registry::PanelKey;
use crate::editor::theme;
use crate::editor::ui::build::{BODY_PAD, fill};
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

// One meter per resource: RAM, VRAM, CPU.
const ROWS: usize = 3;

const PANEL_W: f32 = 340.0;
// The bar spans the body.
const BAR_W: f32 = PANEL_W - 2.0 * BODY_PAD;
const BAR_H: f32 = 10.0;
// A meter's caption line, its bar, and the dim line naming its numbers.
const LINE_H: f32 = 20.0;
const METER_H: f32 = LINE_H + BAR_H + LINE_H;
const TAG_ROW_H: f32 = 18.0;
// The width of a breakdown line's tag name, clear of its realm.
const TAG_NAME_W: f32 = 78.0;

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

// The state the panel shows: the snapshot it shows, each meter's numbers, the
// churn, drift, and size-class lines, and each breakdown line's three texts.
const SHOWN: &str = "editor.health.shown";
const VALUES: [&str; ROWS] = [
    "editor.health.value.0",
    "editor.health.value.1",
    "editor.health.value.2",
];
const CHURN: &str = "editor.health.churn";
const DRIFT: &str = "editor.health.drift";
const HOT_CLASS: &str = "editor.health.hot_class";
static TAGS: LazyLock<Vec<[String; 3]>> = LazyLock::new(|| {
    (0..health::MAX_TAG_ROWS)
        .map(|i| ["name", "realm", "value"].map(|part| format!("editor.health.tag.{i}.{part}")))
        .collect()
});

// Where each changing element sits in the panel's slots: every meter's used
// and tracked fill, then every breakdown line's value.
const fn used_slot(i: usize) -> usize {
    2 * i
}
const fn tracked_slot(i: usize) -> usize {
    2 * i + 1
}
const fn tag_value_slot(i: usize) -> usize {
    2 * ROWS + i
}

// The panel's footprint for what `snap` has to show: the meters and the churn
// line always, plus a line per reported tag and the size-class and drift lines
// when there are any.
pub(crate) fn size(snap: &HealthSnapshot) -> [f32; 2] {
    let lines = health::tag_rows(snap).len()
        + usize::from(health::drift_text(snap).is_some())
        + usize::from(health::hot_class_text(snap).is_some());
    let content = ROWS as f32 * METER_H + LINE_H + lines as f32 * TAG_ROW_H;
    [PANEL_W, panel_height(content)]
}

// Top-right, clear of the Assets panel's left anchor and below the top bar.
pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    [
        (vp[0] - PANEL_W - 8.0).max(0.0),
        crate::editor::hud::body_top(),
    ]
}

// A sprite in a bar: the track fills it, a fill takes its own width.
fn bar_sprite(ui: &mut UiBuilder<'_>, tint: [f32; 4], width: f32) -> Option<AssetId> {
    ui.sprite(Sprite {
        width,
        height: BAR_H,
        tint,
        corner_radius: BAR_H * 0.5,
        ..Default::default()
    })
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let mut body = Vec::new();
    let mut slots = Vec::new();
    let meters = health::meters(&HealthSnapshot::default());
    for (i, meter) in meters.iter().enumerate() {
        let caption = ui.label(meter.caption, theme::LABEL)?;
        let value = ui.bound_label(VALUES[i], theme::LABEL, TextAlign::Right)?;
        body.push(ui.row(vec![caption, value], LINE_H)?);
        let track = bar_sprite(ui, TRACK_TINT, 0.0)?;
        let used = bar_sprite(ui, USED_TINT, 0.0)?;
        let tracked = bar_sprite(ui, TRACKED_TINT, 0.0)?;
        slots.extend([used, tracked]);
        let bar = ui.stack(
            LayoutDirection::Stack,
            vec![fill(track), bar_fill(used), bar_fill(tracked)],
            0.0,
            0.0,
        )?;
        body.push(LayoutChild {
            item: Some(Ref::new(bar)),
            width: LayoutSize::Fill(1.0),
            height: LayoutSize::Fixed(BAR_H),
        });
        let parts = ui.label(meter.parts, theme::LABEL_DIM)?;
        body.push(ui.row(vec![parts], LINE_H)?);
    }
    for name in [CHURN, DRIFT, HOT_CLASS] {
        let text = ui.bound_label(name, theme::LABEL_DIM, TextAlign::Left)?;
        let height = if name == CHURN { LINE_H } else { TAG_ROW_H };
        body.push(ui.row(vec![text], height)?);
    }
    for [name, realm, value] in TAGS.iter() {
        let name = ui.bound_label(name, theme::LABEL, TextAlign::Left)?;
        let name = LayoutChild {
            width: LayoutSize::Fixed(TAG_NAME_W),
            ..name
        };
        let realm = ui.bound_label(realm, theme::LABEL_DIM, TextAlign::Left)?;
        let value = ui.bound_label(value, theme::LABEL, TextAlign::Right)?;
        slots.push(value.item?.id());
        body.push(ui.row(vec![name, realm, value], TAG_ROW_H)?);
    }
    let spec = PanelSpec {
        key: PanelKey::Health,
        title: "Health",
        title_state: None,
        size: size(&HealthSnapshot::default()),
        min_size: None,
    };
    let mut panel = ui.panel(&spec, body)?;
    panel.slots = slots;
    Some(panel)
}

// A bar fill, as wide as the width it is given.
fn bar_fill(id: AssetId) -> LayoutChild {
    LayoutChild {
        item: Some(Ref::new(id)),
        width: LayoutSize::Fit,
        height: LayoutSize::Fixed(BAR_H),
    }
}

// Show snapshot number `samples`, unless it is the one shown already.
pub(crate) fn sync(snap: &HealthSnapshot, samples: u64, world: &mut World, built: &BuiltPanel) {
    let shown = i32::try_from(samples % (1 << 30)).unwrap_or(0);
    let current = world
        .resource::<UiState>()
        .and_then(|s| s.get(SHOWN).cloned());
    if current == Some(UiValue::Int(shown)) {
        return;
    }
    for (i, meter) in health::meters(snap).iter().enumerate() {
        state::set_text(world, VALUES[i], format_args!("{}", meter.value_text()));
        let used = meter.used_frac();
        let used_tint = if used >= WARN_FRAC {
            USED_WARN_TINT
        } else {
            USED_TINT
        };
        set_fill(world, built.slots.get(used_slot(i)), used, used_tint);
        let tracked = meter.tracked_frac();
        set_fill(
            world,
            built.slots.get(tracked_slot(i)),
            tracked,
            TRACKED_TINT,
        );
    }
    let text = |t: Option<String>| t.unwrap_or_default();
    state::set_text(
        world,
        CHURN,
        format_args!("{}", text(health::churn_text(snap))),
    );
    state::set_text(
        world,
        DRIFT,
        format_args!("{}", text(health::drift_text(snap))),
    );
    let hot = text(health::hot_class_text(snap));
    state::set_text(world, HOT_CLASS, format_args!("{hot}"));
    let rows = health::tag_rows(snap);
    for (i, [name, realm, value]) in TAGS.iter().enumerate() {
        let row = rows.get(i);
        state::set_text(world, name, format_args!("{}", row.map_or("", |r| r.name)));
        state::set_text(
            world,
            realm,
            format_args!("{}", row.map_or("", |r| r.realm)),
        );
        state::set_text(
            world,
            value,
            format_args!("{}", row.map_or("", |r| &r.value)),
        );
        let color = if row.is_some_and(|r| r.over_budget) {
            OVER_BUDGET_LABEL
        } else {
            theme::LABEL
        };
        if let Some(&id) = built.slots.get(tag_value_slot(i))
            && let Some(label) = world.get_mut_by_id::<TextLabel>(id)
        {
            label.color = color;
        }
    }
    state::with_state(world, |s| s.set(SHOWN, UiValue::Int(shown)));
}

// A proportional fill from the left of its bar. An unknown or empty quantity
// hides the sprite rather than drawing a zero-width pill.
fn set_fill(world: &mut World, id: Option<&AssetId>, frac: f32, tint: [f32; 4]) {
    let Some(sprite) = id.and_then(|&id| world.get_mut_by_id::<Sprite>(id)) else {
        return;
    };
    sprite.visible = frac > 0.0;
    let w = (BAR_W * frac).max(MIN_FILL_W);
    sprite.width = w;
    // Keep the pill from rounding wider than it is, which would bulge a thin fill.
    sprite.corner_radius = (BAR_H * 0.5).min(w * 0.5);
    sprite.tint = tint;
}

#[cfg(test)]
mod tests;
