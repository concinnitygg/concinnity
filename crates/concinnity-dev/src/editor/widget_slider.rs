// src/editor/widget_slider.rs
//
// A drag slider for the editor HUD: a track, a fill from the track's neutral
// point to the handle, the handle, and a value label. Plain `Sprite` /
// `TextLabel` components at reserved ids, placed each frame by the owning
// panel; the hook owns the drag (press on the track, follow the cursor,
// release). The settings-menu `Slider` asset is a different thing.

use super::theme;
use super::widget::{self, place_rounded};
use crate::components::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const TRACK_H: f32 = 4.0;
const HANDLE_W: f32 = 10.0;
const HANDLE_H: f32 = 16.0;
// The value label's column at the right end of the slider rect.
pub(crate) const VALUE_W: f32 = 44.0;

const TRACK_TINT: [f32; 4] = [0.26, 0.27, 0.33, 1.0];
const FILL_TINT: [f32; 4] = theme::ACCENT_TINT;
const HANDLE_TINT: [f32; 4] = [0.82, 0.84, 0.90, 1.0];
const HANDLE_TINT_HOT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

// The reserved ids one slider draws with.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SliderIds {
    pub track: AssetId,
    pub fill: AssetId,
    pub handle: AssetId,
    pub value: AssetId,
}

// The track's rect inside a slider `rect` ([x, y, w, h]): the value column is
// left free at the right, and half a handle at each end so the handle never
// overhangs.
pub(crate) fn track_rect(rect: [f32; 4]) -> [f32; 4] {
    let x = rect[0] + HANDLE_W * 0.5;
    let w = (rect[2] - VALUE_W - HANDLE_W).max(1.0);
    [x, rect[1] + (rect[3] - TRACK_H) * 0.5, w, TRACK_H]
}

// Map `value` in `range` to the handle's centre x on the track.
pub(crate) fn handle_x(rect: [f32; 4], value: f32, range: (f32, f32)) -> f32 {
    let t = track_rect(rect);
    let span = (range.1 - range.0).max(1e-6);
    let f = ((value - range.0) / span).clamp(0.0, 1.0);
    t[0] + t[2] * f
}

// Map a cursor x to a value in `range`, clamped to the track.
pub(crate) fn value_at(rect: [f32; 4], mx: f32, range: (f32, f32)) -> f32 {
    let t = track_rect(rect);
    let f = ((mx - t[0]) / t[2].max(1e-6)).clamp(0.0, 1.0);
    range.0 + (range.1 - range.0) * f
}

// Whether a press at `(mx, my)` lands on the slider (track or handle, the
// full row height, value column excluded).
pub(crate) fn hit(rect: [f32; 4], mx: f32, my: f32) -> bool {
    widget::point_in(
        mx,
        my,
        [rect[0], rect[1], (rect[2] - VALUE_W).max(0.0), rect[3]],
    )
}

// Draw the slider at `rect` showing `value`. `hot` is a hovered or dragged
// handle.
pub(crate) fn place(
    world: &mut World,
    ids: SliderIds,
    rect: [f32; 4],
    value: f32,
    range: (f32, f32),
    hot: bool,
) {
    let t = track_rect(rect);
    place_rounded(world, ids.track, t, TRACK_TINT, TRACK_H * 0.5, true);
    // The fill runs from the neutral point (0, or the range start when 0 is
    // outside it) to the handle, so a bipolar slider fills out from centre.
    let zero = handle_x(rect, 0.0_f32.clamp(range.0, range.1), range);
    let hx = handle_x(rect, value, range);
    let (fx, fw) = if hx >= zero {
        (zero, hx - zero)
    } else {
        (hx, zero - hx)
    };
    place_rounded(
        world,
        ids.fill,
        [fx, t[1], fw, t[3]],
        FILL_TINT,
        TRACK_H * 0.5,
        fw > 0.5,
    );
    let tint = if hot { HANDLE_TINT_HOT } else { HANDLE_TINT };
    place_rounded(
        world,
        ids.handle,
        [
            hx - HANDLE_W * 0.5,
            rect[1] + (rect[3] - HANDLE_H) * 0.5,
            HANDLE_W,
            HANDLE_H,
        ],
        tint,
        3.0,
        true,
    );
    if let Some(l) = widget::label_mut(world, ids.value) {
        l.x = rect[0] + rect[2];
        l.y = rect[1] + rect[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Right;
        l.color = theme::LABEL_DIM;
        l.visible = true;
        l.content = format_value(value);
    }
}

pub(crate) fn hide(world: &mut World, ids: SliderIds) {
    widget::set_sprite_visible(world, ids.track, false);
    widget::set_sprite_visible(world, ids.fill, false);
    widget::set_sprite_visible(world, ids.handle, false);
    widget::set_label_visible(world, ids.value, false);
}

// Two decimals, with an explicit sign on positive values so a bipolar slider
// reads its direction at a glance.
pub(crate) fn format_value(v: f32) -> String {
    if v > 0.0 {
        format!("+{v:.2}")
    } else {
        format!("{v:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: [f32; 4] = [100.0, 50.0, 244.0, 24.0];

    #[test]
    fn value_and_handle_round_trip_across_the_track() {
        let t = track_rect(RECT);
        assert_eq!(t[0], 105.0);
        assert_eq!(t[2], 190.0);
        let bipolar = (-1.0, 1.0);
        assert!((handle_x(RECT, 0.0, bipolar) - 200.0).abs() < 1e-4);
        assert!((value_at(RECT, 200.0, bipolar)).abs() < 1e-4);
        assert!((value_at(RECT, 105.0, bipolar) + 1.0).abs() < 1e-4);
        assert!((value_at(RECT, 295.0, bipolar) - 1.0).abs() < 1e-4);
        // Past either end clamps.
        assert_eq!(value_at(RECT, 0.0, bipolar), -1.0);
        assert_eq!(value_at(RECT, 1000.0, (0.0, 1.0)), 1.0);
        let x = handle_x(RECT, 0.37, bipolar);
        assert!((value_at(RECT, x, bipolar) - 0.37).abs() < 1e-4);
    }

    #[test]
    fn hit_excludes_the_value_column() {
        assert!(hit(RECT, 150.0, 60.0));
        assert!(!hit(RECT, 330.0, 60.0), "the value label is not grabbable");
        assert!(!hit(RECT, 150.0, 10.0));
    }

    #[test]
    fn values_format_with_a_sign() {
        assert_eq!(format_value(0.5), "+0.50");
        assert_eq!(format_value(-0.25), "-0.25");
        assert_eq!(format_value(0.0), "0.00");
    }
}
