// A track dragged to set a number, with the number beside it.

use alloc::string::String;

use crate::components::{Binding, Screen, SpriteFit};
use crate::ecs::Ref;

use super::WidgetStyle;

/// A track and handle dragged to set the number `value` names, between `min`
/// and `max`, with the number written at its right end.
///
/// A press on the track moves the handle there and drags it until the button
/// lets go, setting the number as it moves, snapped to `step` when that is
/// above zero. Each change sends a `UiEvent` named by `on_change` carrying the
/// new number, and letting go sends `on_commit` with the number it settled
/// on. The fill runs from zero, or from the end of the range nearer zero, to
/// the handle. A disabled slider keeps its value and draws dimmed.
///
/// ```rust
/// # use concinnity_engine::components::{Binding, ValueSlider};
/// ValueSlider {
///     value: Some(Binding::State("radius".into())),
///     min: 1.0,
///     max: 50.0,
///     step: 0.5,
///     on_commit: "radius_set".into(),
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct ValueSlider {
    /// The number the slider shows and sets: a `state:` or `var:` binding.
    pub value: Option<Binding>,
    /// The number at the track's left end.
    pub min: f32,
    /// The number at the track's right end.
    #[asset(default = 1.0)]
    pub max: f32,
    /// The increment the number snaps to; 0 sets it freely.
    pub step: f32,
    /// Digits after the point the number is written with.
    #[asset(default = 2)]
    pub decimals: u8,
    /// Width of the column the number is written in at the right end; 0
    /// writes no number.
    #[asset(default = 44.0)]
    pub value_width: f32,
    /// Whether the slider takes no input and draws dimmed.
    pub disabled: bool,
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width, the number's column included.
    #[asset(default = 200.0)]
    pub width: f32,
    /// Height.
    #[asset(default = 24.0)]
    pub height: f32,
    /// The event each change sends while dragging.
    pub on_change: String,
    /// The event letting go sends.
    pub on_commit: String,
    /// How the slider draws.
    pub style: WidgetStyle,
    /// When false the slider is not drawn and takes no input.
    #[asset(default = true)]
    pub visible: bool,
    /// How a screen-owned slider maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) the slider belongs to. Unset, it joins the world's
    /// HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime: whether the cursor is over the slider. Not authored.
    #[serde(skip)]
    pub hovered: bool,
}

// The handle's size.
pub(crate) const HANDLE_W: f32 = 10.0;
pub(crate) const HANDLE_H: f32 = 16.0;
const TRACK_H: f32 = 4.0;

impl ValueSlider {
    // Where a press grabs the slider: its track and the handle's reach past
    // each end, the slider's height tall, short of the number's column.
    pub(crate) fn grab_rect(&self) -> [f32; 4] {
        let track = self.track();
        [
            track[0] - HANDLE_W * 0.5,
            self.y,
            track[2] + HANDLE_W,
            self.height,
        ]
    }

    // The track, short of the number's column and of half a handle at each
    // end, so the handle never overhangs.
    pub(crate) fn track(&self) -> [f32; 4] {
        let w = (self.width - self.value_width.max(0.0) - HANDLE_W).max(1.0);
        [
            self.x + HANDLE_W * 0.5,
            self.y + (self.height - TRACK_H) * 0.5,
            w,
            TRACK_H,
        ]
    }

    fn span(&self) -> f32 {
        self.max - self.min
    }

    // Where `value` sits along the track, 0 to 1.
    pub(crate) fn fraction(&self, value: f32) -> f32 {
        if self.span().abs() <= f32::EPSILON {
            return 0.0;
        }
        ((value - self.min) / self.span()).clamp(0.0, 1.0)
    }

    // The number at `x` along the track, snapped to the step.
    pub(crate) fn value_at(&self, x: f32) -> f32 {
        let track = self.track();
        let t = ((x - track[0]) / track[2]).clamp(0.0, 1.0);
        self.snapped(self.min + t * self.span())
    }

    // `value` snapped to the step, inside the range.
    pub(crate) fn snapped(&self, value: f32) -> f32 {
        let (lo, hi) = if self.min <= self.max {
            (self.min, self.max)
        } else {
            (self.max, self.min)
        };
        let v = if self.step > 0.0 {
            self.min + concinnity_core::math::round((value - self.min) / self.step) * self.step
        } else {
            value
        };
        v.clamp(lo, hi)
    }

    // Where the fill starts along the track: at zero inside the range, else
    // at the end nearer it.
    pub(crate) fn origin(&self) -> f32 {
        self.fraction(0.0)
    }

    // The handle's center for `value`.
    pub(crate) fn handle_x(&self, value: f32) -> f32 {
        let track = self.track();
        track[0] + self.fraction(value) * track[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slider() -> ValueSlider {
        ValueSlider {
            min: -10.0,
            max: 10.0,
            step: 0.5,
            width: 144.0,
            value_width: 44.0,
            ..Default::default()
        }
    }

    #[test]
    fn the_track_maps_to_the_range_snapped_to_the_step() {
        let s = slider();
        let track = s.track();
        assert_eq!(track[0], 5.0);
        assert_eq!(track[2], 90.0, "the number's column and half handles");
        assert_eq!(s.value_at(track[0]), -10.0);
        assert_eq!(s.value_at(track[0] + track[2]), 10.0);
        assert_eq!(s.value_at(track[0] + 47.0), 0.5, "snapped from 0.44");
        assert_eq!(s.value_at(-100.0), -10.0, "clamped");
        assert_eq!(s.handle_x(0.0), track[0] + 45.0);
    }

    #[test]
    fn the_fill_starts_at_zero_or_the_end_nearer_it() {
        assert_eq!(slider().origin(), 0.5);
        let positive = ValueSlider {
            min: 2.0,
            max: 4.0,
            ..slider()
        };
        assert_eq!(positive.origin(), 0.0);
        let flat = ValueSlider {
            min: 1.0,
            max: 1.0,
            ..slider()
        };
        assert_eq!(flat.fraction(5.0), 0.0, "an empty range has no span");
    }
}
