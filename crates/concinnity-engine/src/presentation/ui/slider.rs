// Slider drags: a press over a slider region begins a drag, the held button
// tracks the cursor, and the release commits the final value.

use crate::components::{Binding, FrameInput, SettingCommand, SettingOp, UiValue, ValueOp};
use crate::ecs::PipelineContext;
use crate::gfx::overlay::OverlayTransform;

use super::region_ops::apply_value;
use super::regions::setting_row_disabled;
use super::{UiInputSystem, point_in_rect, region_rect};

// The slider value at reference-space `qx` along a `[x, y, width, height]`
// track, clamped to `0..=1` so a drag past either end pins the value.
fn track_fraction(qx: f32, rect: [f32; 4]) -> f32 {
    ((qx - rect[0]) / rect[2]).clamp(0.0, 1.0)
}

// Set the slider's bound value to `fraction`. A setting persists only when
// `persist` is set, so a drag writes the store once, on release.
fn set_fraction(ctx: &mut PipelineContext<'_>, binding: &Binding, fraction: f32, persist: bool) {
    if let Binding::Setting(key) = binding {
        ctx.events_mut::<SettingCommand>().send(SettingCommand {
            setting: *key,
            op: SettingOp::SetFraction(fraction),
            persist,
        });
    } else {
        apply_value(binding, &ValueOp::Set(UiValue::Float(fraction)), ctx);
    }
}

impl UiInputSystem {
    // Drive the active screen's slider regions for one frame. A press starts a
    // drag on a shown, enabled slider; the dragged region is remembered so the
    // drag continues after the cursor leaves the track; the value is set only
    // when it moves, and the release commits it.
    pub(super) fn step_slider_drag(
        &mut self,
        input: &FrameInput,
        overlay: &OverlayTransform,
        ctx: &mut PipelineContext<'_>,
    ) {
        let cursor = |screen| {
            self.screens
                .cursor(screen, overlay, input.mouse_x, input.mouse_y)
        };
        if !input.left_button_down {
            if let Some((i, _)) = self.dragging.take()
                && self.screens.receives_input(self.regions[i].screen)
                && let Some(binding) = &self.regions[i].region.slider
            {
                let (qx, _) = cursor(self.regions[i].screen);
                let fraction = track_fraction(qx, region_rect(&self.regions[i].region));
                set_fraction(ctx, binding, fraction, true);
            }
            return;
        }
        if self.dragging.is_none() && input.left_click {
            self.dragging = self.regions.iter().enumerate().find_map(|(i, entry)| {
                let (qx, qy) = cursor(entry.screen);
                (entry.region.slider.is_some()
                    && self.screens.receives_input(entry.screen)
                    && !(entry.scroll_row.is_some() && entry.hidden)
                    && !setting_row_disabled(&self.disabled_rows_cache, &entry.region)
                    && point_in_rect(qx, qy, region_rect(&entry.region))
                    && self.widgets.stack.reaches_owned(
                        entry.screen,
                        entry.region.label.and_then(|l| ctx.entity_of(l.id())),
                        cursor,
                    ))
                .then_some((i, None))
            });
        }
        let Some((i, last)) = self.dragging else {
            return;
        };
        let entry = &self.regions[i];
        let Some(binding) = &entry.region.slider else {
            return;
        };
        if !self.screens.receives_input(entry.screen) {
            return;
        }
        let (qx, _) = cursor(entry.screen);
        let fraction = track_fraction(qx, region_rect(&entry.region));
        if last != Some(fraction) {
            set_fraction(ctx, binding, fraction, false);
            self.dragging = Some((i, Some(fraction)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRACK: [f32; 4] = [100.0, 50.0, 200.0, 20.0];

    #[test]
    fn below_the_track_clamps_to_zero() {
        assert_eq!(track_fraction(40.0, TRACK), 0.0);
    }

    #[test]
    fn inside_the_track_is_proportional() {
        assert_eq!(track_fraction(100.0, TRACK), 0.0);
        assert_eq!(track_fraction(150.0, TRACK), 0.25);
        assert_eq!(track_fraction(300.0, TRACK), 1.0);
    }

    #[test]
    fn past_the_track_clamps_to_one() {
        assert_eq!(track_fraction(900.0, TRACK), 1.0);
    }
}
