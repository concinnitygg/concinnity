// Slider-track drags: a press over a track begins a drag, the held button
// tracks the cursor, and the release commits the final value.

use concinnity_core::components::{FrameInput, SettingCommand, SettingOp};
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::gfx::overlay::OverlayTransform;

use super::{UiInputSystem, point_in_rect, region_rect};

// The slider value at reference-space `qx` along a `[x, y, width, height]`
// track, clamped to `0..=1` so a drag past either end pins the value.
fn track_fraction(qx: f32, rect: [f32; 4]) -> f32 {
    ((qx - rect[0]) / rect[2]).clamp(0.0, 1.0)
}

impl UiInputSystem {
    // Drive the active screen's slider tracks for one frame. The dragged region
    // is remembered so the drag continues after the cursor leaves the track;
    // live updates skip the disk write, which only the release persists.
    pub(super) fn step_slider_drag(
        &mut self,
        input: &FrameInput,
        active_screen: Option<AssetId>,
        overlay: &OverlayTransform,
        ctx: &mut PipelineContext,
    ) {
        // Slider tracks are overlay UI: map the cursor to reference space.
        let (qx, qy) = overlay.inverse(input.mouse_x, input.mouse_y);
        if !input.left_button_down {
            if let Some(i) = self.dragging.take()
                && self.regions[i].screen == active_screen
                && let Some(key) = self.regions[i].slider_key.clone()
            {
                let r = &self.regions[i].region;
                ctx.events_mut::<SettingCommand>().send(SettingCommand {
                    setting: key,
                    op: SettingOp::SetFraction(track_fraction(qx, region_rect(r))),
                    value_label: r.label,
                    persist: true,
                });
            }
            return;
        }
        for (i, entry) in self.regions.iter().enumerate() {
            if entry.screen != active_screen {
                continue;
            }
            let Some(key) = entry.slider_key.as_ref() else {
                continue;
            };
            let rect = region_rect(&entry.region);
            if self.dragging.is_none() && input.left_click && point_in_rect(qx, qy, rect) {
                self.dragging = Some(i);
            }
            if self.dragging == Some(i) {
                ctx.events_mut::<SettingCommand>().send(SettingCommand {
                    setting: key.clone(),
                    op: SettingOp::SetFraction(track_fraction(qx, rect)),
                    value_label: entry.region.label,
                    persist: false,
                });
            }
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
