// What a hit region does: its rectangle, the hover styling of its label, and
// the action it fires.

use crate::components::{
    HitRegion, SceneCommand, ScreenCommand, SettingCommand, SettingOp, SettingVerb, StoryCommand,
    TextLabel, UiAction,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, by_asset_id};
use crate::gfx::overlay::UI_REFERENCE_SIZE;

// Whether a point lies inside an `[x, y, width, height]` rectangle.
pub(super) fn point_in_rect(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

// A region's `[x, y, width, height]` rectangle.
pub(super) fn region_rect(r: &HitRegion) -> [f32; 4] {
    [r.x, r.y, r.width, r.height]
}

// Whether a region spans the whole reference canvas (a full-screen "click
// anywhere" region). Such a region covers the entire live window, including any
// letterbox margin, rather than only the fitted canvas rect.
pub(super) fn region_covers_canvas(r: &HitRegion) -> bool {
    r.x <= 0.0
        && r.y <= 0.0
        && r.x + r.width >= UI_REFERENCE_SIZE[0]
        && r.y + r.height >= UI_REFERENCE_SIZE[1]
}

// Write the given color + scale onto a region's referenced label, if any.
// Drives hover-in (hover style), hover-out (captured style), and the restore
// applied when a hovered region goes inert (its screen hides, its row collapses,
// or it is disabled) so its hover styling never strands on the label.
pub(super) fn set_label_style(
    ctx: &mut PipelineContext<'_>,
    label: Option<AssetId>,
    color: Option<[f32; 3]>,
    scale: Option<f32>,
) {
    by_asset_id::update::<TextLabel>(ctx, label, |lbl| {
        if let Some(c) = color {
            lbl.color = c;
        }
        if let Some(s) = scale {
            lbl.scale = s;
        }
    });
}

// Execute an action. Returns Some(StepResult) when the action produces an
// engine-level result (e.g. Quit), None otherwise. `label` is the firing
// region's referenced TextLabel (the value display for a settings row),
// forwarded so the renderer can update it.
pub(super) fn fire_action(
    action: &UiAction,
    label: Option<AssetId>,
    ctx: &mut PipelineContext<'_>,
) -> Option<StepResult> {
    match action {
        UiAction::Quit => return Some(StepResult::Stop),
        UiAction::Scene(scene) => {
            ctx.events_mut::<SceneCommand>().send(SceneCommand {
                scene: *scene,
                transition: crate::components::SceneTransition::FadeBlack,
            });
            // A scene change dismisses every open screen: the user has chosen a
            // new context, so the whole overlay stack clears.
            ctx.events_mut::<ScreenCommand>().send(ScreenCommand::Clear);
        }
        UiAction::Screen(cmd) => ctx.events_mut::<ScreenCommand>().send(*cmd),
        UiAction::Story(cmd) => ctx.events_mut::<StoryCommand>().send(cmd.clone()),
        UiAction::Screenshot(frames) => {
            ctx.insert_resource(crate::protocol::ScreenshotRequest(*frames));
        }
        UiAction::ToggleHud => {
            let hidden = ctx
                .resource::<crate::protocol::HudHidden>()
                .is_some_and(|h| h.0);
            ctx.insert_resource(crate::protocol::HudHidden(!hidden));
        }
        UiAction::Setting {
            key,
            verb: verb @ (SettingVerb::Next | SettingVerb::Prev),
        } => ctx.events_mut::<SettingCommand>().send(SettingCommand {
            setting: *key,
            op: if *verb == SettingVerb::Prev {
                SettingOp::Prev
            } else {
                SettingOp::Next
            },
            value_label: label,
            // A cycle is one discrete change: always persisted.
            persist: true,
        }),
        // Slider drags, rebind captures, dropdown opens, and group toggles are
        // driven by their own passes, not the click-to-fire path.
        UiAction::Setting {
            verb: SettingVerb::Drag | SettingVerb::Rebind | SettingVerb::Open,
            ..
        }
        | UiAction::GroupToggle(_) => {}
    }
    None
}
