// What a hit region does: its rectangle, the hover styling of its label, and
// the action it fires.

use crate::components::{
    Binding, HitRegion, SceneCommand, ScreenCommand, SettingCommand, SettingOp, StoryCommand,
    TextLabel, UiAction, UiEvent, UiState, UiValue, ValueOp, VariableCommand,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, by_asset_id};
use crate::gfx::overlay::UI_REFERENCE_SIZE;
use crate::settings::SettingKey;

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
// engine-level result (e.g. Quit), None otherwise. `source` is the firing
// region or binding, named in an emitted event.
pub(super) fn fire_action(
    action: &UiAction,
    source: Option<AssetId>,
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
        UiAction::Value { binding, op } => apply_value(binding, op, ctx),
        UiAction::Emit(emit) => ctx.events_mut::<UiEvent>().send(UiEvent {
            name: emit.name.clone(),
            element: source,
            value: emit.value.clone(),
        }),
        // Rebind captures and dropdown opens are driven by their own passes,
        // not the click-to-fire path.
        UiAction::Setting { .. } => {}
    }
    None
}

// Change the value `binding` names: a setting through the settings system, a
// world variable through the behavior system, a UI state value in place.
pub(crate) fn apply_value(binding: &Binding, op: &ValueOp, ctx: &mut PipelineContext<'_>) {
    match binding {
        Binding::Setting(key) => {
            for op in setting_ops(*key, op) {
                ctx.events_mut::<SettingCommand>().send(SettingCommand {
                    setting: *key,
                    op,
                    persist: true,
                });
            }
        }
        Binding::Var(name) => ctx.events_mut::<VariableCommand>().send(VariableCommand {
            name: name.clone(),
            op: op.clone(),
        }),
        Binding::State(name) => {
            if ctx.resource::<UiState>().is_none() {
                ctx.insert_resource(UiState::default());
            }
            if let Some(state) = ctx.resource_mut::<UiState>() {
                if let Some(value) = op.apply(state.get(name)) {
                    state.set(name, value);
                } else {
                    tracing::warn!("{op:?} does not apply to state:{name}");
                }
            }
        }
    }
}

// The most option steps one action sends for a setting whose option count is
// only known at runtime.
const MAX_SETTING_STEPS: u32 = 16;

// The settings commands an op on `key` is: a step is one command per option
// stepped, the shorter way round a known option list, a flip the next option,
// a whole number an option index, a number a slider fraction.
fn setting_ops(key: SettingKey, op: &ValueOp) -> impl Iterator<Item = SettingOp> {
    let (one, count) = match op {
        ValueOp::Step(by) => step_ops(key, *by),
        ValueOp::Flip => (Some(SettingOp::Next), 1),
        ValueOp::Set(UiValue::Bool(b)) => (Some(SettingOp::SetIndex(usize::from(*b))), 1),
        ValueOp::Set(UiValue::Int(i)) => {
            if let Ok(index) = usize::try_from(*i) {
                (Some(SettingOp::SetIndex(index)), 1)
            } else {
                tracing::warn!("a setting has no option {i}");
                (None, 0)
            }
        }
        ValueOp::Set(UiValue::Float(f)) => (Some(SettingOp::SetFraction(f.clamp(0.0, 1.0))), 1),
        ValueOp::Set(UiValue::Text(_)) | ValueOp::Add(_) => {
            tracing::warn!("{op:?} does not apply to a setting");
            (None, 0)
        }
    };
    one.into_iter()
        .flat_map(move |op| core::iter::repeat_n(op, count))
}

// A step of `by` options on `key`: around a known option list the shorter way,
// otherwise at most `MAX_SETTING_STEPS`.
fn step_ops(key: SettingKey, by: i32) -> (Option<SettingOp>, usize) {
    let len = crate::settings::options(key).map_or(0, <[_]>::len);
    let (dir, n) = if len > 0 {
        let len = i64::try_from(len).unwrap_or(i64::MAX);
        let ahead = i64::from(by).rem_euclid(len);
        // Halfway round either way goes the way `by` points.
        if ahead * 2 > len || (ahead * 2 == len && by < 0) {
            (SettingOp::Prev, len - ahead)
        } else {
            (SettingOp::Next, ahead)
        }
    } else {
        let dir = if by < 0 {
            SettingOp::Prev
        } else {
            SettingOp::Next
        };
        (dir, i64::from(by.unsigned_abs().min(MAX_SETTING_STEPS)))
    };
    (Some(dir), usize::try_from(n).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps(key: SettingKey, by: i32) -> (Option<SettingOp>, usize) {
        step_ops(key, by)
    }

    #[test]
    fn a_step_goes_the_shorter_way_round_a_known_option_list() {
        let len = crate::settings::options(SettingKey::Vsync).unwrap().len();
        assert_eq!(steps(SettingKey::Vsync, 1), (Some(SettingOp::Next), 1));
        let whole_turn = i32::try_from(len).unwrap();
        assert_eq!(
            steps(SettingKey::Vsync, whole_turn),
            (Some(SettingOp::Next), 0)
        );
        assert!(steps(SettingKey::Vsync, i32::MAX).1 <= len);
        assert!(steps(SettingKey::Vsync, i32::MIN).1 <= len);
    }

    #[test]
    fn a_step_without_an_option_list_is_capped() {
        assert_eq!(
            steps(SettingKey::Resolution, -2_000_000_000),
            (Some(SettingOp::Prev), MAX_SETTING_STEPS as usize)
        );
    }
}
