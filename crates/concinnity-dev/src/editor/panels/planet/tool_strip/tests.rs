use super::*;
use concinnity_engine::components::UiState;

use crate::editor::planet_tools::Brush;

fn event(name: &str, value: Option<UiValue>) -> UiEvent {
    UiEvent {
        name: name.into(),
        element: None,
        value,
    }
}

#[test]
fn every_tool_button_picks_its_tool() {
    for (i, tool) in Tool::ALL.into_iter().enumerate() {
        let value = Some(UiValue::Int(i32::try_from(i).unwrap()));
        assert_eq!(action(&event(TOOL, value)), Some(StripAction::Toggle(tool)));
    }
    assert_eq!(action(&event(TOOL, Some(UiValue::Int(99)))), None);
    assert_eq!(action(&event(PICK, None)), Some(StripAction::Cycle));
}

#[test]
fn each_slider_moves_its_setting() {
    for (s, setting) in Setting::ALL.into_iter().enumerate() {
        assert_eq!(
            action(&event(POSITION[s], Some(UiValue::Float(0.25)))),
            Some(StripAction::Slide(setting, 0.25))
        );
    }
    assert_eq!(action(&event(POSITION[0], None)), None);
}

// The picked tool's button shows pressed, and each setting's slider and
// value follow the brush.
#[test]
fn sync_marks_the_picked_tool_and_writes_the_brush() {
    let mut world = World::new();
    let ids = StripIds {
        brush: AssetId(1),
        pick: AssetId(2),
    };
    let mut tools = Tools {
        brush: Brush {
            radius: 8.0,
            strength: 0.5,
            falloff: 0.25,
        },
        ..Tools::default()
    };
    tools.toggle(Tool::Lower);
    sync(
        &StripView {
            tools: &tools,
            pick: "",
        },
        ids,
        &mut world,
    );
    let state = world.resource::<UiState>().unwrap();
    assert_eq!(state.get(PICKED[1]), Some(&UiValue::Bool(true)));
    assert_eq!(state.get(PICKED[0]), Some(&UiValue::Bool(false)));
    assert_eq!(state.get(VALUE[1]), Some(&UiValue::Text("50%".into())));
    assert_eq!(
        state.get(POSITION[0]),
        Some(&UiValue::Float(tools.brush.position(Setting::Radius)))
    );
    assert_eq!(state.get(CAPTION), Some(&UiValue::Text("Lower (J)".into())));
}
