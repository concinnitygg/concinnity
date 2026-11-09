// Each test seeds the components UiInputSystem drains at init (HitRegion /
// Screen / KeyBinding) before `start`, which runs the system on its own.
use super::capture::REBIND_PROMPT;
use super::*;
use crate::components::{
    GamepadAction, GamepadButton, HitRegion, InputKey, KeyEvent, KeyMods, KeyPress, ScrollGroup,
    ScrollRow, TextAlign, TextLabel,
};
use crate::components::{NavDirection, ScrollPanel, SettingOp};
use crate::ecs::{Phase, World};
use crate::gfx::dropdown;
use crate::gfx::overlay::UI_REFERENCE_SIZE;
use crate::input::keymap::Bindable;
use crate::window::display_mode;
use alloc::string::ToString;
use alloc::vec;

fn start(world: &mut World) {
    crate::test_support::start_with(
        world,
        Phase::PreRender,
        "UiInputSystem",
        UiInputSystem::new(),
    );
}

// A test action from its text form, with integer targets.
fn act(action: serde_json::Value) -> UiAction {
    serde_json::from_value(action).unwrap()
}

fn make_frame_input(mx: f32, my: f32, clicked: bool) -> FrameInput {
    FrameInput {
        mouse_x: mx,
        mouse_y: my,
        left_click: clicked,
        ..Default::default()
    }
}

// The ScreenCommand UiInputSystem sent this step, read with a fresh cursor so
// the system's own cursor (which applies them a frame later) is untouched.
// Returns the first if several were sent.
fn produced_screen_command(world: &World) -> Option<ScreenCommand> {
    let mut cursor = EventCursor::default();
    world
        .events::<ScreenCommand>()
        .and_then(|e| e.read(&mut cursor).next().copied())
}

// Every SettingCommand the system sent, read with a fresh cursor (in send
// order). The renderer applies these, but these tests run UiInputSystem
// alone, so they inspect the queue directly via .first()/.last()/.is_empty().
fn produced_setting_commands(world: &World) -> Vec<SettingCommand> {
    let mut cursor = EventCursor::default();
    world
        .events::<SettingCommand>()
        .map(|e| e.read(&mut cursor).cloned().collect())
        .unwrap_or_default()
}

// A screen-owned TextLabel used as a scroll-panel element.
fn panel_label(y: f32, screen: AssetId, content: &str) -> TextLabel {
    TextLabel {
        font: None,
        content: content.to_string(),
        x: 0.0,
        y,
        color: [1.0, 1.0, 1.0],
        color_runs: Vec::new(),
        scale: 1.0,
        centered: false,
        align: TextAlign::Left,
        fit: SpriteFit::Fit,
        background: [0.0, 0.0, 0.0, 0.0],
        padding: 0.0,
        visible: true,
        screen: Some(Ref::new(screen)),
        wrap_width: 0.0,
        max_lines: 0,
    }
}

fn label_field<T>(world: &World, id: AssetId, f: impl Fn(&TextLabel) -> T) -> T {
    world.get_by_id::<TextLabel>(id).map(f).unwrap()
}

mod actions;
mod dropdown_list;
mod focus;
mod key_bindings;
mod panels;
mod rebind;
mod regions;
mod screens;
