// src/gfx/input_system.rs
//
// Consumes the frame's `InputMailbox` packet (sampled beside the backend
// right after the draw, whose event pump produced it), merges the active
// gamepad's state, and publishes the `FrameInput` snapshot (as a resource and
// as the component column) plus the `CursorState` resource. Runs immediately
// after GraphicsSystem in the schedule, so the packet it takes is the one
// that system just deposited -- the same freshness the sample had when this
// system polled the backend itself. Every input consumer (camera controllers,
// UI, text input) runs later in the same tick.
//
// The gamepad is polled here, not per render backend, so all backends share
// one implementation. Digital buttons merge into the existing boolean fields
// (d-pad onto the movement keys, the bound buttons onto sprint / jump /
// interact, Start onto escape); the sticks publish as the analog
// `move_axis` / `look_axis` fields.

use crate::assets::{FrameInput, GamepadButton, GamepadMap, NavDirection};
use crate::ecs::{InputMailbox, PipelineContext, StepResult, System};
use crate::input::gamepad::{GamepadSource, PadSnapshot, PadState};
use crate::input::nav::NavRepeat;
use concinnity_render::input::RenderInput;
use std::time::Instant;

// Frame dt ceiling for the nav auto-repeat, so a hitch or debugger pause never
// counts as a long hold.
const NAV_DT_MAX: f32 = 0.25;

#[derive(Debug, Default)]
pub struct InputSystem {
    gamepad: GamepadSource,
    pad: PadState,
    map: GamepadMap,
    deadzone: f32,
    // Auto-repeat state for the d-pad / stick navigation pulse.
    nav: NavRepeat,
    // Previous step's timestamp, for the nav repeat dt.
    last_step: Option<Instant>,
    // Cursor into the Events<ControlsCommand> queue (live settings changes).
    controls_cursor: crate::ecs::EventCursor,
}

impl InputSystem {
    pub fn new() -> Self {
        Self {
            deadzone: crate::gfx::settings::DEFAULT_GAMEPAD_DEADZONE,
            ..Self::default()
        }
    }
}

// Merge the window's keyboard/mouse snapshot with the gamepad snapshot under
// the gameplay gate. Pure, so the merge and gating are unit-tested without a
// backend; the caller fills in the viewport.
fn compose_frame_input(
    raw: &RenderInput,
    pad: &PadSnapshot,
    map: GamepadMap,
    nav: Option<NavDirection>,
    gameplay: bool,
) -> FrameInput {
    let axis = |v: [f32; 2]| if gameplay { v } else { [0.0, 0.0] };
    FrameInput {
        forward: (raw.forward || pad.held(GamepadButton::DpadUp)) && gameplay,
        backward: (raw.backward || pad.held(GamepadButton::DpadDown)) && gameplay,
        left: (raw.left || pad.held(GamepadButton::DpadLeft)) && gameplay,
        right: (raw.right || pad.held(GamepadButton::DpadRight)) && gameplay,
        sprint: (raw.sprint || pad.held(map.sprint)) && gameplay,
        interact: (raw.interact || pad.pressed(map.interact)) && gameplay,
        jump: (raw.jump || pad.pressed(map.jump)) && gameplay,
        move_axis: axis(pad.move_axis),
        look_axis: axis(pad.look_axis),
        mouse_dx: if gameplay { raw.mouse_dx } else { 0.0 },
        mouse_dy: if gameplay { raw.mouse_dy } else { 0.0 },
        // Not gated by `gameplay`: a scrollable menu still scrolls
        // while it is open (the camera is what freezes behind it).
        scroll_delta: raw.scroll_delta,
        mouse_x: raw.mouse_x,
        mouse_y: raw.mouse_y,
        left_click: raw.left_click,
        left_button_down: raw.left_button_down,
        // Not gated by `gameplay` either: like `left_click`, the press stays
        // false at the backend while the cursor is captured.
        right_click: raw.right_click,
        viewport: [0.0, 0.0],
        hud_toggle: raw.hud_toggle,
        // Start mirrors Escape (pause / back), so like `raw.escape` it stays
        // live while a menu is open.
        escape: raw.escape || pad.pressed(GamepadButton::Start),
        // Not gated by `gameplay`: a story's Ctrl fast-forward works
        // while its stage (a view) is up, like the rebind capture below.
        ctrl: raw.ctrl,
        // Not gated by `gameplay`: the editor's additive selection needs the
        // modifier while it holds the cursor. `raw.sprint` is the Shift key.
        shift: raw.sprint,
        // Not gated by `gameplay`: the editor's orbit drag needs the modifier
        // while it holds the cursor, like `shift`.
        alt: raw.alt,
        // Not gated by `gameplay` either: it carries the same UI shortcuts
        // Ctrl does on the platforms that use it.
        cmd: raw.cmd,
        // Not gated by `gameplay`: the rebind captures work while the
        // settings menu is open (the camera is what freezes behind it).
        captured_key: raw.captured_key,
        captured_button: pad.first_pressed(),
        // Not gated by `gameplay`: menu focus movement + confirm/back consume
        // these while a screen is active; during play UI ignores them and the
        // buttons keep their merged meanings above.
        nav,
        confirm: pad.pressed(GamepadButton::South),
        back: pad.pressed(GamepadButton::East),
        // Not gated by `gameplay`: text-input fields type while a menu
        // (or the in-engine editor) is up, like the rebind capture.
        typed_char: raw.typed_char,
    }
}

impl System for InputSystem {
    fn init(&mut self, _ctx: &mut PipelineContext) {
        // Persisted settings-menu choices override the engine defaults.
        let settings = crate::config::Settings::load();
        self.map = settings.controls.gamepad_map.unwrap_or_default();
        if let Some(dz) = settings.controls.gamepad_deadzone {
            self.deadzone = dz;
        }
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // No deposited packet (graphics failed, the editor transplanted the
        // backend away, or the frame stopped before sampling): nothing to
        // consume; consumers keep the last snapshot.
        let Some(packet) = ctx
            .resource_mut::<InputMailbox>()
            .and_then(|mailbox| mailbox.0.take())
        else {
            return StepResult::Continue;
        };
        let raw = packet.raw;
        let cursor_outside = packet.cursor_outside_window;
        // Live viewport for UiInputSystem's overlay hit-testing, so a scaled
        // menu's HitRegions map back to the cursor consistently.
        let (vp_w, vp_h) = packet.viewport;

        // The cursor position + window-bounds state for next frame's draw
        // list (`follow_cursor` sprites are positioned a frame after the input
        // that moved them; the draw list is built before this poll).
        ctx.insert_resource(crate::ecs::CursorState {
            pos: (raw.mouse_x, raw.mouse_y),
            outside_window: cursor_outside,
        });

        // Live controls changes (a gamepad rebind or deadzone slider) sent by
        // SettingsSystem earlier this tick.
        if let Some(events) = ctx.events::<crate::assets::ControlsCommand>() {
            for cmd in events.read(&mut self.controls_cursor) {
                if let Some(map) = cmd.gamepad_map {
                    self.map = map;
                }
                if let Some(dz) = cmd.gamepad_deadzone {
                    self.deadzone = dz;
                }
            }
        }
        self.gamepad.poll(&mut self.pad);
        let pad = self.pad.snapshot(self.deadzone);

        // Shape the held d-pad + stick state into this frame's navigation
        // pulse (auto-repeat needs a dt; a first frame or a hitch clamps).
        let now = Instant::now();
        let dt = self
            .last_step
            .map(|t| (now - t).as_secs_f32().min(NAV_DT_MAX))
            .unwrap_or(0.0);
        self.last_step = Some(now);
        let dpad_held = [
            pad.held(GamepadButton::DpadUp),
            pad.held(GamepadButton::DpadDown),
            pad.held(GamepadButton::DpadLeft),
            pad.held(GamepadButton::DpadRight),
        ];
        let nav = self.nav.step(dpad_held, pad.move_axis, dt);

        // While a world-pausing screen is open (the overlay build published
        // the state earlier this tick), freeze gameplay input so the camera
        // does not drift behind the menu; the UI still gets the cursor
        // position, clicks, and Escape. A non-pausing screen that captures
        // input (a live console) also suppresses gameplay keys -- the world
        // keeps simulating, but keystrokes belong to the screen.
        let menu_active = ctx
            .resource::<crate::ecs::MenuActive>()
            .map(|m| m.0)
            .unwrap_or(false);
        let screen_captures = ctx
            .resource::<crate::ecs::ScreenStack>()
            .is_some_and(|s| s.captures_input);
        // The editor's fly camera keeps navigation live while its menu
        // override freezes the world (the editor integrates the camera
        // itself); a shipped runtime never publishes FlyCam.
        let fly = ctx.resource::<crate::ecs::FlyCam>().is_some_and(|f| f.0);
        let gameplay = (!menu_active && !screen_captures) || fly;

        let frame_input = FrameInput {
            viewport: [vp_w, vp_h],
            ..compose_frame_input(&raw, &pad, self.map, nav, gameplay)
        };
        // Publish the same snapshot two ways: the resource readers can
        // fetch by type, and the component column the camera and UI
        // systems still query. Both readers query rather than drain, so the
        // column carries one entity for the world's life and each frame
        // overwrites it in place.
        ctx.insert_resource(frame_input.clone());
        match ctx.query_mut::<FrameInput>().next() {
            Some(slot) => *slot = frame_input,
            None => ctx.push(frame_input),
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::gamepad::{PadAxis, PadEvent};

    fn pad_snapshot(events: &[PadEvent]) -> PadSnapshot {
        let mut state = PadState::default();
        state.ingest(PadEvent::Connected { pad: 0 });
        for &e in events {
            state.ingest(e);
        }
        state.snapshot(0.15)
    }

    fn press(button: GamepadButton) -> PadEvent {
        PadEvent::Button {
            pad: 0,
            button,
            pressed: true,
        }
    }

    #[test]
    fn gamepad_buttons_merge_into_the_boolean_fields() {
        let pad = pad_snapshot(&[
            press(GamepadButton::DpadUp),
            press(GamepadMap::DEFAULT.jump),
            press(GamepadMap::DEFAULT.sprint),
            press(GamepadMap::DEFAULT.interact),
            press(GamepadButton::Start),
        ]);
        let input = compose_frame_input(
            &RenderInput::default(),
            &pad,
            GamepadMap::DEFAULT,
            None,
            true,
        );
        assert!(input.forward, "d-pad up drives forward");
        assert!(input.jump && input.sprint && input.interact);
        assert!(input.escape, "Start mirrors Escape");
        assert!(!input.backward && !input.left && !input.right);
    }

    #[test]
    fn keyboard_fields_pass_through_unchanged() {
        let raw = RenderInput {
            forward: true,
            sprint: true,
            mouse_dx: 3.0,
            mouse_dy: -2.0,
            scroll_delta: 1.5,
            ..Default::default()
        };
        let input = compose_frame_input(
            &raw,
            &PadSnapshot::default(),
            GamepadMap::DEFAULT,
            None,
            true,
        );
        assert!(input.forward && input.sprint);
        assert_eq!(input.mouse_dx, 3.0);
        assert_eq!(input.mouse_dy, -2.0);
        assert_eq!(input.scroll_delta, 1.5);
        assert_eq!(input.move_axis, [0.0, 0.0]);
        assert_eq!(input.captured_button, None);
    }

    // The UI modifiers pass the menu gate: they carry shortcuts that must work
    // while a menu (or the in-engine editor) holds the screen.
    #[test]
    fn ui_modifiers_stay_live_behind_the_menu_gate() {
        let raw = RenderInput {
            ctrl: true,
            alt: true,
            cmd: true,
            ..Default::default()
        };
        for gameplay in [true, false] {
            let input = compose_frame_input(
                &raw,
                &PadSnapshot::default(),
                GamepadMap::DEFAULT,
                None,
                gameplay,
            );
            assert!(input.ctrl && input.alt && input.cmd, "gameplay={gameplay}");
        }
    }

    #[test]
    fn rebound_map_routes_buttons_to_their_actions() {
        let mut map = GamepadMap::DEFAULT;
        map.rebind(crate::assets::GamepadAction::Jump, GamepadButton::North);
        let pad = pad_snapshot(&[press(GamepadButton::North)]);
        let input = compose_frame_input(&RenderInput::default(), &pad, map, None, true);
        assert!(input.jump, "jump follows the rebound button");
        let default_pad = pad_snapshot(&[press(GamepadButton::South)]);
        let input = compose_frame_input(&RenderInput::default(), &default_pad, map, None, true);
        assert!(!input.jump, "the old button no longer jumps");
    }

    #[test]
    fn menu_gate_freezes_gameplay_but_keeps_capture_and_escape() {
        let pad = pad_snapshot(&[
            press(GamepadButton::DpadUp),
            press(GamepadMap::DEFAULT.jump),
            press(GamepadButton::Start),
            PadEvent::Axis {
                pad: 0,
                axis: PadAxis::LeftY,
                value: 1.0,
            },
            PadEvent::Axis {
                pad: 0,
                axis: PadAxis::RightX,
                value: 1.0,
            },
        ]);
        let raw = RenderInput {
            forward: true,
            sprint: true,
            mouse_dx: 5.0,
            ..Default::default()
        };
        let input = compose_frame_input(&raw, &pad, GamepadMap::DEFAULT, None, false);
        // Movement, jump, axes, and mouse deltas all freeze behind the menu.
        assert!(!input.forward && !input.jump);
        assert!(!input.sprint, "sprint freezes behind the menu");
        assert!(input.shift, "the raw Shift modifier stays live for UI");
        assert_eq!(input.move_axis, [0.0, 0.0]);
        assert_eq!(input.look_axis, [0.0, 0.0]);
        assert_eq!(input.mouse_dx, 0.0);
        // The rebind capture and Escape stay live for the menu itself. South
        // precedes Start in declaration order, so it is the captured button.
        assert_eq!(input.captured_button, Some(GamepadMap::DEFAULT.jump));
        assert!(input.escape);
    }

    #[test]
    fn mouse_clicks_pass_the_menu_gate() {
        // Both click pulses serve UI, so neither freezes behind a menu: the
        // backend already withholds them while the cursor is captured.
        let raw = RenderInput {
            left_click: true,
            right_click: true,
            ..Default::default()
        };
        let pad = pad_snapshot(&[]);
        let input = compose_frame_input(&raw, &pad, GamepadMap::DEFAULT, None, false);
        assert!(input.left_click);
        assert!(input.right_click);
    }

    #[test]
    fn stick_axes_publish_when_gameplay_is_live() {
        let pad = pad_snapshot(&[
            PadEvent::Axis {
                pad: 0,
                axis: PadAxis::LeftY,
                value: 1.0,
            },
            PadEvent::Axis {
                pad: 0,
                axis: PadAxis::RightY,
                value: 1.0,
            },
        ]);
        let input = compose_frame_input(
            &RenderInput::default(),
            &pad,
            GamepadMap::DEFAULT,
            None,
            true,
        );
        assert!(
            (input.move_axis[1] - 1.0).abs() < 1e-6,
            "{:?}",
            input.move_axis
        );
        // Stick up looks up: negative Y in the mouse-delta convention.
        assert!(input.look_axis[1] < -0.9, "{:?}", input.look_axis);
    }

    #[test]
    fn nav_confirm_and_back_pass_the_menu_gate() {
        let pad = pad_snapshot(&[press(GamepadButton::South), press(GamepadButton::East)]);
        let nav = Some(crate::assets::NavDirection::Down);
        let input = compose_frame_input(
            &RenderInput::default(),
            &pad,
            GamepadMap::DEFAULT,
            nav,
            false,
        );
        assert_eq!(input.nav, nav);
        assert!(input.confirm, "South edge surfaces behind the menu");
        assert!(input.back, "East edge surfaces behind the menu");
        // The same fields surface during play too; UI just ignores them.
        let input = compose_frame_input(
            &RenderInput::default(),
            &pad,
            GamepadMap::DEFAULT,
            nav,
            true,
        );
        assert!(input.confirm && input.back);
    }
}
