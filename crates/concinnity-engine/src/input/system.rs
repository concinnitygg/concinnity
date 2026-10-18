//! Consumes the tick's `InputMailbox` packet (sampled beside the backend before
//! the tick began), merges the active gamepad's state, and publishes the
//! `FrameInput` snapshot (as a resource and as the component column) plus the
//! `CursorState` resource. Runs at the top of the tick, right after the menu
//! gate, so every input consumer (behaviors, camera controllers, UI, text
//! input) runs later in the same tick and the frame it ends with shows them.
//!
//! The gamepad is polled here, not per render backend, so all backends share
//! one implementation. Digital buttons merge into the existing boolean fields
//! (d-pad onto the movement keys, the bound buttons onto sprint / jump /
//! interact, Start onto escape); the sticks publish as the analog
//! `move_axis` / `look_axis` fields.

use concinnity_core::components::{ControlsCommand, FrameInput, GamepadButton, GamepadMap};
use concinnity_core::ecs::{
    Access, CursorState, EventCursor, FlyCam, FrameTime, MenuActive, PipelineContext, ScreenStack,
    StepResult, System, WindowMinimized,
};
use concinnity_core::input::compose::{compose_frame_input, nav_dt};
use concinnity_core::input::nav::NavRepeat;
use concinnity_core::input::pad::PadState;

use crate::ecs::InputMailbox;
use crate::input::gamepad::GamepadSource;

#[derive(Debug, Default)]
pub(crate) struct InputSystem {
    gamepad: GamepadSource,
    pad: PadState,
    map: GamepadMap,
    deadzone: f32,
    // Auto-repeat state for the d-pad / stick navigation pulse.
    nav: NavRepeat,
    // Cursor into the Events<ControlsCommand> queue (live settings changes).
    controls_cursor: EventCursor,
}

impl InputSystem {
    pub(crate) fn new() -> Self {
        Self {
            deadzone: concinnity_core::settings::registry::DEFAULT_GAMEPAD_DEADZONE,
            ..Self::default()
        }
    }
}

impl System for InputSystem {
    fn access(&self) -> Access {
        Access::new()
            .writes_components(concinnity_core::component_mask![FrameInput])
            .reads_resources(concinnity_core::resource_mask![
                MenuActive,
                ScreenStack,
                FlyCam,
                ControlsCommand,
                FrameTime,
            ])
            .writes_resources(concinnity_core::resource_mask![
                crate::ecs::InputMailbox,
                concinnity_core::ecs::InputSampledAt,
                CursorState,
                WindowMinimized,
                FrameInput,
            ])
    }

    fn init(&mut self, ctx: &mut PipelineContext) {
        // Stored settings-menu choices override the engine defaults.
        if let Some(stored) = ctx.resource::<concinnity_core::settings::StoredSettings>() {
            let controls = stored.controls_command();
            self.map = controls.gamepad_map.unwrap_or_default();
            if let Some(dz) = controls.gamepad_deadzone {
                self.deadzone = dz;
            }
        }
        // Seed the one FrameInput row here so `step` only ever overwrites it
        // in place: minting the row is a structural change, which the step's
        // declared (non-exclusive) access cannot perform. Consumers treat the
        // default snapshot exactly like the empty column: no keys, no clicks.
        if ctx.query::<FrameInput>().next().is_none() {
            ctx.push(FrameInput::default());
        }
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // No deposited packet (graphics failed, the editor transplanted the
        // backend away, or the frame stopped before sampling): nothing to
        // consume; consumers keep the last snapshot.
        let Some((packet, sampled_us)) = ctx
            .resource_mut::<InputMailbox>()
            .and_then(InputMailbox::take)
        else {
            return StepResult::Continue;
        };
        ctx.insert_resource(concinnity_core::ecs::InputSampledAt(sampled_us));
        let raw = packet.raw;
        let cursor_outside = packet.cursor_outside_window;
        // Live viewport for UiInputSystem's overlay hit-testing, so a scaled
        // menu's HitRegions map back to the cursor consistently.
        let (vp_w, vp_h) = packet.viewport;
        // Window chrome overlapping the top of the frame (macOS's transparent
        // title bar), for UI that must stay clear of the OS window buttons.
        let top_inset = packet.top_inset;

        // The cursor position + window-bounds state for next frame's draw
        // list (`follow_cursor` sprites are positioned a frame after the input
        // that moved them; the draw list is built before this poll).
        ctx.insert_resource(CursorState {
            pos: (raw.mouse_x, raw.mouse_y),
            outside_window: cursor_outside,
        });
        ctx.insert_resource(WindowMinimized(packet.window_minimized));

        // Live controls changes (a gamepad rebind or deadzone slider) sent by
        // SettingsSystem earlier this tick.
        if let Some(events) = ctx.events::<ControlsCommand>() {
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
        // pulse (auto-repeat needs a dt; a hitch clamps).
        let dt = nav_dt(ctx.resource::<FrameTime>().copied().unwrap_or_default());
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
        let menu_active = ctx.resource::<MenuActive>().map(|m| m.0).unwrap_or(false);
        let screen_captures = ctx
            .resource::<ScreenStack>()
            .is_some_and(|s| s.captures_input);
        // The editor's fly camera keeps navigation live while its menu
        // override freezes the world (the editor integrates the camera
        // itself); a shipped runtime never publishes FlyCam.
        let fly = ctx.resource::<FlyCam>().is_some_and(|f| f.0);
        let gameplay = (!menu_active && !screen_captures) || fly;

        let frame_input = FrameInput {
            viewport: [vp_w, vp_h],
            top_inset,
            ..compose_frame_input(&raw, &pad, self.map, nav, gameplay)
        };
        // Publish the same snapshot two ways: the resource readers can
        // fetch by type, and the component column the camera and UI
        // systems still query. Both readers query rather than drain, so the
        // column carries one entity for the world's life (seeded at init) and
        // each frame overwrites it in place.
        ctx.insert_resource(frame_input.clone());
        if let Some(slot) = ctx.query_mut::<FrameInput>().next() {
            *slot = frame_input;
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use concinnity_core::input::snapshot::InputPacket;

    // The window-metrics seam: a sampled packet carries whatever chrome the
    // backend reports over the top of its frame, and a backend reporting none
    // leaves it at zero (which is every platform but macOS).
    #[test]
    fn a_sampled_packet_carries_the_backends_top_content_inset() {
        use concinnity_core::render::backend::recording::recording_backend;
        let (state, mut backend) = recording_backend();
        assert_eq!(InputPacket::sample(&mut backend).top_inset, 0.0);
        state.lock().top_inset = 28.0;
        assert_eq!(InputPacket::sample(&mut backend).top_inset, 28.0);
    }
}
