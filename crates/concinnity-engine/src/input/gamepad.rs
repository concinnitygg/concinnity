//! The gilrs adapter: OS gamepad events converted to core's `PadEvent` and
//! folded into a `PadState`.

use concinnity_core::components::GamepadButton;
use concinnity_core::input::pad::{PadAxis, PadEvent, PadState};

// The OS-facing gamepad event source. Construction is deferred to the first
// poll so a world without a step never starts the platform's HID machinery;
// an init failure (no HID access) logs once and leaves gamepads inert.
#[derive(Debug, Default)]
pub(crate) struct GamepadSource {
    gilrs: Option<gilrs::Gilrs>,
    tried: bool,
}

impl GamepadSource {
    // Drain pending OS gamepad events into `state`.
    pub(crate) fn poll(&mut self, state: &mut PadState) {
        if !self.tried {
            self.tried = true;
            match gilrs::Gilrs::new() {
                Ok(gilrs) => {
                    // Seed pads that were already connected before init.
                    for (id, _) in gilrs.gamepads() {
                        state.ingest(PadEvent::Connected {
                            pad: usize::from(id) as u32,
                        });
                    }
                    self.gilrs = Some(gilrs);
                }
                Err(e) => tracing::warn!("gamepad input unavailable: {e}"),
            }
        }
        let Some(gilrs) = self.gilrs.as_mut() else {
            return;
        };
        while let Some(event) = gilrs.next_event() {
            if let Some(pad_event) = convert(&event) {
                state.ingest(pad_event);
            }
        }
    }
}

// A gilrs event as a `PadEvent`, or `None` for event kinds and buttons the
// engine does not consume. Triggers arrive as buttons (gilrs applies its press
// threshold), so they bind like any other button.
fn convert(event: &gilrs::Event) -> Option<PadEvent> {
    let pad = usize::from(event.id) as u32;
    Some(match event.event {
        gilrs::EventType::ButtonPressed(b, _) => PadEvent::Button {
            pad,
            button: button_of(b)?,
            pressed: true,
        },
        gilrs::EventType::ButtonReleased(b, _) => PadEvent::Button {
            pad,
            button: button_of(b)?,
            pressed: false,
        },
        gilrs::EventType::AxisChanged(a, value, _) => PadEvent::Axis {
            pad,
            axis: axis_of(a)?,
            value,
        },
        gilrs::EventType::Connected => PadEvent::Connected { pad },
        gilrs::EventType::Disconnected => PadEvent::Disconnected { pad },
        _ => return None,
    })
}

// gilrs names the shoulder row LeftTrigger/LeftTrigger2 (bumper/trigger); map
// both onto the engine's Shoulder/Trigger naming. Mode (the vendor button) is
// dropped: the OS commonly reserves it.
fn button_of(button: gilrs::Button) -> Option<GamepadButton> {
    Some(match button {
        gilrs::Button::South => GamepadButton::South,
        gilrs::Button::East => GamepadButton::East,
        gilrs::Button::West => GamepadButton::West,
        gilrs::Button::North => GamepadButton::North,
        gilrs::Button::LeftTrigger => GamepadButton::LeftShoulder,
        gilrs::Button::LeftTrigger2 => GamepadButton::LeftTrigger,
        gilrs::Button::RightTrigger => GamepadButton::RightShoulder,
        gilrs::Button::RightTrigger2 => GamepadButton::RightTrigger,
        gilrs::Button::LeftThumb => GamepadButton::LeftStick,
        gilrs::Button::RightThumb => GamepadButton::RightStick,
        gilrs::Button::DPadUp => GamepadButton::DpadUp,
        gilrs::Button::DPadDown => GamepadButton::DpadDown,
        gilrs::Button::DPadLeft => GamepadButton::DpadLeft,
        gilrs::Button::DPadRight => GamepadButton::DpadRight,
        gilrs::Button::Start => GamepadButton::Start,
        gilrs::Button::Select => GamepadButton::Select,
        _ => return None,
    })
}

fn axis_of(axis: gilrs::Axis) -> Option<PadAxis> {
    Some(match axis {
        gilrs::Axis::LeftStickX => PadAxis::LeftX,
        gilrs::Axis::LeftStickY => PadAxis::LeftY,
        gilrs::Axis::RightStickX => PadAxis::RightX,
        gilrs::Axis::RightStickY => PadAxis::RightY,
        _ => return None,
    })
}
