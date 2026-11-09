//! Gamepad state folding. Everything here operates on the small `PadEvent`
//! enum the host's gamepad adapter produces, so the fold and snapshot logic is
//! driven synthetically in tests without hardware. One gamepad is active at a
//! time: the most recently used pad wins, and a disconnect zeroes the state so
//! no input sticks.

use crate::components::GamepadButton;

use super::stick;

// Exponent of the look-stick response curve: squared magnitude gives fine aim
// near center while full deflection stays full.
const LOOK_CURVE_EXPONENT: f32 = 2.0;
// Deflection past which an inactive pad's stick claims the active slot, well
// clear of any sane deadzone so drift on an idle pad never steals input.
const ACTIVATE_DEFLECTION: f32 = 0.5;

/// A stick axis in the raw gamepad convention: positive X is rightward,
/// positive Y is up/forward (the look Y flip to mouse convention happens in
/// [`PadState::snapshot`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadAxis {
    /// Left stick, horizontal.
    LeftX,
    /// Left stick, vertical.
    LeftY,
    /// Right stick, horizontal.
    RightX,
    /// Right stick, vertical.
    RightY,
}

/// A backend-agnostic gamepad event, converted from the host's gamepad library
/// (or synthesized by tests) and folded into [`PadState`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadEvent {
    /// A pad connected.
    Connected {
        /// The host's pad id.
        pad: u32,
    },
    /// A pad disconnected.
    Disconnected {
        /// The host's pad id.
        pad: u32,
    },
    /// A button changed.
    Button {
        /// The host's pad id.
        pad: u32,
        /// Which button.
        button: GamepadButton,
        /// Pressed or released.
        pressed: bool,
    },
    /// Stick deflection in [-1, 1].
    Axis {
        /// The host's pad id.
        pad: u32,
        /// Which axis.
        axis: PadAxis,
        /// The deflection.
        value: f32,
    },
}

fn bit(button: GamepadButton) -> u32 {
    1 << (button as u32)
}

/// Accumulated state of the active gamepad, folded from [`PadEvent`]s.
#[derive(Debug, Default)]
pub struct PadState {
    active: Option<u32>,
    left: [f32; 2],
    right: [f32; 2],
    // Buttons currently held, as a bitmask by `GamepadButton` discriminant.
    held: u32,
    // Press edges since the last snapshot (drained by `snapshot`).
    pressed: u32,
}

impl PadState {
    /// Fold one event into the state.
    pub fn ingest(&mut self, event: PadEvent) {
        match event {
            PadEvent::Connected { pad } => {
                if self.active.is_none() {
                    self.active = Some(pad);
                }
            }
            PadEvent::Disconnected { pad } => {
                if self.active == Some(pad) {
                    *self = Self::default();
                }
            }
            PadEvent::Button {
                pad,
                button,
                pressed,
            } => {
                if Some(pad) != self.active {
                    // A press on another pad claims the active slot; a stray
                    // release does not.
                    if !pressed {
                        return;
                    }
                    self.switch_to(pad);
                }
                if pressed {
                    self.held |= bit(button);
                    self.pressed |= bit(button);
                } else {
                    self.held &= !bit(button);
                }
            }
            PadEvent::Axis { pad, axis, value } => {
                if Some(pad) != self.active {
                    // Idle-pad drift never steals input; a deliberate
                    // deflection does.
                    if value.abs() < ACTIVATE_DEFLECTION {
                        return;
                    }
                    self.switch_to(pad);
                }
                match axis {
                    PadAxis::LeftX => self.left[0] = value,
                    PadAxis::LeftY => self.left[1] = value,
                    PadAxis::RightX => self.right[0] = value,
                    PadAxis::RightY => self.right[1] = value,
                }
            }
        }
    }

    // Make `pad` the active gamepad, dropping the previous pad's state so its
    // held buttons and deflections do not leak across the switch.
    fn switch_to(&mut self, pad: u32) {
        *self = Self::default();
        self.active = Some(pad);
    }

    /// Shape the accumulated state into this frame's snapshot, draining the
    /// press edges. `deadzone` is the radial deadzone fraction in [0, 1).
    pub fn snapshot(&mut self, deadzone: f32) -> PadSnapshot {
        let pressed = core::mem::take(&mut self.pressed);
        let look = stick::response_curve(
            stick::radial_deadzone(self.right, deadzone),
            LOOK_CURVE_EXPONENT,
        );
        PadSnapshot {
            move_axis: stick::radial_deadzone(self.left, deadzone),
            // Stick Y up -> mouse-convention Y down.
            look_axis: [look[0], -look[1]],
            held: self.held,
            pressed,
        }
    }
}

/// One frame's shaped gamepad input, composed into the frame's `FrameInput`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PadSnapshot {
    /// Left stick after the radial deadzone, magnitude <= 1; +Y is forward.
    pub move_axis: [f32; 2],
    /// Right stick after deadzone + response curve, in the mouse-delta sign
    /// convention (+Y looks down), magnitude <= 1.
    pub look_axis: [f32; 2],
    held: u32,
    pressed: u32,
}

impl PadSnapshot {
    /// Whether `button` is held.
    pub fn held(&self, button: GamepadButton) -> bool {
        self.held & bit(button) != 0
    }

    /// Whether `button` was pressed since the last snapshot.
    pub fn pressed(&self, button: GamepadButton) -> bool {
        self.pressed & bit(button) != 0
    }

    /// The button pressed this frame, for the settings rebind capture, or
    /// `None`. With several pressed on one frame, the first in declaration
    /// order wins.
    pub fn first_pressed(&self) -> Option<GamepadButton> {
        GamepadButton::ALL
            .iter()
            .copied()
            .find(|&b| self.pressed(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(pad: u32, button: GamepadButton) -> PadEvent {
        PadEvent::Button {
            pad,
            button,
            pressed: true,
        }
    }

    fn release(pad: u32, button: GamepadButton) -> PadEvent {
        PadEvent::Button {
            pad,
            button,
            pressed: false,
        }
    }

    #[test]
    fn press_sets_held_and_a_one_frame_edge() {
        let mut state = PadState::default();
        state.ingest(PadEvent::Connected { pad: 0 });
        state.ingest(press(0, GamepadButton::South));

        let snap = state.snapshot(0.15);
        assert!(snap.held(GamepadButton::South));
        assert!(snap.pressed(GamepadButton::South));
        assert_eq!(snap.first_pressed(), Some(GamepadButton::South));

        // The edge drains with the snapshot; held persists until release.
        let snap = state.snapshot(0.15);
        assert!(snap.held(GamepadButton::South));
        assert!(!snap.pressed(GamepadButton::South));
        assert_eq!(snap.first_pressed(), None);

        state.ingest(release(0, GamepadButton::South));
        assert!(!state.snapshot(0.15).held(GamepadButton::South));
    }

    #[test]
    fn press_between_snapshots_is_not_lost() {
        // A press and release inside one frame still surfaces its edge.
        let mut state = PadState::default();
        state.ingest(PadEvent::Connected { pad: 0 });
        state.ingest(press(0, GamepadButton::West));
        state.ingest(release(0, GamepadButton::West));
        let snap = state.snapshot(0.15);
        assert!(snap.pressed(GamepadButton::West));
        assert!(!snap.held(GamepadButton::West));
    }

    #[test]
    fn axes_shape_through_deadzone_and_look_flips_y() {
        let mut state = PadState::default();
        state.ingest(PadEvent::Connected { pad: 0 });
        state.ingest(PadEvent::Axis {
            pad: 0,
            axis: PadAxis::LeftY,
            value: 1.0,
        });
        state.ingest(PadEvent::Axis {
            pad: 0,
            axis: PadAxis::RightY,
            value: 1.0,
        });

        let snap = state.snapshot(0.15);
        // Full forward stays full after the deadzone rescale.
        assert!(
            (snap.move_axis[1] - 1.0).abs() < 1e-6,
            "{:?}",
            snap.move_axis
        );
        // Stick up (+1) becomes look up: negative in the mouse convention.
        assert!(
            (snap.look_axis[1] + 1.0).abs() < 1e-6,
            "{:?}",
            snap.look_axis
        );

        // A small deflection inside the deadzone reads as rest.
        state.ingest(PadEvent::Axis {
            pad: 0,
            axis: PadAxis::LeftY,
            value: 0.1,
        });
        assert_eq!(state.snapshot(0.15).move_axis, [0.0, 0.0]);
    }

    #[test]
    fn most_recently_used_pad_wins_and_state_resets_on_switch() {
        let mut state = PadState::default();
        state.ingest(PadEvent::Connected { pad: 0 });
        state.ingest(PadEvent::Connected { pad: 1 });
        state.ingest(press(0, GamepadButton::South));

        // A deliberate press on pad 1 claims the slot; pad 0's held button
        // does not leak across.
        state.ingest(press(1, GamepadButton::North));
        let snap = state.snapshot(0.15);
        assert!(snap.held(GamepadButton::North));
        assert!(!snap.held(GamepadButton::South));
    }

    #[test]
    fn inactive_pad_drift_and_releases_are_ignored() {
        let mut state = PadState::default();
        state.ingest(PadEvent::Connected { pad: 0 });
        state.ingest(PadEvent::Axis {
            pad: 0,
            axis: PadAxis::LeftY,
            value: 1.0,
        });
        // Drift under the activation threshold on pad 1: pad 0 stays active.
        state.ingest(PadEvent::Axis {
            pad: 1,
            axis: PadAxis::LeftX,
            value: 0.2,
        });
        let snap = state.snapshot(0.15);
        assert!(
            (snap.move_axis[1] - 1.0).abs() < 1e-6,
            "{:?}",
            snap.move_axis
        );

        // A stray release on an inactive pad neither switches nor clears.
        state.ingest(release(1, GamepadButton::South));
        assert!((state.snapshot(0.15).move_axis[1] - 1.0).abs() < 1e-6);

        // A deliberate deflection does switch, resetting pad 0's stick.
        state.ingest(PadEvent::Axis {
            pad: 1,
            axis: PadAxis::LeftX,
            value: 0.9,
        });
        let snap = state.snapshot(0.15);
        assert!(snap.move_axis[0] > 0.5, "{:?}", snap.move_axis);
        assert!(snap.move_axis[1].abs() < 1e-6, "{:?}", snap.move_axis);
    }

    #[test]
    fn disconnect_of_the_active_pad_zeroes_everything() {
        let mut state = PadState::default();
        state.ingest(PadEvent::Connected { pad: 0 });
        state.ingest(press(0, GamepadButton::South));
        state.ingest(PadEvent::Axis {
            pad: 0,
            axis: PadAxis::LeftY,
            value: 1.0,
        });
        state.ingest(PadEvent::Disconnected { pad: 0 });

        let snap = state.snapshot(0.15);
        assert_eq!(snap.move_axis, [0.0, 0.0]);
        assert!(!snap.held(GamepadButton::South));

        // A disconnect of some other pad leaves the active one untouched.
        state.ingest(PadEvent::Connected { pad: 1 });
        state.ingest(press(1, GamepadButton::East));
        state.ingest(PadEvent::Disconnected { pad: 7 });
        assert!(state.snapshot(0.15).held(GamepadButton::East));
    }

    #[test]
    fn first_pressed_takes_declaration_order() {
        let mut state = PadState::default();
        state.ingest(PadEvent::Connected { pad: 0 });
        state.ingest(press(0, GamepadButton::Start));
        state.ingest(press(0, GamepadButton::South));
        // South precedes Start in declaration order regardless of press order.
        assert_eq!(
            state.snapshot(0.15).first_pressed(),
            Some(GamepadButton::South)
        );
    }
}
