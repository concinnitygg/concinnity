//! Which editing command a key press asks for, by platform convention: macOS
//! builds shortcuts on Command and word steps on Option, Windows and Linux both
//! on Control.

use concinnity_core::components::{InputKey, KeyPress};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Platform {
    Mac,
    Other,
}

impl Platform {
    pub(crate) const fn current() -> Self {
        if cfg!(target_os = "macos") {
            Platform::Mac
        } else {
            Platform::Other
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Motion {
    Left,
    Right,
    WordLeft,
    WordRight,
    Up,
    Down,
    LineStart,
    LineEnd,
    DocStart,
    DocEnd,
    PageUp,
    PageDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Command {
    // Move the caret; `extend` keeps the anchor, growing the selection.
    Move { motion: Motion, extend: bool },
    Backspace { word: bool },
    Delete { word: bool },
    Newline,
    Indent,
    Outdent,
    SelectAll,
    Cut,
    Copy,
    Paste,
    Undo,
    Redo,
    Save,
}

// The command `press` asks for, or `None` for a key the text area ignores.
pub(crate) fn command(press: KeyPress, platform: Platform) -> Option<Command> {
    let m = press.mods;
    let mac = platform == Platform::Mac;
    // AltGr arrives as Ctrl+Alt on Windows and types a character rather than
    // naming a shortcut.
    let shortcut = if mac { m.cmd } else { m.ctrl && !m.alt };
    let word = if mac { m.alt } else { m.ctrl };
    let extend = m.shift;
    let step = |motion| Some(Command::Move { motion, extend });
    match press.key {
        InputKey::Left if mac && m.cmd => step(Motion::LineStart),
        InputKey::Right if mac && m.cmd => step(Motion::LineEnd),
        InputKey::Up if mac && m.cmd => step(Motion::DocStart),
        InputKey::Down if mac && m.cmd => step(Motion::DocEnd),
        InputKey::Left if word => step(Motion::WordLeft),
        InputKey::Right if word => step(Motion::WordRight),
        InputKey::Left => step(Motion::Left),
        InputKey::Right => step(Motion::Right),
        InputKey::Up => step(Motion::Up),
        InputKey::Down => step(Motion::Down),
        InputKey::Home if shortcut => step(Motion::DocStart),
        InputKey::End if shortcut => step(Motion::DocEnd),
        InputKey::Home => step(Motion::LineStart),
        InputKey::End => step(Motion::LineEnd),
        InputKey::PageUp => step(Motion::PageUp),
        InputKey::PageDown => step(Motion::PageDown),
        InputKey::Backspace => Some(Command::Backspace { word }),
        InputKey::Delete => Some(Command::Delete { word }),
        InputKey::Enter if !shortcut => Some(Command::Newline),
        InputKey::Tab if !shortcut => Some(if m.shift {
            Command::Outdent
        } else {
            Command::Indent
        }),
        InputKey::A if shortcut => Some(Command::SelectAll),
        InputKey::X if shortcut => Some(Command::Cut),
        InputKey::C if shortcut => Some(Command::Copy),
        InputKey::V if shortcut => Some(Command::Paste),
        InputKey::Z if shortcut && m.shift => Some(Command::Redo),
        InputKey::Z if shortcut => Some(Command::Undo),
        InputKey::Y if shortcut && !mac => Some(Command::Redo),
        InputKey::S if shortcut => Some(Command::Save),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::KeyMods;

    fn cmd(key: InputKey, mods: KeyMods, platform: Platform) -> Option<Command> {
        command(KeyPress::new(key, mods), platform)
    }

    fn mv(motion: Motion, extend: bool) -> Option<Command> {
        Some(Command::Move { motion, extend })
    }

    #[test]
    fn arrows_move_and_shift_extends() {
        for p in [Platform::Mac, Platform::Other] {
            assert_eq!(
                cmd(InputKey::Left, KeyMods::NONE, p),
                mv(Motion::Left, false)
            );
            assert_eq!(
                cmd(InputKey::Down, KeyMods::SHIFT, p),
                mv(Motion::Down, true)
            );
            assert_eq!(
                cmd(InputKey::PageUp, KeyMods::NONE, p),
                mv(Motion::PageUp, false)
            );
        }
    }

    #[test]
    fn word_steps_use_option_on_mac_and_ctrl_elsewhere() {
        assert_eq!(
            cmd(InputKey::Left, KeyMods::ALT, Platform::Mac),
            mv(Motion::WordLeft, false)
        );
        assert_eq!(
            cmd(InputKey::Right, KeyMods::CTRL.with_shift(), Platform::Other),
            mv(Motion::WordRight, true)
        );
        assert_eq!(
            cmd(InputKey::Backspace, KeyMods::ALT, Platform::Mac),
            Some(Command::Backspace { word: true })
        );
        assert_eq!(
            cmd(InputKey::Delete, KeyMods::CTRL, Platform::Other),
            Some(Command::Delete { word: true })
        );
        assert_eq!(
            cmd(InputKey::Backspace, KeyMods::NONE, Platform::Other),
            Some(Command::Backspace { word: false })
        );
    }

    #[test]
    fn command_arrows_reach_line_and_document_ends_on_mac() {
        let m = Platform::Mac;
        assert_eq!(
            cmd(InputKey::Left, KeyMods::CMD, m),
            mv(Motion::LineStart, false)
        );
        assert_eq!(
            cmd(InputKey::Right, KeyMods::CMD, m),
            mv(Motion::LineEnd, false)
        );
        assert_eq!(
            cmd(InputKey::Up, KeyMods::CMD, m),
            mv(Motion::DocStart, false)
        );
        assert_eq!(
            cmd(InputKey::Down, KeyMods::CMD.with_shift(), m),
            mv(Motion::DocEnd, true)
        );
    }

    #[test]
    fn home_and_end_reach_the_line_or_with_the_shortcut_the_document() {
        assert_eq!(
            cmd(InputKey::Home, KeyMods::NONE, Platform::Other),
            mv(Motion::LineStart, false)
        );
        assert_eq!(
            cmd(InputKey::End, KeyMods::CTRL, Platform::Other),
            mv(Motion::DocEnd, false)
        );
        assert_eq!(
            cmd(InputKey::Home, KeyMods::CMD, Platform::Mac),
            mv(Motion::DocStart, false)
        );
    }

    #[test]
    fn shortcuts_follow_the_platform_modifier() {
        let (m, o) = (Platform::Mac, Platform::Other);
        assert_eq!(cmd(InputKey::C, KeyMods::CMD, m), Some(Command::Copy));
        assert_eq!(
            cmd(InputKey::C, KeyMods::CTRL, m),
            None,
            "Ctrl+C is not copy on mac"
        );
        assert_eq!(cmd(InputKey::V, KeyMods::CTRL, o), Some(Command::Paste));
        assert_eq!(cmd(InputKey::X, KeyMods::CTRL, o), Some(Command::Cut));
        assert_eq!(cmd(InputKey::A, KeyMods::CMD, m), Some(Command::SelectAll));
        assert_eq!(cmd(InputKey::S, KeyMods::CTRL, o), Some(Command::Save));
        assert_eq!(cmd(InputKey::Z, KeyMods::CMD, m), Some(Command::Undo));
        assert_eq!(
            cmd(InputKey::Z, KeyMods::CMD.with_shift(), m),
            Some(Command::Redo)
        );
        assert_eq!(cmd(InputKey::Y, KeyMods::CTRL, o), Some(Command::Redo));
        assert_eq!(cmd(InputKey::Y, KeyMods::CMD, m), None);
        assert_eq!(
            cmd(InputKey::A, KeyMods::NONE, o),
            None,
            "a plain letter types"
        );
    }

    #[test]
    fn altgr_is_not_a_shortcut() {
        let altgr = KeyMods {
            ctrl: true,
            alt: true,
            ..KeyMods::NONE
        };
        assert_eq!(cmd(InputKey::V, altgr, Platform::Other), None);
    }

    #[test]
    fn tab_indents_and_shift_tab_outdents() {
        let p = Platform::Other;
        assert_eq!(cmd(InputKey::Tab, KeyMods::NONE, p), Some(Command::Indent));
        assert_eq!(
            cmd(InputKey::Tab, KeyMods::SHIFT, p),
            Some(Command::Outdent)
        );
        assert_eq!(
            cmd(InputKey::Enter, KeyMods::NONE, p),
            Some(Command::Newline)
        );
    }
}
