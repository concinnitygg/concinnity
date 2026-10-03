// An authored action read as a name-level move: the part of an action that
// changes where the world is, carrying the names the world declares rather than
// the ids a build interns.

use concinnity_core::components::{AuthoredAction, NamedAction};
use serde_json::Value;

/// A move an action makes through the world's places.
///
/// The flow half of [`UiAction`](concinnity_core::components::UiAction): the
/// variants that change where the world is, carrying the names a world declares
/// rather than the ids a build interns. Operating a settings row or folding a
/// settings group changes no place, so neither has a variant here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Move {
    /// Change to a scene, dismissing every open screen.
    Scene(String),
    /// Show a screen, replacing the top of the stack.
    Show(String),
    /// Push a screen over the stack.
    Push(String),
    /// Show a screen, or hide it when it is already the top of the stack.
    Toggle(String),
    /// Close the top screen, uncovering what it sat over.
    Back,
    /// Drive the world's story.
    Story,
    /// Stop the application.
    Quit,
}

impl Move {
    /// The place the move names, or `None` for one that names none.
    pub fn target(&self) -> Option<&str> {
        match self {
            Move::Scene(name) | Move::Show(name) | Move::Push(name) | Move::Toggle(name) => {
                Some(name)
            }
            Move::Back | Move::Story | Move::Quit => None,
        }
    }

    /// The move as one word, short enough to label a wire between two places.
    pub fn verb(&self) -> &'static str {
        match self {
            Move::Scene(_) => "scene",
            Move::Show(_) => "show",
            Move::Push(_) => "push",
            Move::Toggle(_) => "toggle",
            Move::Back => "back",
            Move::Story => "story",
            Move::Quit => "quit",
        }
    }
}

impl Move {
    /// The move `action` makes, or `None` when it makes none: a settings row or
    /// group toggle, which change no place, or a target left empty, which the
    /// build rejects and which would put a nameless place on a map.
    pub fn of(action: &NamedAction) -> Option<Move> {
        let named = |name: &str| (!name.is_empty()).then(|| name.to_string());
        match action {
            AuthoredAction::Quit => Some(Move::Quit),
            AuthoredAction::Scene(name) => named(name).map(Move::Scene),
            AuthoredAction::Show(name) => named(name).map(Move::Show),
            AuthoredAction::Push(name) => named(name).map(Move::Push),
            AuthoredAction::Toggle(name) => named(name).map(Move::Toggle),
            AuthoredAction::Hide => Some(Move::Back),
            AuthoredAction::Story(_) => Some(Move::Story),
            AuthoredAction::GroupToggle(_) | AuthoredAction::Setting { .. } => None,
        }
    }
}

/// The move an authored action value makes, or `None` when it makes none or is
/// no action at all.
pub fn read_action(value: &Value) -> Option<Move> {
    serde_json::from_value::<NamedAction>(value.clone())
        .ok()
        .as_ref()
        .and_then(Move::of)
}

#[cfg(test)]
mod tests {
    use concinnity_core::components::{SettingVerb, StoryCommand};
    use concinnity_core::settings::SettingKey;
    use serde_json::json;

    use super::*;

    #[test]
    fn reads_each_move_to_its_target() {
        let read = |v: Value| read_action(&v);
        assert_eq!(read(json!("quit")), Some(Move::Quit));
        assert_eq!(
            read(json!({"scene": "pause"})),
            Some(Move::Scene("pause".to_string()))
        );
        assert_eq!(
            read(json!({"show": "pause"})),
            Some(Move::Show("pause".to_string()))
        );
        assert_eq!(
            read(json!({"push": "pause"})),
            Some(Move::Push("pause".to_string()))
        );
        assert_eq!(
            read(json!({"toggle": "pause"})),
            Some(Move::Toggle("pause".to_string()))
        );
        assert_eq!(read(json!("hide")), Some(Move::Back));
        assert_eq!(read(json!({"story": {"choose": 1}})), Some(Move::Story));
    }

    #[test]
    fn a_settings_row_an_empty_target_or_no_action_makes_no_move() {
        for value in [
            json!({"group_toggle": 3}),
            json!({"setting": {"key": "master_volume", "verb": "drag"}}),
            json!({"show": null}),
            json!("teleport"),
            json!(null),
        ] {
            assert_eq!(read_action(&value), None, "{value}");
        }
    }

    #[test]
    fn only_a_move_naming_a_place_answers_a_target() {
        assert_eq!(Move::Scene("pause".to_string()).target(), Some("pause"));
        assert_eq!(Move::Back.target(), None);
        assert_eq!(Move::Quit.target(), None);
        assert_eq!(Move::Story.target(), None);
    }

    // The word goes in the gap between two cards, so it has to stay one.
    #[test]
    fn every_verb_is_one_short_word() {
        let actions: [NamedAction; 7] = [
            AuthoredAction::Quit,
            AuthoredAction::Scene("s".into()),
            AuthoredAction::Show("s".into()),
            AuthoredAction::Push("s".into()),
            AuthoredAction::Toggle("s".into()),
            AuthoredAction::Hide,
            AuthoredAction::Story(StoryCommand::Advance),
        ];
        for action in &actions {
            let verb = Move::of(action).unwrap().verb();
            assert!(
                verb.len() <= 8 && verb.chars().all(|c| c.is_ascii_lowercase()),
                "`{verb}`"
            );
        }
        let row = AuthoredAction::Setting {
            key: SettingKey::Vsync,
            verb: SettingVerb::Next,
        };
        assert_eq!(Move::of(&row), None);
    }
}
