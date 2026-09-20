// Action text read back to a name-level move. The builders in
// `spec::asset::ui_action` write these strings and `UiAction::parse` reads them
// into interned ids; an authoring tool needs the names they were written with,
// before a build interns anything.

use concinnity_core::components::StoryCommand;

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

/// The move `text` makes, or `None` when it makes none: a settings row, a group
/// toggle, `screen:clear` (which only the engine sends), or text naming no
/// action at all.
///
/// A malformed target is no move either. The build rejects one, and answering
/// with it would put a nameless place on a map.
pub fn parse_action(text: &str) -> Option<Move> {
    if text == "quit" {
        return Some(Move::Quit);
    }
    let (kind, rest) = text.split_once(':')?;
    let named = |name: &str| (!name.is_empty()).then(|| name.to_string());
    match kind {
        "scene" => named(rest).map(Move::Scene),
        "screen" => match rest.split_once(':') {
            Some(("show", name)) => named(name).map(Move::Show),
            Some(("push", name)) => named(name).map(Move::Push),
            Some(("toggle", name)) => named(name).map(Move::Toggle),
            None if rest == "hide" => Some(Move::Back),
            _ => None,
        },
        "story" => {
            let verb = rest.split_once(':').map_or(rest, |(verb, _)| verb);
            StoryCommand::VERBS.contains(&verb).then_some(Move::Story)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use concinnity_core::components::{ScreenCommand, UiAction};
    use concinnity_core::ecs::asset_id::AssetId;

    use super::*;

    // Every text below resolves against a world declaring exactly this name, so
    // a target that fails to resolve is the text's fault and not the world's.
    fn ui_action(text: &str) -> Option<UiAction> {
        UiAction::parse(text, |name| (name == "pause").then_some(AssetId(9))).ok()
    }

    // Whether the engine's own reading of the text changes where the world is.
    fn ui_action_moves(text: &str) -> bool {
        matches!(
            ui_action(text),
            Some(UiAction::Quit | UiAction::Scene(_) | UiAction::Screen(_) | UiAction::Story(_))
        )
    }

    // Everything the two readings are held to agreeing on. A vocabulary that
    // grows on one side without the other shows up here as a disagreement.
    const TEXTS: &[&str] = &[
        "quit",
        "scene:pause",
        "screen:show:pause",
        "screen:push:pause",
        "screen:toggle:pause",
        "screen:hide",
        "story:advance",
        "story:page:2",
        "group:toggle:3",
        "setting:volume_master:drag",
        "screen:clear",
        "screen:hide:",
        "screen:show:",
        "screen:nudge:pause",
        "scene:",
        "story:",
        "story:bogus",
        "",
        "pause",
    ];

    #[test]
    fn reads_the_same_texts_as_a_move_that_the_engine_reads_as_one() {
        for text in TEXTS {
            assert_eq!(
                parse_action(text).is_some(),
                ui_action_moves(text),
                "`{text}`"
            );
        }
    }

    #[test]
    fn reads_each_move_to_its_target() {
        assert_eq!(parse_action("quit"), Some(Move::Quit));
        assert_eq!(
            parse_action("scene:pause"),
            Some(Move::Scene("pause".to_string()))
        );
        assert_eq!(
            parse_action("screen:show:pause"),
            Some(Move::Show("pause".to_string()))
        );
        assert_eq!(
            parse_action("screen:push:pause"),
            Some(Move::Push("pause".to_string()))
        );
        assert_eq!(
            parse_action("screen:toggle:pause"),
            Some(Move::Toggle("pause".to_string()))
        );
        assert_eq!(parse_action("screen:hide"), Some(Move::Back));
        assert_eq!(parse_action("story:advance"), Some(Move::Story));
    }

    // An unresolved id target is a name like any other here: the engine interns
    // it and the map draws the place it names.
    #[test]
    fn a_numeric_target_is_read_as_the_name_it_was_written_as() {
        assert_eq!(
            parse_action("screen:show:7"),
            Some(Move::Show("7".to_string()))
        );
        assert_eq!(
            ui_action("screen:show:7"),
            Some(UiAction::Screen(ScreenCommand::Show(AssetId(7))))
        );
    }

    #[test]
    fn only_a_move_naming_a_place_answers_a_target() {
        assert_eq!(parse_action("scene:pause").unwrap().target(), Some("pause"));
        assert_eq!(parse_action("screen:hide").unwrap().target(), None);
        assert_eq!(parse_action("quit").unwrap().target(), None);
        assert_eq!(parse_action("story:advance").unwrap().target(), None);
    }

    // The word goes in the gap between two cards, so it has to stay one.
    #[test]
    fn every_verb_is_one_short_word() {
        for text in TEXTS.iter().filter_map(|t| parse_action(t)) {
            let verb = text.verb();
            assert!(
                verb.len() <= 8 && verb.chars().all(|c| c.is_ascii_lowercase()),
                "`{verb}`"
            );
        }
    }
}
