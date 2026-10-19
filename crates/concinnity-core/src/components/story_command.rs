/// Runtime-only event sent by `UiInputSystem` when a
/// [`UiAction::Story`](crate::components::UiAction) action fires (a stage click,
/// a Space press, a choice button, a Start / Restart button). The story system
/// reads these and moves through the story graph. World authors never declare
/// this type directly.
///
/// An action names one by its verb (`"advance"`), or by an object for the two
/// that carry an index (`{"choose": 1}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoryCommand {
    /// Reset to the first node and show the stage.
    Start,
    /// Resume from the saved position, or start fresh when no save exists.
    Continue,
    /// Advance the current page: complete a mid-reveal, else move on.
    #[default]
    Advance,
    /// Pick the current choice menu's option by index.
    Choose(usize),
    /// Toggle auto-advance (pages turn on their own once revealed).
    #[serde(rename = "auto")]
    ToggleAuto,
    /// Toggle fast-forward (instant reveal, rapid page turns; stops at menus).
    #[serde(rename = "skip")]
    ToggleSkip,
    /// Toggle the dialogue-history overlay.
    #[serde(rename = "log")]
    ToggleLog,
    /// Open the slot overlay in save mode.
    #[serde(rename = "save")]
    OpenSave,
    /// Open the slot overlay in load mode.
    #[serde(rename = "load")]
    OpenLoad,
    /// Pick a slot in the open slot overlay.
    Slot(usize),
    /// Toggle the pause menu over the stage: show it from the stage (or close an
    /// open overlay first), and return to the stage when it is up.
    #[serde(rename = "pause")]
    TogglePause,
    /// Open the settings screen, remembering the menu that opened it so it can be
    /// returned to.
    #[serde(rename = "settings")]
    OpenSettings,
    /// Close the settings screen, returning to whichever menu opened it.
    #[serde(rename = "settings_back")]
    CloseSettings,
}

impl StoryCommand {
    /// Every command's verb, one per command.
    pub const VERBS: [&'static str; 13] = [
        "start",
        "continue",
        "advance",
        "choose",
        "slot",
        "auto",
        "skip",
        "log",
        "save",
        "load",
        "pause",
        "settings",
        "settings_back",
    ];

    /// The command a verb names. `choose` and `slot` require an `index`; every
    /// other verb ignores it. `None` for an unknown verb or a missing index.
    pub fn from_verb(verb: &str, index: Option<usize>) -> Option<Self> {
        Some(match verb {
            "start" => Self::Start,
            "continue" => Self::Continue,
            "advance" => Self::Advance,
            "choose" => Self::Choose(index?),
            "slot" => Self::Slot(index?),
            "auto" => Self::ToggleAuto,
            "skip" => Self::ToggleSkip,
            "log" => Self::ToggleLog,
            "save" => Self::OpenSave,
            "load" => Self::OpenLoad,
            "pause" => Self::TogglePause,
            "settings" => Self::OpenSettings,
            "settings_back" => Self::CloseSettings,
            _ => return None,
        })
    }

    /// The verb naming this command, the inverse of [`from_verb`](Self::from_verb).
    pub const fn verb(&self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Continue => "continue",
            Self::Advance => "advance",
            Self::Choose(_) => "choose",
            Self::Slot(_) => "slot",
            Self::ToggleAuto => "auto",
            Self::ToggleSkip => "skip",
            Self::ToggleLog => "log",
            Self::OpenSave => "save",
            Self::OpenLoad => "load",
            Self::TogglePause => "pause",
            Self::OpenSettings => "settings",
            Self::CloseSettings => "settings_back",
        }
    }

    /// The option index a `choose` or `slot` command carries.
    pub const fn index(&self) -> Option<usize> {
        match self {
            Self::Choose(i) | Self::Slot(i) => Some(*i),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_verb_inverts_verb_over_every_verb() {
        for verb in StoryCommand::VERBS {
            let cmd = StoryCommand::from_verb(verb, Some(2)).unwrap();
            assert_eq!(cmd.verb(), verb);
            assert_eq!(StoryCommand::from_verb(cmd.verb(), cmd.index()), Some(cmd));
        }
    }

    // The authored form names each command by its verb, with the index as
    // the value for the two that carry one.
    #[test]
    fn the_authored_form_is_the_verb() {
        for verb in StoryCommand::VERBS {
            let cmd = StoryCommand::from_verb(verb, Some(3)).unwrap();
            let json = serde_json::to_value(&cmd).unwrap();
            let expected = match cmd.index() {
                Some(i) => serde_json::json!({ verb: i }),
                None => serde_json::json!(verb),
            };
            assert_eq!(json, expected);
            assert_eq!(serde_json::from_value::<StoryCommand>(json).unwrap(), cmd);
        }
    }

    #[test]
    fn indexed_verbs_require_an_index() {
        assert_eq!(StoryCommand::from_verb("choose", None), None);
        assert_eq!(StoryCommand::from_verb("slot", None), None);
        assert_eq!(
            StoryCommand::from_verb("start", Some(4)),
            Some(StoryCommand::Start)
        );
        assert_eq!(StoryCommand::from_verb("teleport", None), None);
    }
}
