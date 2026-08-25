// src/components/story_command.rs

/// Runtime-only event sent by UiInputSystem when a `story:*` action fires (a
/// stage click, a Space press, a choice button, a Start / Restart button). The
/// story system reads these and moves through the story graph. World authors
/// never declare this type directly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
    ToggleAuto,
    /// Toggle fast-forward (instant reveal, rapid page turns; stops at menus).
    ToggleSkip,
    /// Toggle the dialogue-history overlay.
    ToggleLog,
    /// Open the slot overlay in save mode.
    OpenSave,
    /// Open the slot overlay in load mode.
    OpenLoad,
    /// Pick a slot in the open slot overlay.
    Slot(usize),
    /// Toggle the pause menu over the stage: show it from the stage (or close an
    /// open overlay first), and return to the stage when it is up.
    TogglePause,
    /// Open the settings screen, remembering the menu that opened it so it can be
    /// returned to.
    OpenSettings,
    /// Close the settings screen, returning to whichever menu opened it.
    CloseSettings,
}
