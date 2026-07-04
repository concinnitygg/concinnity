// src/assets/story_command.rs

// Runtime-only event sent by UiInputSystem when a `story:*` action fires (a
// stage click, a Space press, a choice button, a Start / Restart button). The
// story system reads these and moves through the story graph. World authors
// never declare this type directly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum StoryCommand {
    // Reset to the first node and show the stage.
    Start,
    // Advance the current page: complete a mid-reveal, else move on.
    #[default]
    Advance,
    // Pick the current choice menu's option by index.
    Choose(usize),
}
