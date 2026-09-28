<!-- Auto-generated - do not edit. -->

# StoryCommand

What an [Action](Action.md) does to the story. An index-carrying
command is an object with its name as the only key, e.g. `{"choose": 1}`.

## Values

- `"start"`: Reset to the first node and show the stage.
- `"continue"`: Resume from the saved position, or start fresh when no save exists.
- `"advance"`: Advance the current page: complete a mid-reveal, else move on.
- `{"choose": <integer>}`: Pick the current choice menu's option by index.
- `{"slot": <integer>}`: Pick a slot in the open slot overlay.
- `"auto"`: Toggle auto-advance.
- `"skip"`: Toggle fast-forward, which stops at menus.
- `"log"`: Toggle the dialogue-history overlay.
- `"save"`: Open the slot overlay in save mode.
- `"load"`: Open the slot overlay in load mode.
- `"pause"`: Toggle the pause menu over the stage.
- `"settings"`: Open the settings screen, remembering the menu that opened it.
- `"settings_back"`: Close the settings screen, returning to the menu that opened it.
