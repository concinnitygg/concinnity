<!-- Auto-generated - do not edit. -->

# StoryScaffold

The stage scaffolding a [Story](Story.md)'s build expansion generated: the
[Screen](Screen.md)s, [Sprite](Sprite.md)s, and [TextLabel](TextLabel.md)s the
story system mutates page by page.

## Parameters

- `screen`: A string. The stage [Screen](Screen.md) the story plays inside. Optional.
- `ending`: A string. The [Screen](Screen.md) shown when the story ends. Optional.
- `bg`: A string. Backdrop [Sprite](Sprite.md). Optional.
- `left`: A string. Stage-left portrait [Sprite](Sprite.md). Optional.
- `center`: A string. Stage-center portrait [Sprite](Sprite.md). Optional.
- `right`: A string. Stage-right portrait [Sprite](Sprite.md). Optional.
- `dialog_box`: A string. Dialog box backdrop [Sprite](Sprite.md). Optional.
- `name_label`: A string. Speaker name-plate [TextLabel](TextLabel.md). Optional.
- `text_label`: A string. Dialog text [TextLabel](TextLabel.md). Optional.
- `option_boxes`: An array of strings. Choice button box [Sprite](Sprite.md)s, one per option slot.
- `options`: An array of strings. Choice button [TextLabel](TextLabel.md)s, one per option slot.
- `start_label`: A string. The title screen's Start [TextLabel](TextLabel.md). The story lays the title menu out at runtime, keeping only the buttons that apply contiguous (Continue and Load appear only when a save exists), so these labels are moved and cleared per the save state on disk. Optional.
- `quit_label`: A string. The title screen's Quit [TextLabel](TextLabel.md). Optional.
- `continue_label`: A string. The title screen's Continue [TextLabel](TextLabel.md), hidden while no save exists. Optional.
- `title`: A string. The title screen [Screen](Screen.md), returned to when the load overlay is dismissed before play started. Optional.
- `load_label`: A string. The title screen's Load [TextLabel](TextLabel.md), hidden while no slot save exists. Optional.
- `pause`: A string. The pause-menu [Screen](Screen.md) (the injected Escape overlay), shown over the stage and returned from to the stage. Unset when the world declares no pause menu.
- `settings`: A string. The settings-screen entry [Screen](Screen.md) opened by the pause menu's and the title screen's Settings items. Unset when there is no pause menu.
- `settings_label`: A string. The title screen's Settings [TextLabel](TextLabel.md), laid out with the other title buttons and hidden when there is no settings screen. Optional.
- `advance_marker`: A string. The small pulsing [Sprite](Sprite.md) shown when a fully revealed page waits for input. Optional.
- `log_label`: A string. Quick-row Log [TextLabel](TextLabel.md) (dialogue history toggle). Optional.
- `auto_label`: A string. Quick-row Auto [TextLabel](TextLabel.md) (auto-advance toggle). Optional.
- `skip_label`: A string. Quick-row Skip [TextLabel](TextLabel.md) (fast-forward toggle). Optional.
- `save_label`: A string. Quick-row Save [TextLabel](TextLabel.md) (opens the slot overlay). Optional.
- `overlay_dim`: A string. Full-canvas dim [Sprite](Sprite.md) behind the backlog and slot overlays. Optional.
- `backlog_label`: A string. The backlog overlay's history [TextLabel](TextLabel.md). Optional.
- `slot_title`: A string. The slot overlay's heading [TextLabel](TextLabel.md) ("Save" / "Load"). Optional.
- `slot_boxes`: An array of strings. Slot row box [Sprite](Sprite.md)s.
- `slot_labels`: An array of strings. Slot row [TextLabel](TextLabel.md)s.
