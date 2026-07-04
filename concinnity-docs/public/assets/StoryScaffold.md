<!-- Auto-generated - do not edit. -->

# StoryScaffold

The stage scaffolding a [Story](Story.md)'s build expansion generated: the
[View](View.md)s, [Sprite](Sprite.md)s, and [TextLabel](TextLabel.md)s the
story system mutates page by page.

## Parameters

- `view`: A string. The stage [View](View.md) the story plays inside. Optional.
- `ending`: A string. The [View](View.md) shown when the story ends. Optional.
- `bg`: A string. Backdrop [Sprite](Sprite.md). Optional.
- `left`: A string. Stage-left portrait [Sprite](Sprite.md). Optional.
- `center`: A string. Stage-center portrait [Sprite](Sprite.md). Optional.
- `right`: A string. Stage-right portrait [Sprite](Sprite.md). Optional.
- `dialog_box`: A string. Dialog box backdrop [Sprite](Sprite.md). Optional.
- `name_label`: A string. Speaker name-plate [TextLabel](TextLabel.md). Optional.
- `text_label`: A string. Dialog text [TextLabel](TextLabel.md). Optional.
- `panel`: A string. Choice menu panel [Sprite](Sprite.md). `None` when the story has no choice menus. Optional.
- `options`: An array of strings. Choice button [TextLabel](TextLabel.md)s, one per option slot.
- `continue_label`: A string. The title screen's Continue [TextLabel](TextLabel.md), hidden while no save exists. Optional.
