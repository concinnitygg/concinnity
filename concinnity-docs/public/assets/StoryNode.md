<!-- Auto-generated - do not edit. -->

# StoryNode

One jump target in a [Story](Story.md): a run of pages optionally ending in
a choice menu.

## Parameters

- `slug`: A string. The heading slug this node was compiled from (diagnostics only).
- `pages`: An array of [StoryPage](StoryPage.md) objects. The click-through pages, in order.
- `choices`: An array of [StoryChoice](StoryChoice.md) objects. The choice menu shown after the last page. Empty = no menu.
- `choice_stage`: A [StoryStage](StoryStage.md) object. Stage dressing current at the choice menu.
- `choice_music`: A string. Music current at the choice menu ([AudioClip](AudioClip.md) asset name). Optional.
- `choice_sounds`: An array of strings. One-shots played when the choice menu shows.
