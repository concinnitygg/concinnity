<!-- Auto-generated - do not edit. -->

# StoryPage

One click-through page of a [StoryNode](StoryNode.md).

## Parameters

- `speaker`: A [StorySpeaker](StorySpeaker.md) object. The speaking character, shown as a name plate. `None` = narration. Optional.
- `text`: A string. The dialog text, pre-wrapped with explicit newlines.
- `jump`: An integer. Node index advancing jumps to, overriding the default next-page / fall-through order.
- `music`: A string. Music current at this page ([AudioClip](AudioClip.md) asset name). Re-triggering the already-playing track is seamless. Optional.
- `sounds`: An array of strings. One-shot effects played when the page shows.
- `stage`: A [StoryStage](StoryStage.md) object. Stage dressing current at this page.
