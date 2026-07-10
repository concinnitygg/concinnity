<!-- Auto-generated - do not edit. -->

# Story

A compiled branching story graph, played at runtime by the story system.

A `Story` is normally produced by a [StoryImport](StoryImport.md) expansion
at build time rather than written by hand: the Markdown source compiles
into this graph plus the stage scaffolding (a single dialogue
[View](View.md) whose labels and sprites the story system mutates page by
page). All references are pre-resolved: dialog text is pre-wrapped,
speakers carry their display name and color, stage images carry their
on-canvas rectangle, and jump / choice targets are node indices into
`nodes`.

The story system reads the graph and drives the stage view named
`<name>_stage`: it fills the dialogue and name-plate labels (revealing
text at `text_speed`), swaps the backdrop and portrait sprite textures,
shows the choice menu when a node ends in one, and plays page audio.
Clicking the stage (or pressing Space) advances; `story:start` restarts
from the first node.

## Parameters

- `title`: A string. The story title, as shown on the generated title screen.
- `nodes`: An array of [StoryNode](StoryNode.md) objects. The node graph in document order. Play starts at the first node; a node whose last page has no jump and no choices falls through to the next node, and the last node ends the story.
- `text_speed`: A float. Dialogue reveal speed in characters per second. `0` shows each page instantly. Defaults to `45.0`.
- `scaffold`: A [StoryScaffold](StoryScaffold.md) object. The generated stage assets the story system drives. All references are resolved to ids at build time, like every other cross-reference.
- `save_key`: A string. Stable key naming this story's save file (position + flags, auto-saved page by page under the project data directory). Empty disables saving.
