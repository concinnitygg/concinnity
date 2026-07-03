<!-- Auto-generated - do not edit. -->

# StoryImport

Imports a Markdown story file as a single declaration.

One `StoryImport` stands in for a whole branching, click-through story (a
visual-novel flow). The build parses the Markdown and expands the import
into the UI assets that play it: a [View](View.md) per page with a backdrop
[Sprite](Sprite.md), [TextLabel](TextLabel.md)s for narration and speaker
names, and [HitRegion](HitRegion.md)s wiring page to page, so `world.jsonl`
stays a single readable line while the story lives in the Markdown file.

The `source` file is CommonMark Markdown opening with a YAML frontmatter
block:

- frontmatter declares the story `title` and its `characters`
- each `# heading` starts a node (a jump target)
- each paragraph is one click-through page of narration
- a paragraph opening `**id:**` attributes the line to a declared
  character, shown as a name plate in that character's color
- a bullet list of links is a choice menu; each link targets a heading
  (`[Into the wood](#the-wood)`)
- a paragraph that is a single link shows its label and jumps to its
  target when clicked
- a lone link to an audio file is a media directive: `music` loops from
  the next page onward until replaced, `sound` plays once when the next
  page shows (`[music](assets/theme.ogg)`, `[sound](assets/door.wav)`);
  these expand to [AudioClip](AudioClip.md) + [AudioCue](AudioCue.md) entries
- a node whose last page has no link falls through to the next heading
  in document order; the final node ends the story

Any other Markdown construct (images, tables, code blocks, inline
emphasis, ...) is an error at build time, as are links to headings that
do not exist, undeclared speakers, and duplicate headings.

**Generated names** are prefixed with the import's own asset `name`
(`<name>_title`, `<name>_<node>_p0`, ...), so they never clash with
hand-authored assets.

Characters take a nested block, a one-line name, or a `{ ... }` flow map
(`ayame: { name: Ayame, color: [1.0, 0.85, 0.8] }`).

```markdown
---
title: The Crossroads
characters:
  ayame:
    name: Ayame
    color: [1.0, 0.85, 0.8]
  keeper: Innkeeper
---

# inn

You wake at a roadside inn. A note rests on the pillow.

**ayame:** You came. I wasn't sure you would.

- [Into the wood](#wood)
- [Toward the shore](#shore)
```

```jsonl
{"name":"crossroads","type":"StoryImport","args":{"source":"assets/crossroads.md"}}
```

## Parameters

- `source`: A string. Path to the Markdown story file, relative to the project root.
- `title_screen`: A boolean. Whether to generate a title screen (story title, Start and Quit buttons) as the initial view. When `false`, the story's first page is the initial view and the generated ending offers a Restart instead of Back to title. Defaults to `true`.
