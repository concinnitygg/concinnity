<!-- Auto-generated - do not edit. -->

# Font

Rasterises a TrueType font into a glyph atlas at build time.

Reference a Font by name from a [TextLabel](TextLabel.md). Declaring one is
optional: text naming no Font draws with the engine's built-in face at 24px,
and compiles no atlas at all. Declare a Font to pick the face, or to pick the
size the glyphs are rasterised at.

An empty `path` rasterises that same built-in face, which is how to get it at
a different `size_px`.

## Parameters

- `path`: A string. Path to the TTF file, relative to the project root.
- `size_px`: An integer. Rasterisation size in pixels. Determines the rendered glyph height. Defaults to `20`.
