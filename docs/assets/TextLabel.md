<!-- Auto-generated - do not edit. -->

# TextLabel

Screen-space text drawn as a UI overlay on top of the 3D scene each frame.

Text is laid out using the referenced [Font](Font.md). The `content` field can
be updated every frame (e.g. by an [FpsCounter](FpsCounter.md)).

A `\n` in `content` starts a new line. When `background` has an alpha > 0, a
box is filled behind the glyphs, extended outward by `padding` pixels,
useful for HUD chips.

## Parameters

- `font`: A string. The [Font](Font.md) asset to use for rendering. Optional.
- `content`: A string. Text to display. Can be updated each frame.
- `x`: A float. Horizontal position in pixels from the left edge of the window. Defaults to `10.0`.
- `y`: A float. Vertical position in pixels from the top edge of the window. Defaults to `10.0`.
- `color`: An array of 3 floats. Linear-space RGB text colour. Defaults to `[1.0, 1.0, 1.0]`.
- `scale`: A float. Uniform scale applied on top of the font's `size_px`. 1.0 = native size. Defaults to `1.0`.
- `centered`: A boolean. When true, center the label in the viewport each frame; x and y are ignored. Defaults to `false`.
- `align`: A string (see [TextAlign](TextAlign.md)). Horizontal alignment relative to `x` (measured with the real font metrics). Ignored when `centered` is set.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned label maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). `Bottom` keeps a label flush with a bottom-anchored sprite it labels.
- `background`: An array of 4 floats. RGBA fill of a box drawn behind the text. An alpha of 0 (the default) draws no box; any alpha > 0 draws the box at that opacity.
- `padding`: A float. Pixels the background box extends past the text on every side. Only meaningful when `background` is visible. Defaults to `0.0`.
- `wrap_width`: A float. Width in the label's own pixels that text wraps within. `0` (the default) never wraps, so the text runs as far as it needs to. Any greater value breaks the content into lines at word boundaries, using the real font metrics, splitting a word only when it cannot fit a line on its own. Authored newlines are kept as breaks either way. Ignored when `centered` is set, since a centered label is sized to the viewport rather than to a container.
- `max_lines`: An integer. Most lines the label draws. `0` (the default) draws every line. When the text needs more than this, the last drawn line ends in an ellipsis, so text bounded by `wrap_width` is bounded in both directions and can never spill out of the box that holds it.
- `visible`: A boolean. When false, the label is hidden. Defaults to `true`.
- `screen`: A string. [Screen](Screen.md) this label belongs to. Resolved automatically from the naming convention (`<screen>_*`); you don't set this directly. `None` means the label is always visible. Optional.
