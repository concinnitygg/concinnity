<!-- Auto-generated - do not edit. -->

# Sprite

Screen-space 2D rectangle drawn as a UI overlay each frame.

Sprites are pixel-anchored quads with an RGBA tint. They draw alongside
[TextLabel](TextLabel.md)s, ordered behind labels so text sits on top.

A sprite with a `texture` draws that image, multiplied by the tint (use a
white tint to show the image unchanged; the tint's alpha fades it).
Without one, the tint is drawn as a solid-coloured rectangle.

```jsonl
{
  "name": "title_menu_bg",
  "type": "Sprite",
  "args": {
    "x": 0, "y": 0, "width": 1280, "height": 720,
    "tint": [0.04, 0.06, 0.10, 1.0]
  }
}
```

## Parameters

- `x`: A float. Left edge in screen pixels from the window's top-left. Defaults to `0.0`.
- `y`: A float. Top edge in screen pixels from the window's top-left. Defaults to `0.0`.
- `width`: A float. Width in screen pixels. Defaults to `100.0`.
- `height`: A float. Height in screen pixels. Defaults to `100.0`.
- `texture`: A string. [Texture](Texture.md) to draw, sampled over the sprite's rect and multiplied by `tint`. Omitted, the sprite is a solid `tint` fill.
- `tint`: An array of 4 floats. RGBA colour the rectangle is filled with, each channel in [0, 1]. Defaults to `[1.0, 1.0, 1.0, 1.0]`.
- `follow_cursor`: A boolean. When true, the sprite acts as an in-engine cursor: it is drawn on top of the other overlays as an arrow pointer tracking the mouse, with the pointer at the arrow's tip. `tint` is the arrow fill (a contrasting outline is added automatically) and `height` its size; `width` is ignored so the arrow keeps its shape. The system cursor is hidden while a visible `follow_cursor` sprite exists. Defaults to `false`.
- `visible`: A boolean. When false the sprite is skipped each frame. Defaults to `true`.
- `screen`: A string. [Screen](Screen.md) this sprite belongs to. Resolved automatically from the naming convention (`<screen>_*`); you don't set this directly. `None` means the sprite is always visible (e.g. a scene background). Optional.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned sprite maps from the reference canvas to the window when their aspect ratios differ.
- `corner_radius`: A float. Corner rounding radius in the sprite's own pixel space. `0` keeps sharp corners; larger values round each corner with a quarter-circle arc (clamped to half the sprite's shorter side). The rounded edge is softly anti-aliased. Defaults to `0.0`.
- `border_width`: A float. Border stroke width in the sprite's own pixel space, drawn just inside the sprite's outline and following its rounded corners. `0` draws no border; larger values inset the tinted fill by this width and paint the ring in `border_color` (clamped to half the sprite's shorter side). Defaults to `0.0`.
- `border_color`: An array of 4 floats. RGBA colour of the border stroke, each channel in [0, 1]. Ignored when `border_width` is `0`. Defaults to `[0.0, 0.0, 0.0, 1.0]`.
