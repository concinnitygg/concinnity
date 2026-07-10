<!-- Auto-generated - do not edit. -->

# TextInput

An editable single-line text field drawn as a UI overlay.

A filled rounded box showing the typed `content` (or a dimmer `placeholder`
while empty), plus a caret when the field holds keyboard focus. The engine
gives focus to the field the cursor clicks, appends the characters typed that
frame, and moves or edits at the caret with the arrow / Home / End /
Backspace / Delete keys. Read `content` back to use what the player typed;
set it to pre-fill the field.

Like other overlay elements it belongs to a [View](View.md) resolved from the
naming convention (`<view>_*`), or is always shown when it has none.

```jsonl
{
  "type": "TextInput",
  "name": "menu_playername",
  "args": {
    "font": "ui_font",
    "placeholder": "Enter your name",
    "x": 400, "y": 300, "width": 480, "height": 48,
    "max_len": 24
  }
}
```

## Parameters

- `font`: A string. The [Font](Font.md) used to render the field's text. Optional.
- `content`: A string. The current text. Edited in place as the player types; set an initial value here to pre-fill the field.
- `placeholder`: A string. Dimmer prompt shown while `content` is empty and the field is unfocused.
- `x`: A float. Left edge in screen pixels from the window's top-left. Defaults to `0.0`.
- `y`: A float. Top edge in screen pixels from the window's top-left. Defaults to `0.0`.
- `width`: A float. Field width in screen pixels. Defaults to `240.0`.
- `height`: A float. Field height in screen pixels. Defaults to `40.0`.
- `scale`: A float. Uniform scale applied on top of the font's `size_px`. 1.0 = native size. Defaults to `1.0`.
- `text_color`: An array of 3 floats. Linear-space RGB colour of the typed text. Defaults to `[0.95, 0.95, 0.97]`.
- `placeholder_color`: An array of 3 floats. Linear-space RGB colour of the placeholder prompt. Defaults to `[0.55, 0.55, 0.60]`.
- `background`: An array of 4 floats. RGBA fill of the field's background box, each channel in [0, 1]. Defaults to `[0.10, 0.10, 0.13, 1.0]`.
- `caret_color`: An array of 3 floats. Linear-space RGB colour of the caret bar. Defaults to `[0.95, 0.95, 0.97]`.
- `corner_radius`: A float. Corner rounding radius of the background box, in field pixels. Defaults to `4.0`.
- `padding`: A float. Inner horizontal inset from the box edge to the text, in pixels. Defaults to `8.0`.
- `max_len`: An integer. Maximum number of characters accepted. 0 means no limit. Defaults to `0`.
- `visible`: A boolean. When false the field is skipped each frame and cannot take focus. Defaults to `true`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a view-owned field maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`).
- `view`: A string. [View](View.md) this field belongs to. Resolved automatically from the naming convention (`<view>_*`); you don't set this directly. `None` means the field is always visible. Optional.
