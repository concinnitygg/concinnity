<!-- Auto-generated - do not edit. -->

# Panel

A titled background container for grouping UI overlay elements.

`Panel` is a build-time shorthand: it expands into a filled, optionally
rounded background [Sprite](Sprite.md) and, when `title` is set, a
[TextLabel](TextLabel.md) heading inset from the top-left corner. Place other
overlay elements over it to frame a group (a settings card, a dialog body).

Like the other build-time UI shorthands, generated names are prefixed with
this asset's `name` (`<name>_bg`, `<name>_title`), so a panel named with a
view prefix (`pause_card`) puts its children in that [View](View.md)
(`pause`) via the `<view>_*` rule and they never clash with hand-authored
assets.

```jsonl
{"name":"pause_card","type":"Panel","args":{"title":"Paused","x":440,"y":220,"width":400,"height":280}}
```

## Parameters

- `x`: A float. Left edge of the panel in window pixels. Defaults to `0.0`.
- `y`: A float. Top edge of the panel in window pixels. Defaults to `0.0`.
- `width`: A float. Panel width in window pixels. Defaults to `400.0`.
- `height`: A float. Panel height in window pixels. Defaults to `300.0`.
- `color`: An array of 4 floats. RGBA fill of the background box, each channel in [0, 1]. Defaults to `[0.08, 0.09, 0.12, 0.96]`.
- `corner_radius`: A float. Corner rounding radius of the background box, in panel pixels. 0 keeps sharp corners. Defaults to `8.0`.
- `title`: A string. Heading text drawn at the top-left. Empty draws no heading.
- `title_font`: A string. [Font](Font.md) for the title. Empty uses the built-in font.
- `title_color`: An array of 3 floats. Linear-space RGB colour of the title text. Defaults to `[0.95, 0.95, 0.97]`.
- `title_scale`: A float. Scale applied to the title text. Defaults to `1.0`.
- `padding`: A float. Inset of the title from the panel's top-left corner, in pixels. Defaults to `16.0`.
