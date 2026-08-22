<!-- Auto-generated - do not edit. -->

# HitRegion

A responsive invisible rectangular region in screen space.

When clicked, fires an `action`. When hovered, it optionally restyles a
referenced [TextLabel](TextLabel.md) (colour and/or scale).

The cursor must be free (not captured for camera control) for events to fire.

## Parameters

- `x`: A float. Left edge of the region in window pixels. Defaults to `0.0`.
- `y`: A float. Top edge of the region in window pixels. Defaults to `0.0`.
- `width`: A float. Width of the region in window pixels. Defaults to `100.0`.
- `height`: A float. Height of the region in window pixels. Defaults to `40.0`.
- `label`: A string. A [TextLabel](TextLabel.md) to style on hover. `None` = no label effect. Optional.
- `hover_color`: An array of 3 floats. RGB colour applied to the label while hovered. `None` = no change. Optional.
- `hover_scale`: A float. Scale applied to the label while hovered. None = no change. Optional.
- `action`: A string. Action to fire on click. Recognised forms: `"scene:<name>"`, `"quit"`, `"screen:show:<name>"`, `"screen:hide"`, `"screen:toggle:<name>"`.
- `drag_handle`: A string. The [Sprite](Sprite.md) a [Slider](Slider.md) drag region moves along its track. `None` for ordinary regions. Set automatically when a `Slider` expands; you don't set this directly. Optional.
- `screen`: A string. [Screen](Screen.md) this region belongs to. Resolved automatically from the naming convention (a region named `<screen>_*` belongs to screen `<screen>`); you don't set this directly. While a screen is active, only the top capturing screen's regions fire; with no screen active, only screen-less regions fire. Optional.
- `disabled`: A boolean. Whether this region is inert. A disabled region never hovers or fires. Set by the engine at runtime (e.g. a settings row whose feature the GPU cannot provide is disabled and grayed out); you don't set this directly. Defaults to `false`.
- `follow_label`: A boolean. When set, this region tracks its referenced [`label`](HitRegion.md): it follows the label's vertical position (so a menu the engine lays out at runtime keeps its buttons clickable) and is inert while the label's text is empty (so a hidden menu entry does not catch clicks). Requires `label`. Defaults to `false`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned region maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). `Bottom` keeps a region aligned with bottom-anchored furniture it covers. A region spanning the whole reference canvas always covers the full window regardless of `fit`.
