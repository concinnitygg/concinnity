<!-- Auto-generated - do not edit. -->

# HitRegion

A responsive invisible rectangular region in screen space.

When clicked, fires an `action`. When hovered, it optionally restyles a
referenced [`TextLabel`](TextLabel.md) (color and/or scale).

The cursor must be free (not captured for camera control) for events to fire.

## Parameters

- `x`: A float. Left edge of the region in window pixels. Defaults to `0.0`.
- `y`: A float. Top edge of the region in window pixels. Defaults to `0.0`.
- `width`: A float. Width of the region in window pixels. Defaults to `100.0`.
- `height`: A float. Height of the region in window pixels. Defaults to `40.0`.
- `label`: A string. A [`TextLabel`](TextLabel.md) to style on hover. `None` = no label effect. Optional.
- `hover_color`: An array of 3 floats. RGB color applied to the label while hovered. `None` = no change. Optional.
- `hover_scale`: A float. Scale applied to the label while hovered. None = no change. Optional.
- `action`: A string or single-key object (see [Action](Action.md)). What a click fires. Unset fires nothing.
- `slider`: A string. A number, 0 to 1, that dragging across the region sets to the fraction of its width under the cursor: a `setting:` slider, or a `state:` or `var:` float. The `drag_handle` sprite follows the value. Optional.
- `drag_handle`: A string. The [Sprite](Sprite.md) that slides along the region to show its `slider` value. Optional.
- `screen`: A string. [Screen](Screen.md) this region belongs to. While a capturing screen is open, only the topmost one's regions fire; with none open, persistent screens' regions do. Unset, the region joins the world's HUD: a persistent window-space screen made for the elements that name none.
- `disabled`: A boolean. Whether this region is inert. A disabled region never hovers or fires. Set by the engine at runtime (e.g. a settings row whose feature the GPU cannot provide is disabled and grayed out); you don't set this directly. Defaults to `false`.
- `follow_label`: A boolean. When set, this region tracks its referenced [`label`](HitRegion.md): it moves with the label (so a menu the engine lays out at runtime, or a panel the user drags, keeps its buttons clickable) and is inert while the label's text is empty (so a hidden menu entry does not catch clicks). Requires `label`. Defaults to `false`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned region maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). `Bottom` keeps a region aligned with bottom-anchored furniture it covers. A region spanning the whole reference canvas always covers the full window regardless of `fit`. Defaults to `"fit"`.
