<!-- Auto-generated - do not edit. -->

# Layout

Places overlay elements in a row, a column, or a stack, so a screen does not
hand-position every piece.

Each child is sized per axis: to its content (`fit`), to a fixed size, or to
a share of the space its siblings leave (`fill`). A label fits its measured
text, so the layout reflows as live text changes width. A hidden child, or a
label with no text, takes no space, and a layout with nothing to show takes
none either. The layout owns its children's position, and the size of a
[Sprite](Sprite.md) or [`TextField`](TextField.md) it sizes; a label keeps the
size of its text.

A layout no other layout holds is a root: `width` and `height` size it, and
`anchor` and `margin` place it on its [Screen](Screen.md), the reference
canvas or the window by the screen's `space`.

A `movable` root is a floating panel: a press on the band `grab` tall
across its top drags it about its screen. A `resizable` one is sized by
dragging its right or bottom edge or its corner, never smaller than
`min_size`. Floating panels overlap: a press on one brings it in front of
the others, and the pointer reaches only the frontmost panel under it.

A layout that names an `on_dismiss` event is a transient one, such as a
palette or a picker: while it shows, a press outside it sends the event, so
the app can close it. A press outside every such layout goes no further,
so it never acts on what lies under them; one inside another of them stays
that layout's own.

A `scroll` column holds children taller than it: they scroll with the wheel
and a scrollbar along its right edge, and draw and take input only inside
it. It asks its parent for no height of its own, so it is sized to fill or
to a fixed height.

## Parameters

- `screen`: A string. [Screen](Screen.md) the layout belongs to, whose space a root is placed in. Optional.
- `direction`: A string (see [LayoutDirection](LayoutDirection.md)). How the children are arranged. Defaults to `"column"`.
- `children`: An array of [LayoutChild](LayoutChild.md) objects. The children, in order. Defaults to `[]`.
- `gap`: A float. Pixels between adjacent children. Defaults to `0.0`.
- `padding`: A float. Pixels kept clear inside every edge. Defaults to `0.0`.
- `justify`: A string (see [LayoutAlign](LayoutAlign.md)). Where the children sit along the direction when they leave space over (horizontally in a stack). Defaults to `"start"`.
- `align`: A string (see [LayoutAlign](LayoutAlign.md)). Where each child sits across the direction (vertically in a stack). Defaults to `"start"`.
- `width`: A string or single-key object (see [LayoutSize](LayoutSize.md)). A root's width. Defaults to `"fit"`.
- `height`: A string or single-key object (see [LayoutSize](LayoutSize.md)). A root's height. Defaults to `"fit"`.
- `anchor`: A string (see [LayoutAnchor](LayoutAnchor.md)). The point of its screen a root sits at. Defaults to `"top-left"`.
- `margin`: An array of 2 floats. A root's distance from the screen edges it is anchored to, `[x, y]`. Defaults to `[0.0, 0.0]`.
- `visible`: A boolean. When false, the layout and everything in it is hidden: it takes no space in its parent, and its elements neither draw nor take input. Defaults to `true`.
- `movable`: A boolean. Whether a root is dragged about its screen by the band across its top. Defaults to `false`.
- `resizable`: A boolean. Whether a root is sized by dragging its right or bottom edge. Defaults to `false`.
- `scroll`: A boolean. Whether a column scrolls children taller than itself, clipped to it. Defaults to `false`.
- `modal`: A boolean. Whether a root shown with it set takes the input of its screen and the screens layered beneath it, as a dialog does: it stacks in front of the other roots on its screen's layer and dims what is behind it, and only what it holds takes the pointer, a menu or toast click, or keeps the keyboard. Screens layered above it are untouched. Defaults to `false`.
- `on_dismiss`: A string. The event a press outside the shown layout sends. A press outside every layout naming one goes no further. Defaults to `""`.
- `grab`: A float. Height of the band across a movable root's top that drags it. Defaults to `28.0`.
- `min_size`: An array of 2 floats. The smallest a resizable root is dragged to, `[width, height]`. Defaults to `[120.0, 80.0]`.
