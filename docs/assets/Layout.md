<!-- Auto-generated - do not edit. -->

# Layout

Places overlay elements in a row, a column, or a stack, so a screen does not
hand-position every piece.

Each child is sized per axis: to its content (`fit`), to a fixed size, or to
a share of the space its siblings leave (`fill`). A label fits its measured
text, so the layout reflows as live text changes width. A hidden child, or a
label with no text, takes no space, and a layout with nothing to show takes
none either. The layout owns its children's position, and the size of a
[Sprite](Sprite.md) or [`TextInput`](TextInput.md) it sizes; a label keeps the
size of its text.

A layout no other layout holds is a root: `width` and `height` size it, and
`anchor` and `margin` place it on its [Screen](Screen.md), the reference
canvas or the window by the screen's `space`.

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
- `visible`: A boolean. When false, the layout leaves its children where they are and takes no space in its parent. Defaults to `true`.
