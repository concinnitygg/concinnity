<!-- Auto-generated - do not edit. -->

# Grid

A scrolling grid of the items of a list in the app's
[`UiStateValues`](UiStateValues.md), as many columns of cells as its width
holds, drawing only the cells in view.

Each item draws as a cell of its image over its text, the text in the
item's color; an item with no image shows its detail in the image's place,
in its detail color. The wheel
scrolls the rows of cells and a scrollbar shows where the view sits.

A click selects a cell, setting the whole number `selected` names to its
index, -1 for none, and a double click activates it; a right click asks
for its context menu, and a Shift, Ctrl, or Cmd click sends `on_extend` in
place of `on_select`, so the app can keep a selection of many items it
marks `selected`. A press dragged off a cell sends `on_drag`. Each event
carries the cell's index. With a `selected` binding and the grid last
clicked, the arrow keys move the selection and Enter activates it.

## Parameters

- `items`: A string. The list in the app's UI state whose items the cells show. Defaults to `""`.
- `selected`: A string. A whole number set to the selected cell's index, -1 for none: a `state:` or `var:` binding. Unset, cells select only by the events they send.
- `x`: A float. Left edge. Defaults to `0.0`.
- `y`: A float. Top edge. Defaults to `0.0`.
- `width`: A float. Width. Defaults to `400.0`.
- `height`: A float. Height of the view; rows of cells past it scroll. Defaults to `360.0`.
- `cell_width`: A float. Width of each cell. Defaults to `92.0`.
- `cell_height`: A float. Height of each cell, its text included. Defaults to `112.0`.
- `gap`: A float. Space between neighboring cells. Defaults to `6.0`.
- `on_select`: A string. The event a click on a cell sends. Defaults to `""`.
- `on_activate`: A string. The event a double click or Enter on a cell sends. Defaults to `""`.
- `on_context`: A string. The event a right click on a cell sends. Defaults to `""`.
- `on_extend`: A string. The event a Shift, Ctrl, or Cmd click on a cell sends in place of `on_select`. Unset, such a click selects as a plain one does. Defaults to `""`.
- `on_drag`: A string. The event a press on a cell sends once the pointer moves a few pixels with the button held. Defaults to `""`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the grid draws.
- `visible`: A boolean. When false the grid is not drawn and takes no input. Defaults to `true`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned grid maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) this grid belongs to. Unset, it joins the world's HUD.
