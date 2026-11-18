<!-- Auto-generated - do not edit. -->

# List

A scrolling list of the items of a list in the app's
[`UiStateValues`](UiStateValues.md), drawing only the rows in view, so it
stays cheap at thousands of items.

The app's systems fill the list `items` names; each item draws as a row of
its text, with its dimmer detail at the row's right end. Items with a
`depth` indent, and an expandable one draws an expander, so a tree is a list
the app flattens. The wheel scrolls the rows and a scrollbar shows where the
view sits.

A click selects a row, setting the whole number `selected` names to its
index, -1 for none. A double click activates it, a click on the expander
expands or collapses it, and a right click asks for its context menu: each
sends a `UiEvent` named by `on_select`, `on_activate`, `on_expand`, or
`on_context`, carrying the row's index. With a `selected` binding and the
list last clicked, the Up and Down keys move the selection and Enter
activates it.

An app that selects many items at once marks them `selected` and reads
`on_extend` for the modified clicks that add to the selection. A row can
lead with an image and with the list's toggle columns, and a press dragged
off a row sends `on_drag`.

## Parameters

- `items`: A string. The list in the app's UI state whose items the rows show. Defaults to `""`.
- `selected`: A string. A whole number set to the selected row's index, -1 for none: a `state:` or `var:` binding. Unset, rows do not select.
- `x`: A float. Left edge. Defaults to `0.0`.
- `y`: A float. Top edge. Defaults to `0.0`.
- `width`: A float. Width. Defaults to `240.0`.
- `height`: A float. Height of the view; rows past it scroll. Defaults to `320.0`.
- `row_height`: A float. Height of each row. Defaults to `26.0`.
- `indent`: A float. How far each level of a tree indents its rows. Defaults to `16.0`.
- `on_select`: A string. The event a click on a row sends. Defaults to `""`.
- `on_activate`: A string. The event a double click or Enter on a row sends. Defaults to `""`.
- `on_expand`: A string. The event a click on a row's expander sends. Defaults to `""`.
- `on_context`: A string. The event a right click on a row sends. Defaults to `""`.
- `on_extend`: A string. The event a Shift, Ctrl, or Cmd click on a row sends in place of `on_select`, so the app can add the row to a selection of many. Unset, such a click selects as a plain one does. Defaults to `""`.
- `on_drag`: A string. The event a press on a row sends once the pointer moves a few pixels with the button held, so the app can drag the row's item out. Defaults to `""`.
- `toggles`: An array of [ListToggle](ListToggle.md) objects. Icon columns at the head of the rows, each flipping a flag of its row: at most 8, and building the world drops any past it. Defaults to `[]`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the list draws.
- `visible`: A boolean. When false the list is not drawn and takes no input. Defaults to `true`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned list maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) this list belongs to. Unset, it joins the world's HUD.
