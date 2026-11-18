<!-- Auto-generated - do not edit. -->

# Menu

A popup menu: a column of entries drawn over the rest of its
[Screen](Screen.md) while the boolean `open` names is true.

A [`HitRegion`](HitRegion.md) or [`KeyBinding`](KeyBinding.md) opens it by
setting or flipping that boolean, such as `{"flip": "state:file_menu"}`.
It opens at `x` and `y`, or at the cursor when `at_cursor` is set, as a
context menu does, and moves in to stay on screen. Picking an entry fires
its `action` and closes the menu; a click outside it or Escape closes it
without a pick. While it is open, it takes the clicks over its screen.

Its entries are `entries`, or the items of the list in the app's UI state
that `items` names, picking one of which sends a `UiEvent` named by
`on_pick` carrying its index.

## Parameters

- `open`: A string. The boolean that shows the menu while true: a `state:` or `var:` binding. Closing the menu sets it false. Optional.
- `entries`: An array of [MenuEntry](MenuEntry.md) objects. The entries, top to bottom. Defaults to `[]`.
- `items`: A string. A list in the app's UI state whose items are the entries instead. Defaults to `""`.
- `on_pick`: A string. The event picking an entry of `items` sends. Defaults to `""`.
- `x`: A float. Left edge of the open menu. Defaults to `0.0`.
- `y`: A float. Top edge of the open menu. Defaults to `0.0`.
- `at_cursor`: A boolean. Whether the menu opens at the cursor instead of at `x` and `y`. Defaults to `false`.
- `width`: A float. Width. Defaults to `220.0`.
- `row_height`: A float. Height of each entry. Defaults to `26.0`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the menu draws.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned menu maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) the menu opens over. Unset, it joins the world's HUD.
