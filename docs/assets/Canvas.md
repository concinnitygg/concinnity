<!-- Auto-generated - do not edit. -->

# Canvas

An area that draws a named list of shapes, clipped to itself, such as a
chart of cards and wires or marks laid over a view. The app writes the
shape list into its UI state at runtime.

The shapes are placed in the canvas's own space and moved by its pan. A
press on a shape with a tag sends a `UiEvent` named by `on_press` carrying
the tag, or by `on_extend` when Ctrl, Cmd, or Shift is held and it is set.
A `pannable` canvas is dragged by a press on its empty space and scrolled
by the wheel, never past its shapes, and shows a scrollbar along each edge
its shapes reach past. A `passthrough` canvas takes no input at all, so
what lies under it does.

## Parameters

- `shapes`: A string. The shape list in the app's UI state the canvas draws. Defaults to `""`.
- `on_press`: A string. The event a press on a tagged shape sends. Defaults to `""`.
- `on_extend`: A string. The event a press on a tagged shape with Ctrl, Cmd, or Shift held sends instead, when set. Defaults to `""`.
- `pannable`: A boolean. Whether a drag on empty space and the wheel pan the shapes. Defaults to `false`.
- `passthrough`: A boolean. Whether the canvas takes no input, leaving every press, drag, and wheel to what lies under it. Defaults to `false`.
- `clip`: A boolean. Whether shapes are cut off at the canvas's edges. Defaults to `true`.
- `x`: A float. Left edge. Defaults to `0.0`.
- `y`: A float. Top edge. Defaults to `0.0`.
- `width`: A float. Width. Defaults to `200.0`.
- `height`: A float. Height. Defaults to `150.0`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the canvas draws its text, and its background when that is not clear.
- `visible`: A boolean. When false the canvas is not drawn and takes no input. Defaults to `true`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned canvas maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) the canvas belongs to. Unset, it joins the world's HUD.
