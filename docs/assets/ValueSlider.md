<!-- Auto-generated - do not edit. -->

# ValueSlider

A track and handle dragged to set the number `value` names, between `min`
and `max`, with the number written at its right end.

A press on the track moves the handle there and drags it until the button
lets go, setting the number as it moves, snapped to `step` when that is
above zero. Each change sends a `UiEvent` named by `on_change` carrying the
new number, and letting go sends `on_commit` with the number it settled
on. The fill runs from zero, or from the end of the range nearer zero, to
the handle. A disabled slider keeps its value and draws dimmed.

## Parameters

- `value`: A string. The number the slider shows and sets: a `state:` or `var:` binding. Optional.
- `min`: A float. The number at the track's left end. Defaults to `0.0`.
- `max`: A float. The number at the track's right end. Defaults to `1.0`.
- `step`: A float. The increment the number snaps to; 0 sets it freely. Defaults to `0.0`.
- `decimals`: An integer. Digits after the point the number is written with. Defaults to `2`.
- `value_width`: A float. Width of the column the number is written in at the right end; 0 writes no number. Defaults to `44.0`.
- `disabled`: A boolean. Whether the slider takes no input and draws dimmed. Defaults to `false`.
- `x`: A float. Left edge. Defaults to `0.0`.
- `y`: A float. Top edge. Defaults to `0.0`.
- `width`: A float. Width, the number's column included. Defaults to `200.0`.
- `height`: A float. Height. Defaults to `24.0`.
- `on_change`: A string. The event each change sends while dragging. Defaults to `""`.
- `on_commit`: A string. The event letting go sends. Defaults to `""`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the slider draws.
- `visible`: A boolean. When false the slider is not drawn and takes no input. Defaults to `true`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned slider maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) the slider belongs to. Unset, it joins the world's HUD.
