<!-- Auto-generated - do not edit. -->

# Toasts

A stack of short-lived message cards in a corner of its
[Screen](Screen.md), newest nearest the corner, drawn over the screen's
other elements.

The app's systems send a [`Toast`] event for each message. A card shows
for its seconds and fades out; a click dismisses it. Past `max_visible`
cards, a line counts the ones waiting.

## Parameters

- `anchor`: A string (see [LayoutAnchor](LayoutAnchor.md)). The corner or edge the stack grows from. Defaults to `"bottom-right"`.
- `margin`: An array of 2 floats. Gap from the screen's edges to the stack. Defaults to `[16.0, 16.0]`.
- `width`: A float. Width of each card. Defaults to `320.0`.
- `card_height`: A float. Height of each card. Defaults to `44.0`.
- `max_visible`: An integer. The most cards shown at once. Defaults to `4`.
- `seconds`: A float. Seconds a card shows when its toast names none. Defaults to `4.0`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the cards draw.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned stack maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) the stack draws over. Unset, it joins the world's HUD.
