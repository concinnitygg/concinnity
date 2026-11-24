<!-- Auto-generated - do not edit. -->

# Tabs

A strip of tabs sharing its width, one of them selected.

A click on a tab sets the whole number `selected` names to the tab's index
and sends a `UiEvent` named by `on_select` carrying it. What each tab shows
is the app's: its screens or systems read the same value.

## Parameters

- `tabs`: An array of strings. Each tab's title, left to right. Defaults to `[]`.
- `selected`: A string. A whole number set to the selected tab's index: a `state:` or `var:` binding. Unset, the first tab shows selected.
- `x`: A float. Left edge. Defaults to `0.0`.
- `y`: A float. Top edge. Defaults to `0.0`.
- `width`: A float. Width, shared evenly between the tabs. Defaults to `300.0`.
- `height`: A float. Height. Defaults to `28.0`.
- `on_select`: A string. The event a click on a tab sends. Defaults to `""`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the tabs draw.
- `visible`: A boolean. When false the tabs are not drawn and take no input. Defaults to `true`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How screen-owned tabs map from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) the tabs belong to. Unset, they join the world's HUD.
