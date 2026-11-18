<!-- Auto-generated - do not edit. -->

# Checkbox

A box that shows and flips a boolean, with a caption beside it and an
optional dimmer note under the caption.

A click anywhere on the row flips the boolean `checked` names and sends a
`UiEvent` named by `on_change` carrying the new value. A disabled box keeps
its value, draws dimmed, and takes no clicks; its note can say why.

## Parameters

- `label`: A string. The caption beside the box. Defaults to `""`.
- `note`: A string. Dimmer text under the caption, such as why the box is disabled. Defaults to `""`.
- `checked`: A string. The boolean the box shows and flips: a `setting:`, `state:`, or `var:` binding. Optional.
- `disabled`: A boolean. Whether the box takes no clicks and draws dimmed. Defaults to `false`.
- `x`: A float. Left edge. Defaults to `0.0`.
- `y`: A float. Top edge. Defaults to `0.0`.
- `width`: A float. Width of the clickable row. Defaults to `220.0`.
- `height`: A float. Height of the clickable row; the box is sized from it. Defaults to `26.0`.
- `on_change`: A string. The event a click sends. Defaults to `""`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the box draws.
- `visible`: A boolean. When false the box is not drawn and takes no input. Defaults to `true`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned box maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) the box belongs to. Unset, it joins the world's HUD.
