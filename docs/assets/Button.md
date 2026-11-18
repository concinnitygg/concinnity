<!-- Auto-generated - do not edit. -->

# Button

A captioned button: a click fires its `action`.

A button can show a boolean as pressed, such as a toolbar toggle whose
panel is open, by naming it in `active`. A disabled button draws dimmed and
takes no clicks.

## Parameters

- `label`: A string. The caption. Defaults to `""`.
- `text_binding`: A string. A binding whose value the button shows in place of `label`, kept current as the value changes. Optional.
- `action`: A string or single-key object (see [Action](Action.md)). What a click fires. Unset fires nothing.
- `active`: A string. A boolean the button shows as pressed while true: a `setting:`, `state:`, or `var:` binding. Optional.
- `disabled`: A boolean. Whether the button takes no clicks and draws dimmed. Defaults to `false`.
- `x`: A float. Left edge. Defaults to `0.0`.
- `y`: A float. Top edge. Defaults to `0.0`.
- `width`: A float. Width. Defaults to `96.0`.
- `height`: A float. Height. Defaults to `26.0`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the button draws. Its idle fill is the style's `background`; a fully transparent one draws only the caption until hovered.
- `visible`: A boolean. When false the button is not drawn and takes no input. Defaults to `true`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned button maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) the button belongs to. Unset, it joins the world's HUD.
