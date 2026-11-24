<!-- Auto-generated - do not edit. -->

# Modal

A dialog drawn centered over a dimmed [Screen](Screen.md) while the boolean
`open` names is true: a title, a wrapped message, and a row of buttons.

While it is open the rest of its screen takes no input. Pressing a button
fires its `action` and closes the dialog; Escape closes it without one.
Closing sets the `open` boolean false.

## Parameters

- `open`: A string. The boolean that shows the dialog while true: a `state:` or `var:` binding. Closing the dialog sets it false. Optional.
- `title`: A string. The heading. Defaults to `""`.
- `message`: A string. The text under the heading, wrapped to the dialog's width. The app's systems may set it before opening the dialog. Defaults to `""`.
- `buttons`: An array of [ModalButton](ModalButton.md) objects. The buttons, left to right. Defaults to `[]`.
- `width`: A float. Width of the dialog. Defaults to `420.0`.
- `dim`: An array of 4 floats. RGBA dim drawn over the rest of the screen while the dialog is open. Defaults to `[0.0, 0.0, 0.0, 0.45]`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the dialog draws.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned dialog maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) the dialog opens over. Unset, it joins the world's HUD.
