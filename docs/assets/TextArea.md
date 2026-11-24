<!-- Auto-generated - do not edit. -->

# TextArea

A multi-line text editor: line numbers in a gutter, a scrolling window of
the text, a caret and selection, and scrollbars.

A click focuses it and places the caret; a drag selects, a double click a
word, a triple click a line. While focused it takes the keyboard: typing,
the arrows (by word with Option on macOS, Control elsewhere), Home and End,
Page Up and Down, Tab and Shift+Tab to indent, undo and redo, select all,
and cut, copy, and paste. Escape or a click outside it gives the keyboard
back.

It starts holding `content`, or the text `text` names when set, and writes
every edit back to that binding; the app's systems can also set the
binding to replace the text. Each edit sends a `UiEvent` named by
`on_change`, and the save shortcut one named by `on_save`.

## Parameters

- `content`: A string. The text it starts with when `text` is unset. Defaults to `""`.
- `text`: A string. The text it shows and edits: a `state:` or `var:` binding. Optional.
- `language`: A string (see [TextLanguage](TextLanguage.md)). The language its text is colored as. Defaults to `"plain"`.
- `read_only`: A boolean. Whether the text can be selected and copied but not edited. Defaults to `false`.
- `x`: A float. Left edge. Defaults to `0.0`.
- `y`: A float. Top edge. Defaults to `0.0`.
- `width`: A float. Width. Defaults to `480.0`.
- `height`: A float. Height. Defaults to `240.0`.
- `on_change`: A string. The event an edit sends. Defaults to `""`.
- `on_save`: A string. The event the save shortcut sends while the editor holds the keyboard. Defaults to `""`.
- `style`: A [WidgetStyle](WidgetStyle.md) object. How the editor draws. Its font is best a monospace one.
- `visible`: A boolean. When false the editor is not drawn and takes no input. Defaults to `true`.
- `fit`: A string (see [SpriteFit](SpriteFit.md)). How a screen-owned editor maps from the reference canvas to the window when their aspect ratios differ (matches [Sprite](Sprite.md)'s `fit`). Defaults to `"fit"`.
- `screen`: A string. [Screen](Screen.md) the editor belongs to. Unset, it joins the world's HUD.
