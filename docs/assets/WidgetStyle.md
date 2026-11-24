<!-- Auto-generated - do not edit. -->

# WidgetStyle

How a widget draws: its font, its text colors, and the fills of its
background, frame, and highlights. Every field has a default, so a widget
that sets none draws in the engine's dark UI theme.

## Parameters

- `font`: A string. The [Font](Font.md) the widget's text draws in. Unset draws with the engine's built-in face.
- `scale`: A float. Text scale relative to the font's native size. Defaults to `0.85`.
- `text`: An array of 3 floats. Linear-space RGB color of the widget's text. Defaults to `[0.9, 0.9, 0.92]`.
- `dim_text`: An array of 3 floats. Linear-space RGB color of secondary text: details, hints, and dimmed rows. Defaults to `[0.6, 0.6, 0.66]`.
- `background`: An array of 4 floats. RGBA fill behind the widget. Defaults to `[0.11, 0.11, 0.14, 0.97]`.
- `border`: An array of 4 floats. RGBA color of the hairline framing the widget. Defaults to `[0.3, 0.32, 0.4, 1.0]`.
- `hover`: An array of 4 floats. RGBA fill of the row or control under the cursor. Defaults to `[0.24, 0.27, 0.36, 0.9]`.
- `selected`: An array of 4 floats. RGBA fill of the selected row or tab. Defaults to `[0.18, 0.28, 0.46, 1.0]`.
- `accent`: An array of 4 floats. RGBA fill of an active control: a checked box, a pressed button. Defaults to `[0.26, 0.42, 0.66, 1.0]`.
- `corner_radius`: A float. Corner rounding of the widget's frame and highlights, in pixels. Defaults to `5.0`.
