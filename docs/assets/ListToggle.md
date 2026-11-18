<!-- Auto-generated - do not edit. -->

# ListToggle

A column of icons at the head of a [List](List.md)'s rows, each cell
flipping one flag of its row, such as whether an item shows or is locked.

A row draws the cell when its item has toggles, the `on` icon while its
flag is set and the `off` one while it is not. With no `off` icon, an unset
cell is blank until the row is hovered, when it shows the `on` icon in the
`off` color so it can be found.

## Parameters

- `on`: A [Icon](Icon.md) object. The icon drawn while the flag is set.
- `off`: A [Icon](Icon.md) object. The icon drawn while it is not.
- `on_color`: An array of 4 floats. RGBA color of the `on` icon. Defaults to `[0.82, 0.84, 0.9, 1.0]`.
- `off_color`: An array of 4 floats. RGBA color of the `off` icon, or of the hovered hint. Defaults to `[0.48, 0.49, 0.55, 1.0]`.
- `on_toggle`: A string. The event a click on the cell sends, carrying the row's index. Defaults to `""`.
