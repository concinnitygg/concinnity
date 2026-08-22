<!-- Auto-generated - do not edit. -->

# MaterialPalette

A named set of [Material](Material.md) entries with short aliases.

Expands into [Material](Material.md) assets named `<palette_name>_<alias>`.
[Prop](Prop.md)s reference the expanded names.

## Parameters

- `preset`: A string. Name of a built-in or file-backed preset (e.g. "pal_stone_dungeon"). When set, `entries` is ignored.
- `entries`: An array of [PaletteEntry](PaletteEntry.md) objects. Inline material entries. Ignored when `preset` is set.
