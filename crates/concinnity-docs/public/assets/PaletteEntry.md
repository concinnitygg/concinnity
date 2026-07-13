<!-- Auto-generated - do not edit. -->

# PaletteEntry

One entry in a [MaterialPalette](MaterialPalette.md). Each carries an `alias` (the suffix of the
expanded [Material](Material.md) name) plus the Material fields the expansion
fills in. Names in `albedo` / `normal_map` are unresolved [Texture](Texture.md)
references, resolved on the expanded Material.

## Parameters

- `alias`: A string. Alias suffix; the expanded material is named `<palette>_<alias>`.
- `albedo`: A string. [Texture](Texture.md) name for the material's albedo.
- `normal_map`: A string. [Texture](Texture.md) name for the material's normal map.
- `roughness`: A float. Surface roughness in [0, 1]. Defaults to `0.8`.
- `metallic`: A float. Metallic factor in [0, 1]. Defaults to `0.0`.
- `tint`: An array of 3 floats. Linear-space RGB tint multiplier. Defaults to `[1.0, 1.0, 1.0]`.
- `emissive_factor`: An array of 3 floats. Linear-space RGB emissive factor. Defaults to `[0.0, 0.0, 0.0]`.
