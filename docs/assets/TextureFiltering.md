<!-- Auto-generated - do not edit. -->

# TextureFiltering

How the scene's surface textures (albedo and normal maps) are filtered. The
`texture_filtering` section of [`GraphicsConfig`](GraphicsConfig.md); a field
left unset takes the default below, capped by the quality tier.

The default describes the quality capable hardware runs: the `Auto`
graphics quality preset resolves the detected GPU into a ceiling that caps
it tier by tier.

## Parameters

- `anisotropy`: An integer. Maximum anisotropic-filtering degree, e.g. 8. Higher keeps textures viewed at a grazing angle (floors, walls receding into the distance) sharp instead of blurring along the minor axis, at a small sampling cost. `1` disables anisotropy (plain trilinear). `16` by default, capped by the quality preset down to 4 on the lowest tier. Clamped to `[1, 16]`. Applies at the next launch.
