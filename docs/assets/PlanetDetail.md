<!-- Auto-generated - do not edit. -->

# PlanetDetail

How finely a planet's terrain tiles are drawn. The `planet_detail` section
of [`GraphicsConfig`](GraphicsConfig.md); a field left unset takes the default
below.

The default describes the detail capable hardware runs: the `Auto`
graphics quality preset resolves the detected GPU into a ceiling that
coarsens it tier by tier.

## Parameters

- `finest_cell_size`: An integer. How wide a planet's terrain cells may be under the camera at the finest, in meters: the quadtree stops dividing tiles once their cells are no wider. Each level halves the cells, so the finest land between half this and this. Larger builds and draws far fewer tiles near the camera. `1` by default, coarsened by the quality preset up to 4 on the lowest tier. Clamped to `[1, 64]`. Applies live.
