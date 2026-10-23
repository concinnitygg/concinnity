<!-- Auto-generated - do not edit. -->

# Shadows

Cascaded shadow maps for the world's directional lights: their resolution,
how far from the camera they reach, how many cascades split that range, and
how often each is re-rendered. The `shadows` section of
[`GraphicsConfig`](GraphicsConfig.md); a field left unset takes the default
below.

The defaults describe the quality capable hardware runs, not what every GPU
runs: the `Auto` graphics quality preset resolves the detected GPU into a
ceiling that caps them tier by tier.

## Parameters

- `map_size`: An integer. Shadow map resolution in texels. `4096` by default, capped by the quality preset down to 1024 on the lowest tier. Set to 0 to disable shadows. Applies at the next launch.
- `update`: A string (see [ShadowUpdate](ShadowUpdate.md)). How often shadow cascades are re-rendered. `every_frame` (default) refreshes them all every frame; `hybrid` amortizes the far cascades across frames. Only the top quality tier permits `every_frame`, so everything below it runs `hybrid`. Applies live.
- `distance`: An integer. How far from the camera shadows are cast, in world units (e.g. 80). The cascades cover from the near plane out to this distance; a larger value shadows more of the scene but spreads the same shadow-map resolution over more area (softer, blockier shadows). Capped at the camera far plane. Defaults to `80`. Applies live.
- `cascades`: An integer. Number of shadow cascades (`4` is the default and the maximum). More cascades keep distant shadows sharper by splitting the view range into finer slices, at the cost of an extra shadow-map render per cascade; fewer is cheaper but blockier far from the camera. The slice count covers the same `distance` regardless. Clamped to `[1, 4]`. Applies live.
