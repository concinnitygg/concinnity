<!-- Auto-generated - do not edit. -->

# Cloud

A layer of clouds in the sky: fair-weather cumulus, a gray stratus sheet,
towering heaps or anything between, drawn as a volume that can be seen
from below, flown through and seen from above.

The layer fills the air between `base` and `base + thickness` meters over
the ground (over the sea level of a [`Planet`](Planet.md)). Where clouds form
in it is a weather pattern whose cells are about `feature_size` meters
apart: `coverage` is the share of the sky they cover, from clear at 0
through scattered and broken clouds to overcast at 1. `towering` is how
tall each cloud grows inside the layer: 0 draws flat sheets, around 0.5
rounded heaps with flat bases, and 1 heaps that tower to the top of the
layer. Their edges are worn into wisps about `detail_size` meters across.

Clouds are lit by the sun through the air, so they are white at noon,
golden and pink at sunset, and stay lit a while after the ground below
them has gone dark, since they stand higher. Light passing through a cloud
is scattered on through it, which brightens their sunlit edges and gives
a silver lining when the sun is behind them, while the cores of thick
clouds go gray. The sky lights them from above and the ground from below.
By night a [`NightSky`](NightSky.md)'s moon lights them. The air between the
camera and a cloud hazes it like any distant surface.

Clouds shade the ground, the grass and the sea beneath them from the sun
(or the moon), soft-edged and moving with them, and they stand over the
sea as over the land. Shiny surfaces and the water reflect them, and the
sky's light on the scene is the light of a sky with them in it, so an
overcast sky dims and grays the shade.

`density` is how much a cloud dims the light per meter at its densest:
higher draws harder, brighter edged clouds and darker cores.

Clouds drift with the [`Wind`](Wind.md), at `wind_drift` times its speed,
since winds blow faster aloft, and slowly change shape, each taking about
`evolve_seconds` to become a different cloud. Without a wind they hold
still. `stretch` draws them out along the wind: a thin, high layer
stretched five times or more becomes streaks of cirrus. On a planet each
cloud stays over the same ground, wherever on it the camera roams, so the
clouds seen from orbit are the ones seen from the ground. `seed` places
them: each seed is a different sky.

An [`Atmosphere`](Atmosphere.md) draws the layers its `clouds` names, at most
two, one wholly above the other; a layer nothing names is not drawn.

## Parameters

- `base`: A float. Height of the layer's bottom over the ground, in meters. At least 0. Defaults to `1500.0`.
- `thickness`: A float. Depth of the layer from its bottom to its top, in meters. At least 10. Defaults to `1500.0`.
- `coverage`: A float. The share of the sky the clouds cover, in `[0, 1]`: 0 is a clear sky and 1 overcast. Defaults to `0.45`.
- `towering`: A float. How tall the clouds grow inside the layer, in `[0, 1]`: 0 flat sheets, 1 towers reaching the top of the layer. Defaults to `0.5`.
- `density`: A float. How much a cloud dims the light per meter at its densest. Greater than 0. Defaults to `0.04`.
- `feature_size`: A float. Typical distance between clouds, in meters. At least 100. Defaults to `4000.0`.
- `detail_size`: A float. Size of the wisps the clouds' edges are worn into, in meters. At least 1, and at most a tenth of `feature_size`. Defaults to `120.0`.
- `wind_drift`: A float. How fast the clouds drift, as a multiple of the wind's speed. At least 0. Defaults to `2.0`.
- `evolve_seconds`: A float. Seconds a cloud takes to change into a different one. 0 holds their shapes still. Defaults to `600.0`.
- `seed`: An integer. Places the clouds: each seed is a different sky. Defaults to `0`.
- `stretch`: A float. How far each cloud is drawn out along the wind, as a multiple of its width. At least 1: 1 keeps them round, and 5 or more streaks a thin, high layer into cirrus. Defaults to `1.0`.
