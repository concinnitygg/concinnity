<!-- Auto-generated - do not edit. -->

# RectAreaLight

A rectangular area light: a glowing panel that lights the scene from its
whole surface rather than from a single point.

Unlike a [PointLight](PointLight.md) or [SpotLight](SpotLight.md), the softness of
the shadow terminator and the shape of the specular highlight follow the
panel's real dimensions, so a wide softbox wraps light around a surface and
leaves a stretched rectangular reflection on glossy materials. Use it for
windows, ceiling panels, screens, and practical lights.

The panel is positioned by `centre`, oriented by `normal` (the direction it
emits), and sized by `half_size`, matching [GlassPanel](GlassPanel.md).

## Parameters

- `centre`: An array of 3 floats. World-space position of the panel's centre. Defaults to `[0.0, 3.0, 0.0]`.
- `normal`: An array of 3 floats. Direction the panel emits. Normalised on load; defaults to `+Z` when degenerate.
- `half_size`: An array of 2 floats. Half-width and half-height of the panel, in world units. Defaults to `[1.0, 1.0]`.
- `color`: An array of 3 floats. Linear-space RGB colour of the light. Defaults to `[1.0, 1.0, 1.0]`.
- `intensity`: A float. Intensity multiplier applied to the colour. Defaults to `12.0`.
- `range`: A float. Maximum reach in world units; attenuation is zero at this distance. Defaults to `18.0`.
- `two_sided`: A boolean. When true the panel emits from both faces. A one-sided panel lights only the half-space its `normal` points into. Defaults to `false`.
