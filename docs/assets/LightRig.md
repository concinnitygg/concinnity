<!-- Auto-generated - do not edit. -->

# LightRig

A named grouping of lights.

Use `preset` to expand a built-in setup into named
[DirectionalLight](DirectionalLight.md)/[PointLight](PointLight.md) assets
(`<rig_name>_<light_name>`), or declare lights directly and list their names
in `lights`.

**Library presets:**

## Parameters

- `preset`: A string. Name of a built-in or file-backed preset (e.g. "rig_outdoor_sun_fill"). When set, `lights` is ignored.
- `lights`: An array of strings. Names of existing [DirectionalLight](DirectionalLight.md) or [PointLight](PointLight.md) assets to include in this rig. Ignored when `preset` is set.
