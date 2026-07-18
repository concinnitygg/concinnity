<!-- Auto-generated - do not edit. -->

# Application

Runtime half of the Application asset: the process resource budgets.

## Parameters

- `name`: A string. Display name of the application: the game's window title, the exported archive and executable name, and the macOS bundle display name. Defaults to `"Concinnity"`.
- `id`: A string. Reverse-DNS bundle identifier (e.g. `gg.studio.mygame`). When empty the export derives one from `name`.
- `version`: A string. Human-readable version string (e.g. `1.0.0`). Defaults to `"0.1.0"`.
- `author`: A string. Author or studio name, recorded in the exported bundle's metadata.
- `icon`: A string. Path to a source icon image (a square PNG, 512x512 or larger) relative to the world, used to build the platform icon at export time. Empty for no custom icon.
- `limits`: A [AppLimits](AppLimits.md) object. Optional overrides for the runtime's process resource budgets. When omitted (or left at their defaults) the engine sizes both from the host machine.
