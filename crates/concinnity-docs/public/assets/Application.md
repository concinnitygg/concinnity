<!-- Auto-generated - do not edit. -->

# Application

Names and identifies the application for distribution.

Declare at most one `Application` per world. It supplies the display name,
bundle identifier, version, author, and icon that the export step reads when
it packages the world into a distributable game (the archive and executable
name, and on macOS the `.app` bundle metadata and icon). When the world
declares no [Window](Window.md) title of its own, `name` also fills the window
title, so a running game shows its own name in the title bar.

`icon` is a path to a source image (a square PNG, 512x512 or larger)
relative to the world; it is read by the packaging step and is not compiled
into the world's data. `id` is a reverse-DNS bundle identifier
(e.g. `gg.studio.mygame`); when left empty the export derives one from
`name`. Empty string fields mean "unset".

```jsonl
{"name":"app","type":"Application","args":{"name":"My Game","id":"gg.studio.mygame","version":"1.0.0","author":"Studio","icon":"art/icon.png"}}
```

## Parameters

- `name`: A string. Display name of the application: the game's window title, the exported archive and executable name, and the macOS bundle display name. Defaults to `"Concinnity"`.
- `id`: A string. Reverse-DNS bundle identifier (e.g. `gg.studio.mygame`). When empty the export derives one from `name`.
- `version`: A string. Human-readable version string (e.g. `1.0.0`). Defaults to `"0.1.0"`.
- `author`: A string. Author or studio name, recorded in the exported bundle's metadata.
- `icon`: A string. Path to a source icon image (a square PNG, 512x512 or larger) relative to the world, used to build the platform icon at export time. Empty for no custom icon.
- `limits`: A [AppLimits](AppLimits.md) object. Optional overrides for the runtime's process resource budgets. When omitted (or left at their defaults) the engine sizes both from the host machine.
