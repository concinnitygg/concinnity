<!-- Auto-generated - do not edit. -->

# AppConfig

Names, identifies, and sizes the application.

Declare at most one `AppConfig` per world. It supplies the display name,
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

`home` chooses where the running application keeps what it writes: the
settings file, the save files, crash reports, and the shader caches. Leave
it empty and those sit beside the application's data, which is what a
portable install wants. A relative path resolves against that same content
directory, so `"state"` puts them in a `state/` subfolder; an absolute path
is used verbatim. A read-only install that sets no `home` relocates them to
a per-user directory on its own.

`max_memory_mb` and `job_threads` are `0` for "auto", where the engine sizes
both from the host machine. A non-zero value overrides that choice, clamped
to what the machine can safely give.

## Parameters

- `name`: A string. Display name of the application: the game's window title, the exported archive and executable name, and the macOS bundle display name. Defaults to `"Concinnity"`.
- `id`: A string. Reverse-DNS bundle identifier (e.g. `gg.studio.mygame`). When empty the export derives one from `name`. Defaults to `""`.
- `version`: A string. Human-readable version string (e.g. `1.0.0`). Defaults to `"0.1.0"`.
- `author`: A string. Author or studio name, recorded in the exported bundle's metadata. Defaults to `""`.
- `icon`: A string. Path to a source icon image (a square PNG, 512x512 or larger) relative to the world, used to build the platform icon at export time. Empty for no custom icon. Defaults to `""`.
- `home`: A string. Where the running application writes its settings, saves, crash reports, and shader caches. Empty means beside the application's data; a relative path resolves against that directory; an absolute path is used verbatim. Defaults to `""`.
- `max_memory_mb`: An integer. Soft ceiling on host memory the runtime aims to stay under, in mebibytes. `0` = auto (a fraction of total RAM, capped by a built-in ceiling). A non-zero value is clamped so it never exceeds what the machine can safely give. Defaults to `0`.
- `job_threads`: An integer. Worker threads for the shared job pool. `0` = auto (one per core, less one for the main thread). A non-zero value never exceeds the core count. Defaults to `0`.
