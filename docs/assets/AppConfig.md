<!-- Auto-generated - do not edit. -->

# AppConfig

Runtime half of the AppConfig asset: where the application keeps what it
writes, and its process resource budgets.

## Parameters

- `name`: A string. Display name of the application: the game's window title, the exported archive and executable name, and the macOS bundle display name. Defaults to `"Concinnity"`.
- `id`: A string. Reverse-DNS bundle identifier (e.g. `gg.studio.mygame`). When empty the export derives one from `name`.
- `version`: A string. Human-readable version string (e.g. `1.0.0`). Defaults to `"0.1.0"`.
- `author`: A string. Author or studio name, recorded in the exported bundle's metadata.
- `icon`: A string. Path to a source icon image (a square PNG, 512x512 or larger) relative to the world, used to build the platform icon at export time. Empty for no custom icon.
- `home`: A string. Where the running application writes its settings, saves, crash reports, and shader caches. Empty means beside the application's data; a relative path resolves against that directory; an absolute path is used verbatim.
- `max_memory_mb`: An integer. Soft ceiling on host memory the runtime aims to stay under, in mebibytes. `0` = auto (a fraction of total RAM, capped by a built-in ceiling). A non-zero value is clamped so it never exceeds what the machine can safely give. Defaults to `0`.
- `job_threads`: An integer. Worker threads for the shared job pool. `0` = auto (one per core, less one for the main thread). A non-zero value never exceeds the core count. Defaults to `0`.
