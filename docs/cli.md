<!-- Auto-generated - do not edit. -->

# Command line

`concinnity <COMMAND> --help` prints a command's section in a terminal.

## Options every command takes

- `-q, --quiet`: Do not print log messages.

## `concinnity init`

Create a new app in the current directory

Writes a starter world to `world.jsonl` and cooks it. A directory that
already holds a world is left alone.

```sh
concinnity init
```

Usage: `concinnity init [OPTIONS]`

## `concinnity new`

Create a new app in a new directory

Creates the directory, writes a starter world to `world.jsonl` in it and
cooks it. A directory that already holds a world is refused.

```sh
concinnity new my-app
```

Usage: `concinnity new [OPTIONS] <PATH>`

- `<PATH>`: Directory to create the project in.

## `concinnity cook`

Cook an app into binary blobs

Compiles the world into the blobs a run loads and writes
`world-lock.json`, showing each step as it runs and the warnings the
build raises.

```sh
concinnity cook
concinnity cook -f arena.jsonl --verbose
```

Usage: `concinnity cook [OPTIONS]`

- `-f, --file <FILE>`: Path to a world JSONL file. Defaults to the nearest `world.jsonl` up from the current directory.
- `-v, --verbose`: Also show the cook's informational messages, such as each imported mesh.

## `concinnity check`

Validate an app without cooking it

```sh
concinnity check
concinnity check -f arena.jsonl
```

Usage: `concinnity check [OPTIONS]`

- `-f, --file <FILE>`: Path to a world JSONL file. Defaults to the nearest `world.jsonl` up from the current directory.

## `concinnity run`

Run an app

Without `-f`, runs the cooked blobs, cooking the discovered world first
when there are none. With `-f`, cooks the named world in memory and
reloads it as its files change. `--debug-port` serves the debug
protocol (MCP) either way.

```sh
concinnity run
concinnity run -f arena.jsonl --debug-port
concinnity run --graphics-api vulkan --graphics-quality high --no-vsync
```

Usage: `concinnity run [OPTIONS]`

- `-f, --file <FILE>`: Path to a world JSONL file. Defaults to the nearest `world.jsonl` up from the current directory.
- `--graphics-quality <PRESET>`: Force the graphics-quality preset for this run. One of `auto`, `low`, `medium`, `high`, `ultra`, `custom`.
- `--graphics-api <API>`: Render with this graphics API. One of `metal`, `directx`, `vulkan`.
- `--window-mode <MODE>`: Open the window in this mode. One of `windowed`, `fullscreen`, `borderless`.
- `--[no-]graphics-validation`: Turn the graphics API's validation layer on or off (default: on in debug builds).
- `--[no-]upscaling`: Turn temporal upscaling on or off.
- `--rt-dynamic <MODE>`: How the ray-tracing acceleration structure tracks moving props. One of `off`, `auto`, `rebuild`, `tlas`.
- `--[no-]rt-reflections`: Turn ray-traced reflections on or off.
- `--[no-]rt-skinned-geometry`: Put skinned meshes in the ray-tracing acceleration structure or leave them out.
- `--[no-]vsync`: Turn vertical sync on or off.
- `--[no-]hud`: Start with the stats HUD shown or hidden (a toggle_hud key still flips it).
- `--dlss-preset <PRESET>`: Force the DLSS render preset. One of `default`, `k`, `l`, `m`.
- `--fixed-frame-rate <RATE>`: Step the frame clock by exactly 1/RATE seconds per frame, whatever the wall time.
- `--screenshots-dir <DIR>`: Write screenshots under DIR (default `screenshots`).
- `--hide-pass <PASS>`: Start with a feature pass off (repeatable). One of `shadows`, `fog`, `bloom`, `ssgi`, `reflections`, `lines`, `reactive`.
- `--limit-frames <N>`: Stop after N frames.
- `--limit-fps <N>`: Cap the frame rate at N frames a second (0 leaves it uncapped).
- `--debug-port [<PORT>]`: Serve the debug protocol on this localhost port (default 8777).

## `concinnity bench`

Benchmark an app and print what its frames cost

Finds its world the way `run` does, and prints the frame costs when the
run ends.

```sh
concinnity bench
concinnity bench --start-at 10 --stop-at 40 --screenshot-at 20:mid.png
```

Usage: `concinnity bench [OPTIONS]`

- `-f, --file <FILE>`: Path to a world JSONL file. Defaults to the nearest `world.jsonl` up from the current directory.
- `--graphics-quality <PRESET>`: Force the graphics-quality preset for this run. One of `auto`, `low`, `medium`, `high`, `ultra`, `custom`.
- `--graphics-api <API>`: Render with this graphics API. One of `metal`, `directx`, `vulkan`.
- `--window-mode <MODE>`: Open the window in this mode. One of `windowed`, `fullscreen`, `borderless`.
- `--[no-]graphics-validation`: Turn the graphics API's validation layer on or off (default: on in debug builds).
- `--[no-]upscaling`: Turn temporal upscaling on or off.
- `--rt-dynamic <MODE>`: How the ray-tracing acceleration structure tracks moving props. One of `off`, `auto`, `rebuild`, `tlas`.
- `--[no-]rt-reflections`: Turn ray-traced reflections on or off.
- `--[no-]rt-skinned-geometry`: Put skinned meshes in the ray-tracing acceleration structure or leave them out.
- `--[no-]vsync`: Turn vertical sync on or off.
- `--[no-]hud`: Start with the stats HUD shown or hidden (a toggle_hud key still flips it).
- `--dlss-preset <PRESET>`: Force the DLSS render preset. One of `default`, `k`, `l`, `m`.
- `--fixed-frame-rate <RATE>`: Step the frame clock by exactly 1/RATE seconds per frame, whatever the wall time.
- `--screenshots-dir <DIR>`: Write screenshots under DIR (default `screenshots`).
- `--hide-pass <PASS>`: Start with a feature pass off (repeatable). One of `shadows`, `fog`, `bloom`, `ssgi`, `reflections`, `lines`, `reactive`.
- `--limit-frames <N>`: Stop after N frames.
- `--limit-fps <N>`: Cap the frame rate at N frames a second (0 leaves it uncapped).
- `--debug-port [<PORT>]`: Serve the debug protocol on this localhost port (default 8777).
- `--start-at <SECONDS>`: Run the simulation ahead at the most ticks a frame allows until SECONDS into the run.
- `--stop-at <SECONDS>`: Stop the run SECONDS into it.
- `--screenshot-at <SECONDS[:FILE]>`: Save the frame SECONDS into the run as FILE under the screenshot directory (repeatable).

## `concinnity edit`

Edit an app in-engine with a save-back HUD

Cooks the world in memory, so it needs no `concinnity cook` first and
neither reads nor refreshes the cooked blobs. The editor HUD saves edits
back to the world file and adds assets. `--debug-port` serves the same
debug protocol `run` does, so an MCP client can drive the editor.

```sh
concinnity edit
concinnity edit -f arena.jsonl --debug-port 8777
```

Usage: `concinnity edit [OPTIONS]`

- `-f, --file <FILE>`: Path to a world JSONL file. Defaults to the nearest `world.jsonl` up from the current directory.
- `--graphics-quality <PRESET>`: Force the graphics-quality preset for this run. One of `auto`, `low`, `medium`, `high`, `ultra`, `custom`.
- `--graphics-api <API>`: Render with this graphics API. One of `metal`, `directx`, `vulkan`.
- `--window-mode <MODE>`: Open the window in this mode. One of `windowed`, `fullscreen`, `borderless`.
- `--[no-]graphics-validation`: Turn the graphics API's validation layer on or off (default: on in debug builds).
- `--[no-]upscaling`: Turn temporal upscaling on or off.
- `--rt-dynamic <MODE>`: How the ray-tracing acceleration structure tracks moving props. One of `off`, `auto`, `rebuild`, `tlas`.
- `--[no-]rt-reflections`: Turn ray-traced reflections on or off.
- `--[no-]rt-skinned-geometry`: Put skinned meshes in the ray-tracing acceleration structure or leave them out.
- `--[no-]vsync`: Turn vertical sync on or off.
- `--[no-]hud`: Start with the stats HUD shown or hidden (a toggle_hud key still flips it).
- `--dlss-preset <PRESET>`: Force the DLSS render preset. One of `default`, `k`, `l`, `m`.
- `--fixed-frame-rate <RATE>`: Step the frame clock by exactly 1/RATE seconds per frame, whatever the wall time.
- `--screenshots-dir <DIR>`: Write screenshots under DIR (default `screenshots`).
- `--hide-pass <PASS>`: Start with a feature pass off (repeatable). One of `shadows`, `fog`, `bloom`, `ssgi`, `reflections`, `lines`, `reactive`.
- `--limit-frames <N>`: Stop after N frames.
- `--limit-fps <N>`: Cap the frame rate at N frames a second (0 leaves it uncapped).
- `--debug-port [<PORT>]`: Serve the debug protocol on this localhost port (default 8777).

## `concinnity add`

Add an asset to the active app

TARGET is one of:

- A file path, such as `shaders/pbr.vert` or `models/scene.glb`. The type
  comes from the file extension or the JSON `type` field.
- A type name, such as `PointLight`. The asset takes the type's
  registered default args.
- A whole entry, such as `'["Window", {}]'`.

```sh
concinnity add models/scene.glb --template minimal-3d-world
concinnity add PointLight --id lamp
concinnity add '["Window", {}]'
```

Usage: `concinnity add [OPTIONS] <TARGET>`

- `<TARGET>`: Path to an asset file, a type name, or an entry like '["Window", {}]'.
- `-i, --id <ID>`: The `$id` written into the world. Without one, a file target's id comes from its file name, and a type name or an entry without one is added anonymous.
- `-t, --template <TEMPLATE>`: Named scaffold preset used when bootstrapping a new world. `minimal-3d-world` adds a camera, sun, room and sky to the base scaffold. Ignored when the world already exists.

## `concinnity rm`

Remove an asset from the active app by its id

ID is the `$id` an entry declares in its args, or the `<Type>#<ordinal>`
handle of an anonymous one.

```sh
concinnity rm lamp
concinnity rm 'Prop#3'
```

Usage: `concinnity rm [OPTIONS] <ID>`

- `<ID>`: The asset's `$id`, or the `<Type>#<ordinal>` of an anonymous one.

## `concinnity list`

List all declared assets

```sh
concinnity list
concinnity list --expanded --systems
```

Usage: `concinnity list [OPTIONS]`

- `-f, --file <FILE>`: Path to a world JSONL file. Defaults to the nearest `world.jsonl` up from the current directory.
- `--expanded`: List the expanded world the cook produces. Cook-time macros are expanded and injected defaults included, each row tagged with where it came from: authored, injected or expanded.
- `--systems`: List the systems this world runs, in order, each with the condition that includes it.

## `concinnity explain`

Print an asset's effective entry from the expanded app

Prints the whole world line for ID as the cook sees it, including assets
that only exist through cook-time expansion or injection. The line can
be pasted into the world file as it stands to override a default.

```sh
concinnity explain lamp
concinnity explain 'Window#0' -f arena.jsonl
```

Usage: `concinnity explain [OPTIONS] <ID>`

- `<ID>`: The asset's `$id`, or the `<Type>#<ordinal>` of an anonymous one.
- `-f, --file <FILE>`: Path to a world JSONL file. Defaults to the nearest `world.jsonl` up from the current directory.

## `concinnity docs`

Regenerate the reference pages

Writes the asset reference, the graphics settings page and the command
line reference from the rustdoc compiled into this build, so it needs no
source tree.

```sh
concinnity docs
concinnity docs --root ../engine
```

Usage: `concinnity docs [OPTIONS]`

- `--root <ROOT>`: Directory to write the docs/ pages under. Defaults to `.`.

## `concinnity export`

Package a cooked app for distribution

```sh
concinnity export
concinnity export --name "My App" --version 1.2.0 --format dir --out build
```

Usage: `concinnity export [OPTIONS]`

- `-f, --file <FILE>`: Path to a world JSONL file. Defaults to the nearest `world.jsonl` up from the current directory.
- `-n, --name <NAME>`: Override the application name.
- `--version <VERSION>`: Override the application version.
- `--platform <PLATFORM>`: Target platform, which must be this machine's (`macos`, `windows` or `linux`).
- `--out <OUT>`: Output directory for the exported app. Defaults to `dist`.
- `--format <FORMAT>`: Output format. One of `zip`, `dir`. Defaults to `zip`.
- `--dmg`: Also produce a .dmg wrapping the .app (macOS-only).

## `concinnity mcp`

Serve the debug protocol to an MCP client over stdio

An MCP client spawns it as a child process. It speaks JSON-RPC over
stdin and stdout and forwards each tool call to a running app's debug
port, which serves the same protocol to any client that posts to it.

```sh
concinnity mcp
concinnity mcp --debug-port 9100
```

Usage: `concinnity mcp [OPTIONS]`

- `--debug-port <DEBUG_PORT>`: Port the runtime debug server is listening on. Defaults to `8777`.

## `concinnity version`

Print the version

```sh
concinnity version
```

Usage: `concinnity version [OPTIONS]`
