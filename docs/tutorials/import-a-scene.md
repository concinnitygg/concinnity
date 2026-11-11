# From scene file to running app

A scene file and a sky are enough to make an app. This tutorial takes a
downloaded scene to a running app with one `concinnity add` and one
`concinnity run`, and no hand-written world file. The example is the Amazon
Lumberyard Bistro, a 2.8 million triangle street scene, but the same steps
work for any `.glb` or `.fbx` scene and `.hdr` sky. It assumes the
`concinnity` CLI is installed.

## Download the Bistro

Download the Bistro pack from NVIDIA's
[Open Research Content Archive](https://developer.nvidia.com/orca/amazon-lumberyard-bistro),
then make a folder for the app and unzip the pack into it:

```sh
mkdir bistro
cd bistro
unzip ~/Downloads/Bistro_v5_2.zip
```

The folder now holds `Bistro_v5_2/`, which contains the exterior scene
`BistroExterior.fbx`, its textures, and an HDR sky
`san_giuseppe_bridge_4k.hdr`.

## Add the scene and the sky

```sh
concinnity add Bistro_v5_2/BistroExterior.fbx Bistro_v5_2/san_giuseppe_bridge_4k.hdr
```

There is no world yet, so `add` creates `world.jsonl` and writes two lines
to it: a [SceneImport](../assets/SceneImport.md) for the scene and an
[EnvironmentMap](../assets/EnvironmentMap.md) that draws the HDR as the sky
and lights the scene with it. That is the whole world:

```sh
concinnity list
```

Everything else is filled in when the world is cooked. The import expands
into the scene's meshes, materials, textures, and props, and frames a camera
on them. A window and graphics settings are supplied as defaults, and the
engine lights the scene with a built-in sun when the world declares none.
`concinnity list --expanded` shows the full world and where each asset came
from.

## Run

```sh
concinnity run
```

`run` cooks the world before it opens the window. Cooking compiles the world
into the binary blob the engine loads, under `.concinnity/`. The first cook
imports every mesh and texture in the scene and takes the longest; later
cooks reuse what has not changed. `concinnity cook` does the same compile
without running.

The window opens on the whole Bistro. Click in the window to look around
with the mouse, and fly with the keys below. Escape releases the mouse.

| Key        | Move                       |
| ---------- | -------------------------- |
| W, A, S, D | Forward, left, back, right |
| Space      | Up                         |
| Ctrl       | Down                       |
| Shift      | Faster                     |

## Next steps

- [Command line](../cli.md) lists every command and its options.
- `concinnity edit` opens the same world in the in-engine editor, which
  saves changes back to `world.jsonl`.
