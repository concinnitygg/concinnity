<!-- Auto-generated - do not edit. -->

# StatHud

Requests the default on-screen stats HUD: two rows of
[`TextLabel`](TextLabel.md) chips with live engine stats, refreshed on a fixed
interval and packed at the top-left of the window.

The top row is the run's health, left to right:
- `fps_label`: the frame rate, averaged over the refresh interval.
- `worst_frame_label`: the longest frame of that interval (`MAX 23.4 ms`),
  which shows a hitch the average hides.
- `cpu_label`: the CPU work per frame: every system's step plus the render
  side's submission, with any wait on the GPU left out.
- `gpu_label`: the GPU time per frame.
- `allocs_label`: heap allocations per frame, and the most any one frame
  of the interval made (`ALLOC 3/f max 40`). Counted across the whole
  process, background loading included. Blank in a program built without
  the counting allocator.
- `vram_label`: the GPU-memory use. On unified-memory hardware (Apple
  Silicon) that is the device's share of system memory.
- `ram_label`: the process resident set size, against the memory budget
  when it is known.
- `streaming_label`: loads in flight across the streaming pools (textures,
  meshes, chunks, planet tiles).
- `ev_label`: the adapted auto-exposure value.
- `edr_label`: the HDR panel's headroom multiplier (`EDR x2.0` on an
  HDR400 panel).

Comparing `CPU` with `GPU` tells which side bounds the frame: the larger
one is near the frame time and the other has headroom. Both are averaged
over the same frames as `FPS`.

The second row appears in a benchmarked run (one armed with a
[`BenchmarkConfig`](BenchmarkConfig.md)):
- `time_label`: how far the [`CameraTrack`](CameraTrack.md) has run, on the
  clock the benchmark report times its frames on, so a moment seen on
  screen (`TIME 47.3 s`) can be found in the report.
- `phase_label`: the benchmark phase and how far into it the run is
  (`PHASE ground 12.4/60.0 s`).
- `api_label`: the graphics API the run renders with.
- `resolution_label`: the size the scene renders at and, when it is
  upscaled, the size it is shown at and the upscaler
  (`RES 1280x720 -> 2560x1440 MetalFX`).
- `objects_label`: the renderable objects in the scene.
- `draws_label`: the draw calls the CPU issued. Draws the GPU culls and
  issues for itself are not counted.

`time_label` also shows in a run without a benchmark whenever a track is
running. Chips whose stat is unavailable stay blank, and a blank chip
reserves no width. The chips' on-screen position is fixed by the engine
rather than the authored coordinates.

The frame-rate, frame-time, allocation, and GPU-memory chips are shown or
hidden from the in-game video settings ("Display performance stats"). A
`toggle_hud` action hides and shows the whole HUD.

Developer-facing readouts (per-pass GPU timings, cursor position, live
camera pose) live on the separate [`DebugHud`](DebugHud.md), toggled with F1.

A world that declares a [`MainMenu`](MainMenu.md) receives a `StatHud` from the
build when it declares none, since the menu's performance-stats toggles
drive the chips, and any label field left unset receives a chip at start.
So the example below is only needed to restyle the chips or run a HUD
without a menu. Declare an [`EngineDefaults`](EngineDefaults.md) with
`"hud": false` to leave the chips unfilled.

```jsonl
["Font",{"$id":"hud_font","size_px":20}]
["TextLabel",{"$id":"fps_chip","font":"hud_font","scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}]
["StatHud",{"fps_label":"fps_chip"}]
```

## Parameters

- `fps_label`: A string. [`TextLabel`](TextLabel.md) that receives the frame-rate chip text. Optional.
- `worst_frame_label`: A string. [`TextLabel`](TextLabel.md) that receives the longest-frame chip text. Optional.
- `cpu_label`: A string. [`TextLabel`](TextLabel.md) that receives the CPU-time chip text. Optional.
- `gpu_label`: A string. [`TextLabel`](TextLabel.md) that receives the GPU-time chip text. Optional.
- `allocs_label`: A string. [`TextLabel`](TextLabel.md) that receives the heap-allocation chip text. Optional.
- `vram_label`: A string. [`TextLabel`](TextLabel.md) that receives the GPU-memory chip text. Optional.
- `ram_label`: A string. [`TextLabel`](TextLabel.md) that receives the host-memory (RSS) chip text. Optional.
- `streaming_label`: A string. [`TextLabel`](TextLabel.md) that receives the streaming chip text. Optional.
- `ev_label`: A string. [`TextLabel`](TextLabel.md) that receives the auto-exposure chip text. Optional.
- `edr_label`: A string. [`TextLabel`](TextLabel.md) that receives the HDR-headroom chip text. Optional.
- `time_label`: A string. [`TextLabel`](TextLabel.md) that receives the run-time chip text. Optional.
- `phase_label`: A string. [`TextLabel`](TextLabel.md) that receives the benchmark-phase chip text. Optional.
- `api_label`: A string. [`TextLabel`](TextLabel.md) that receives the graphics-API chip text. Optional.
- `resolution_label`: A string. [`TextLabel`](TextLabel.md) that receives the resolution chip text. Optional.
- `objects_label`: A string. [`TextLabel`](TextLabel.md) that receives the object-count chip text. Optional.
- `draws_label`: A string. [`TextLabel`](TextLabel.md) that receives the draw-call chip text. Optional.
