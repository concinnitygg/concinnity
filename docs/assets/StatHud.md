<!-- Auto-generated - do not edit. -->

# StatHud

Requests the default on-screen stats HUD. Drives a set of
[TextLabel](TextLabel.md) chips with live engine stats, refreshed on a fixed
interval.

Each label field, when set, receives one chip: `fps_label` the averaged
frame rate, `vram_label` the GPU-memory use, `ram_label` the host process
memory (resident set size, against the memory budget when known), `ev_label`
the auto-exposure value, and `edr_label` the HDR headroom multiplier. Chips
whose stat is unavailable stay blank. The frame-rate and GPU-memory chips
are shown or hidden from the in-game video settings ("Display performance
stats"); the host-memory, exposure, and HDR chips show whenever their
reading is available.

The chips are packed into a tight strip anchored at the top-left of the
window, left to right in the order fps, vram, ram, ev, edr; a blank chip
reserves no width, so hidden readouts leave no gap. Their on-screen position
is fixed by the engine rather than the authored coordinates.

Developer-facing readouts (per-pass GPU timings, cursor position, live
camera pose) live on the separate [DebugHud](DebugHud.md), toggled with F1.

A world that declares a [MainMenu](MainMenu.md) receives a `StatHud` from the
build when it declares none, since the menu's performance-stats toggles
drive the chips, and any label field left unset receives a chip at start.
So the example below is only needed to restyle the chips or run a HUD
without a menu. Declare an [EngineDefaults](EngineDefaults.md) with
`"hud": false` to leave the chips unfilled.

## Parameters

- `fps_label`: A string. [TextLabel](TextLabel.md) that receives the frame-rate chip text. Optional.
- `vram_label`: A string. [TextLabel](TextLabel.md) that receives the GPU-memory chip text. Optional.
- `ram_label`: A string. [TextLabel](TextLabel.md) that receives the host-memory (RSS) chip text. Optional.
- `ev_label`: A string. [TextLabel](TextLabel.md) that receives the auto-exposure chip text. Optional.
- `edr_label`: A string. [TextLabel](TextLabel.md) that receives the HDR-headroom chip text. Optional.
