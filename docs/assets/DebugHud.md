<!-- Auto-generated - do not edit. -->

# DebugHud

Requests the developer debug HUD: a set of [TextLabel](TextLabel.md) chips
with diagnostic readouts, anchored to the top-right of the window and
toggled with F1 (hidden by default).

Each label field, when set, receives one chip: `passes_label` a multi-line
list of the heaviest rendering steps of the last frame, `mouse_label` the
cursor position in window pixels, `camera_label` the live camera pose
(position, yaw, pitch) in the exact form a fixed viewpoint is reproduced
with, and `sys_label` the worker-thread and host-memory budgets (the job
pool's thread count, and the process resident set against the memory
budget). Chips whose stat is unavailable stay blank. The chips stack
vertically from the top-right corner in the order cursor, then camera, then
system, then passes (passes is last because its height varies with the
frame's step count), so their on-screen position is fixed by the engine
rather than the authored coordinates.

The always-on frame-rate and GPU-memory readouts live on the separate
[StatHud](StatHud.md).

Every rendering world receives a `DebugHud` and its chip labels at build
time when it declares none, so the example below is only needed to restyle
the chips. The HUD only activates in developer contexts: a debug build of
the host binary, or a world launched through `cn debug`; release builds
leave it inert even when declared. Declare an
[EngineDefaults](EngineDefaults.md) with `"debug_hud": false` to remove it
from the build entirely.

## Parameters

- `passes_label`: A string. [TextLabel](TextLabel.md) that receives the per-step GPU-timing chip text. Optional.
- `mouse_label`: A string. [TextLabel](TextLabel.md) that receives the cursor-position chip text. Optional.
- `camera_label`: A string. [TextLabel](TextLabel.md) that receives the live camera-pose chip text. Optional.
- `sys_label`: A string. [TextLabel](TextLabel.md) that receives the thread / memory budget chip text. Optional.
