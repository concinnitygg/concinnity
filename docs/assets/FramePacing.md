<!-- Auto-generated - do not edit. -->

# FramePacing

How the render loop paces its frames: how many may be in flight, whether
presentation waits for the display refresh, and the frame-rate cap. The
`frame_pacing` section of [GraphicsConfig](GraphicsConfig.md); a field left
unset takes the default below.

Pacing is a preference of whoever runs the build rather than a quality
tier, so no graphics quality preset changes it.

## Parameters

- `frames_in_flight`: An integer. Preferred number of frames in flight. Higher can smooth pacing at the cost of input latency. Defaults to `2`. Clamped to `[1, 3]`. Applies at the next launch.
- `vsync`: A boolean. Cap the frame rate to the display refresh (vsync). Defaults to `false`: presentation is not tied to the refresh (DirectX presents with tearing allowed, Vulkan uses a mailbox present mode) and `fps_cap` alone bounds the frame rate. Set to `true` to lock presentation to the monitor refresh, eliminating tearing and the wasted frames that never reach the screen. Applies live.
- `fps_cap`: An integer. Cap the frame rate to this many frames per second, `300` by default. `0` leaves the loop uncapped, which is what GPU cost measurement wants: a capped GPU idles between frames and lowers its clocks. The cap is a CPU-side frame pacer, so it composes with `vsync`: the more restrictive of the two wins. Lower caps limit heat, fan noise, and power draw. Applies live.
