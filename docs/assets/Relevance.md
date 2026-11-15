<!-- Auto-generated - do not edit. -->

# Relevance

When an effect stops contributing to what the camera sees, so its passes
can stand down. The `relevance` section of [`GraphicsConfig`](GraphicsConfig.md);
a field left unset takes the default below.

Each rule reads the view rather than the world, measuring against the
nearest surface the camera sees of what the effect acts on, so the same
thresholds hold for a pebble at arm's length and a planet seen from orbit:

- Ambient occlusion, the screen-space bounce and the reflections run while
  their reach (the occlusion's radius, each trace's distance) spans
  `min_reach_pixels` at that surface.
- The sun's cascades run while something they shade lies within their
  shadow distance: a surface in view, fog thick enough near the camera, or
  a raymarched volume in view. They have no threshold of their own.
- The ocean simulation runs while its longest visible wave spans
  `min_wave_pixels` at the nearest water in view.
- The reflection probe a prop rides captures while the prop is in view and
  spans `min_subject_pixels` across.
- The disocclusion mark runs while something moves `min_motion_pixels`
  against the view in a frame.

An effect that stood down comes back once its rule holds with a quarter to
spare, and stands down only after its rule has failed for half a second, so
a camera resting at a threshold does not toggle it. A threshold of 0 keeps
its effects on, and `enabled` false keeps every effect on.

## Parameters

- `enabled`: A boolean. Whether effects stand down where they do not contribute. Off runs every effect the other settings ask for, whatever the view. Defaults to `true`. Applies live.
- `min_reach_pixels`: A float. How many pixels an effect's reach must span at the nearest surface it acts on: ambient occlusion's radius, the screen-space bounce's trace and the reflections' trace. Below it, what the effect adds is smaller than a pixel. Defaults to `1.0`. Clamped to `[0, 64]`. Applies live.
- `min_wave_pixels`: A float. How many pixels the longest wave the ocean simulation shows (the longest still holding a thousandth of its sea's peak energy) must span at the nearest visible water for the simulation to run. The water draws no wave narrower than 3 pixels. Defaults to `3.0`. Clamped to `[0, 64]`. Applies live.
- `min_subject_pixels`: A float. How many pixels across the prop a reflection probe rides must appear for the probe to capture. Defaults to `16.0`. Clamped to `[0, 1024]`. Applies live.
- `min_motion_pixels`: A float. How many pixels a surface must move against the view in a frame for the disocclusion mark to run: the camera's own travel seen at the nearest still surface, or anything moving in view, wind-swayed grass included. The mark itself passes over anything uncovered by less than a pixel. Defaults to `1.0`. Clamped to `[0, 64]`. Applies live.
