<!-- Auto-generated - do not edit. -->

# ScreenInput

How a [Screen](Screen.md) treats input while it is active.

## Values

- `capture`: The screen owns input while it is the topmost capturing screen: gameplay input is suppressed and lower screens' [HitRegion](HitRegion.md)s stop firing.
- `passthrough`: The screen only draws; input passes through to whatever is beneath it.
