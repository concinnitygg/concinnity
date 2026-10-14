<!-- Auto-generated - do not edit. -->

# Exposure

Scene exposure: a fixed offset in photographic stops, or auto-exposure that
measures the scene's average brightness each frame and adapts toward a
balanced mid-tone. One per world; with none declared, the defaults below
apply.

When `auto` is on, `ev` acts as an additive bias (in stops) on top of the
adapted value.

## Parameters

- `ev`: A float. Exposure offset in photographic stops. Each +1 doubles scene brightness before bloom and tonemapping; 0 is neutral. Defaults to `0.0`. Clamped to `[-24, 24]`. Applies live.
- `auto`: A boolean. Auto-exposure. Adapts exposure each frame toward a balanced mid-tone; `ev` then acts as an additive bias in stops on top of the adapted value. Defaults to `false`. Applies live, rebuilding the passes it gates.
- `auto_min_ev`: A float. Lower bound on the adapted exposure (EV). The `ev` bias is applied before this clamp. Defaults to `-8.0`. Clamped to `[-24, 24]`. Applies live.
- `auto_max_ev`: A float. Upper bound on the adapted exposure (EV): how far a dark scene is brightened, at most 24. A night under a [NightSky](NightSky.md) adapts at around 19 to 20, so wants 24. Defaults to `8.0`. Clamped to `[-24, 24]`. Applies live.
- `auto_speed`: A float. How quickly exposure chases a new target (per second). Higher converges faster but can pump under flickering content; 1-3 is comfortable. Defaults to `1.5`. Clamped to `[0.001, 20]`. Applies live.
