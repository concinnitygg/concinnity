<!-- Auto-generated - do not edit. -->

# ShadowUpdate

How often each cascaded-shadow-map slice is re-rendered. The shadow pass
re-rasterizes all scene geometry into every cascade, so it is one of the
heavier passes; updating distant cascades less often cuts that cost.

`hybrid` re-renders every cascade covering the first few meters in front of
the camera, or out past what a follow camera or a camera track films, each
frame (so close shadows stay crisp and keep up with whatever moves there)
and rotates through the farther cascades one per frame. At a short reach
that can be every cascade. The shadows of things moving far away then lag a
few frames, which is imperceptible at that range. `every_frame` re-renders all cascades every
frame: pick it for scenes with fast-moving shadow casters where even distant
shadow lag is unacceptable. Each cascade is always primed (rendered once)
before it is sampled, so there is never missing shadow data.

## Values

- `every_frame`: Re-render every cascade every frame.
- `hybrid`: Re-render the near cascades every frame and the distant ones on a rotation.
