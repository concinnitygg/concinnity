<!-- Auto-generated - do not edit. -->

# ShadowUpdate

How often each cascaded-shadow-map slice is re-rendered. The shadow pass
re-rasterizes all scene geometry into every cascade, so it is one of the
heavier passes; updating distant cascades less often cuts that cost.

`hybrid` re-renders the nearest cascade every frame, and any cascade a
moving, animating, appearing or vanishing caster reaches on the frame it
does so, so no shadow is left behind by what moves. It rotates through the
other cascades one per frame, so a still scene costs about two cascades a
frame at any reach. `every_frame` re-renders all cascades every frame:
pick it where even the rotation's few frames of lag on what does not move
are unacceptable, such as a fast-turning sun. Each cascade is always primed
(rendered once) before it is sampled, so there is never missing shadow
data.

## Values

- `every_frame`: Re-render every cascade every frame.
- `hybrid`: Re-render the nearest cascade every frame, the ones a moving caster reaches as it moves, and the rest on a rotation.
