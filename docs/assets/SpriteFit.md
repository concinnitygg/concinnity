<!-- Auto-generated - do not edit. -->

# SpriteFit

How a screen-owned overlay element (a [Sprite](Sprite.md), [TextLabel](TextLabel.md),
or [HitRegion](HitRegion.md)) maps from the 1280x720 reference canvas to the
live window when their aspect ratios differ.

Screen-owned UI is authored against a fixed reference canvas and uniformly
scaled to the window at runtime.

## Values

- `fit`: The canvas fits inside the window, centered, leaving margins on the shorter axis. UI elements keep their proportions and stay fully visible.
- `cover`: The canvas fills the window, centered, cropping the overflowing axis equally on both sides. Full-bleed stage imagery (scene backdrops, character portraits) reaches the window edges without distorting, and content anchored to a canvas edge stays flush with the window edge.
- `bottom`: The canvas keeps the `fit` scale (no cropping), but the whole overlay is shifted so the reference bottom edge lands on the window bottom edge. Bottom-anchored furniture (a visual-novel dialog box and its controls) hugs the window bottom at any aspect ratio instead of floating above a letterbox margin.
