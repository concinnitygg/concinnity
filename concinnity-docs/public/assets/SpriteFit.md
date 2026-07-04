<!-- Auto-generated - do not edit. -->

# SpriteFit

How a view-owned [Sprite](Sprite.md) maps from the 1280x720 reference canvas
to the live window when their aspect ratios differ.

View-owned UI is authored against a fixed reference canvas and uniformly
scaled to the window at runtime.

## Values

- `fit`: The canvas fits inside the window, centered, leaving margins on the shorter axis. UI elements keep their proportions and stay fully visible.
- `cover`: The canvas fills the window, centered, cropping the overflowing axis equally on both sides. Full-bleed stage imagery (scene backdrops, character portraits) reaches the window edges without distorting, and content anchored to a canvas edge stays flush with the window edge.
