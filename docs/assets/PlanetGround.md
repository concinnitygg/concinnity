<!-- Auto-generated - do not edit. -->

# PlanetGround

The ground sculpted into one square of a [`Planet`](Planet.md): how far it
is raised or lowered, and how much of the generated hills it keeps, at 33
by 33 points spread evenly across the square.

The square is `tile`, `[face, level, column, row]` of the planet's
squares: each `face` (0 to 5: `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`) is one
side of the cube the sphere is seen through, `level` 0 is the whole face
and each level up to 20 halves the side, and `column` and `row` count the
squares across the face at that level. The points run along the column
first, row by row, the first and last of each row on the square's edges,
so a square's edge points are its neighbor's too.

`height` raises (or, below 0, lowers) the ground at each point, in meters.
`detail`, in `[0, 1]`, scales the generated hills there: 1 keeps them
whole, 0 leaves only the smooth continents and coasts. An empty `detail`
keeps the hills whole everywhere. Between points the ground blends
smoothly.

A place on the planet takes its ground from the smallest square over it,
never from several at once, so finer sculpting in a small area sits on a
larger square sculpted around it. A world holds at most one square per
tile.

## Parameters

- `tile`: An array of 4 integers. The square: `[face, level, column, row]`. Defaults to `[0, 0, 0, 0]`.
- `height`: An array of floats. The ground raised at each point, in meters; 33 x 33 values. Defaults to `[]`.
- `detail`: An array of floats. How much of the generated hills each point keeps, in `[0, 1]`; 33 x 33 values, or none to keep them whole. Defaults to `[]`.
