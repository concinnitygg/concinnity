<!-- Auto-generated - do not edit. -->

# PlanetTour

How a [`BenchmarkConfig`](BenchmarkConfig.md) tours a [`Planet`](Planet.md).

The tour follows one great circle round the planet through where the world
starts. It sets out over the ground, crosses the shore and the sea, climbs
toward the sun until it leaves the air, shows the whole planet from orbit,
drops into the night and skims it, and comes down on the far side as the
sun rises. The camera films the benchmark's `subject` from behind, framed
by where it is on the tour and where the sun is.

A subject with a [`RollingDrive`](RollingDrive.md) rolls itself over the
ground first: through the grass, down the beach and into the sea, where it
floats and paddles until the tour picks it up and flies it; on a planet
with no sea it is picked up as the roll ends. A subject
without one, or none at all, skims the ground instead.

```json
["BenchmarkConfig",{"subject":"ball","tour":{"heading_deg":96.0}}]
```

## Parameters

- `heading_deg`: A float. The direction the tour sets out in from where the world starts, in degrees clockwise from north. Defaults to `96.0`.
- `start_back`: A float. How far behind where the world starts the tour begins, in meters. Defaults to `150.0`.
