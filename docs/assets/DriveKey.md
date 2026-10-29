<!-- Auto-generated - do not edit. -->

# DriveKey

One stretch of a scripted [`RollingDrive`](RollingDrive.md): how the drive
is steered from `seconds` until the next key.

The heading is a compass bearing, taken at the body every tick: degrees
clockwise from north along the ground, north being toward the
[`Planet`](Planet.md)'s north pole, or toward `-Z` without a planet. It
stays true however far the body rolls.

## Parameters

- `seconds`: A float. When the key takes over, in seconds after a benchmark's warm-up. Defaults to `0.0`.
- `heading_deg`: A float. Where to roll, in degrees clockwise from north. Defaults to `0.0`.
- `throttle`: A float. How hard to drive, from 0 (coast) to 1. Defaults to `1.0`.
- `sprint`: A boolean. Whether to sprint. Defaults to `false`.
- `jump`: A boolean. Jump as the key takes over, if the body is on the ground. Defaults to `false`.
- `pace`: A float. A speed to tow the body at along the heading, in meters per second, in place of the drive: the body is pulled along the ground toward it, gaining it at a steady rate, however the drive could move it, while what holds it up (the ground, the water) still does. 0 leaves it to the drive. Defaults to `0.0`.
