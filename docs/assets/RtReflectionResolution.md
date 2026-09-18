<!-- Auto-generated - do not edit. -->

# RtReflectionResolution

Internal resolution of the ray-traced reflection trace (only meaningful when
`ray_traced_reflections` is on). Tracing is the expensive part of ray-traced
reflections, so `half` (the default) casts rays for a quarter of the pixels
and the reflection composite upsamples them with a depth- and normal-aware
filter that keeps edges from bleeding. `full` traces every pixel; `quarter`
is the cheapest.

## Values

- `full`: Trace at native resolution.
- `half`: Trace at half resolution per axis.
- `quarter`: Trace at quarter resolution per axis.
