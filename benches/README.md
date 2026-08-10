# concinnity-bench

Microbenchmarks for the engine's in-process foundations, measured with the
engine's own instruments. Each benchmark reports CPU time per unit of work next
to the memory traffic one iteration causes:

- time and p95 come from calibrated timing samples,
- allocation counts and heap deltas come from `concinnity_memory::stats()`
  (the same `TrackingAlloc` every engine binary runs on),
- the vram column comes from the tagged ledger's device realm, so benchmarks
  over device-side accounting report through the same table.

This deliberately uses no external bench framework: the point is that the
numbers agree with what the engine's own readouts would say.

## Running

```
cargo bench -p concinnity-bench                     # every target
cargo bench -p concinnity-bench --bench ecs         # one target
cargo bench -p concinnity-bench --bench ecs -- join # substring filter
cargo bench -p concinnity-bench -- --json out.json  # machine-readable report
```

## Targets

- `ecs`: the EAS storage primitives. Spawn and despawn churn, dense column
  scans, two- and three-way joins (partner columns shuffled so probes pay a
  real scattered read), targeted lookups, deferred command record and apply,
  sparse-column churn, and the event queue.
- `memory`: the allocation layer. Frame arena against the heap path it
  replaces, pool churn, the inline-vec single-element case, and ledger report
  cost.
- `cook`: the world cook and blob load path. Front-half parse + expand +
  validate, the full in-memory compile, and the blob encode / parse pair, on
  a procedural prop world at 1k and 10k entities. The payload cache is
  redirected to `target/bench-cook-state/` so runs never touch a real
  project's build state.
- `anim`: the CPU animation pipeline on a 64-joint rig. Clip sampling at
  import-baked (61-key) and sparse (2-key) track densities, weighted pose
  blending, skinning-matrix resolution, the composed two-clip per-character
  cost, and the two-bone IK solve.
- `render`: the GPU-free render-prep layer. BVH build and frustum query
  over a 10k-object scene, light packing for the clustered forward pass,
  the streaming planner's per-frame re-rank under sustained pool pressure
  (churn asserted), and draw-slot recycling. No backend is involved.
- `physics`: the physics wrapper and rapier. Stepping benches rebuild an
  identical stacked world and step it a fixed count per iteration, so the
  measured work is bit-identical run to run (asserted, so a rapier upgrade
  that breaks determinism fails loudly): sustained-contact settling,
  contact-free fall, the sleeping-island idle step, world build, body
  churn, raycasts, and the character-move solve.

## Reading the report

`items` is how many units of work one body call performs (entities scanned,
lookups made); all columns are stated per item. A steady-state benchmark should
show `0 B` heap per item; a build-and-drop benchmark shows its allocation count
but no retained bytes. Bodies are deterministic (fixed seeds, fixed layouts),
so numbers are comparable across runs on one machine.

## Adding a benchmark

Call `Bench::run(name, items, body)` from a bench target. Name benchmarks
`target/what/size`. Keep bodies self-contained: tear down what you build, or
the heap column will show the drift. New targets are a file in the package
root with a `[[bench]]` entry naming it via `path`, `harness = false`.

This crate must not depend on `concinnity-engine`: the harness installs the
tracking allocator itself, and linking the engine would install a second one.

Whole-frame, whole-world measurements (system schedules, streaming, physics
under load) are a different instrument: they run a real client against a
generated world and are out of scope here.
