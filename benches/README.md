# concinnity-bench

Microbenchmarks for the engine's in-process foundations, measured with the
engine's own instruments. Each benchmark reports CPU time per unit of work next
to the memory traffic one iteration causes:

- time and p95 come from calibrated timing samples,
- allocation counts and heap deltas come from `concinnity_core::memory::stats()`
  (the same `TrackingAlloc` every engine binary runs on),
- the vram column comes from the tagged ledger's device realm, so benchmarks
  over device-side accounting report through the same table.

This deliberately uses no external bench framework: the point is that the
numbers agree with what the engine's own readouts would say.

## Running

```
cargo bench -p concinnity-bench                      # every target
cargo bench -p concinnity-bench --bench cook         # one target
cargo bench -p concinnity-bench --bench cook -- 10k  # substring filter
cargo bench -p concinnity-bench -- --json out.json   # machine-readable report
```

## Targets

- `cook`: the world cook and blob load path. Front-half parse + expand +
  validate, the full in-memory compile, and the blob encode / parse pair, on
  a procedural prop world at 1k and 10k entities. The payload cache is
  redirected to `target/bench-cook-state/` so runs never touch a real
  project's build state.
- `anim`: the CPU animation pipeline on a 64-joint rig. Clip sampling at
  import-baked (61-key) and sparse (2-key) track densities, weighted pose
  blending, skinning-matrix resolution, the composed two-clip per-character
  cost, and the two-bone IK solve.
- `engine`: the engine's public `World` surface, against the real registered
  component set rather than the synthetic three the in-crate storage
  benchmarks use. World populate (with and without the manifest pre-size),
  column iteration, targeted lookups, and column drain. `World::despawn` is
  `#[cfg(test)]` and so is not reachable here.
- `render`: the GPU-free render-prep layer (`concinnity_core::render`). Light
  packing for the clustered forward pass, the streaming planner's per-frame
  re-rank under sustained pool pressure (churn asserted), and draw-slot
  recycling. No backend is involved; the BVH build and frustum query are
  in-crate, below.

## In-crate benchmarks

Some benchmarks live in the crate they measure rather than here, because the
macro or the private type they drive is expanded by its consumers rather than
called across a crate boundary. They run as ignored tests, and report through
the same `Bench::run` instrument:

- the ECS storage primitives (`concinnity-core`). Spawn and despawn churn,
  dense column scans, two- and three-way joins (partner columns shuffled so
  probes pay a real scattered read), targeted lookups, and the event queue;
  plus the three-way join comparison that sized the draw-list build path.

  ```
  cargo test -p concinnity-core --release -- --ignored --nocapture --test-threads=1 bench_storage
  cargo test -p concinnity-core --release -- --ignored --nocapture join_probe
  ```

- the allocation layer (`concinnity_core::memory`). Frame arena against the heap
  path it replaces, pool churn, the inline-vec single-element case, and ledger
  report cost.

  ```
  cargo test -p concinnity-core --release -- --ignored --nocapture --test-threads=1 memory::bench
  ```

- the rigid-body simulation (`concinnity_core::physics`). Stepping benches rebuild
  an identical stacked world and step it a fixed count per iteration, so the
  measured work is bit-identical run to run (asserted, so a change that breaks
  determinism fails loudly): sustained-contact settling, contact-free fall, the
  sleeping-island idle step, world build, body churn, joints, sensor regions,
  contact reporting, terrain, raycasts, shape casts, and the character-move
  solve. Stepping is measured on the engine's job pool, the way the driver runs
  it; the `_serial` twins step the same world on the calling thread, so a pair
  reads as the scaling the split bought. Both must land in the same place,
  which the determinism assertion checks.

  ```
  cargo test -p concinnity-core --release -- --ignored --nocapture --test-threads=1 physics::bench
  ```

- the render-prep BVH (`concinnity_core::render::bvh`). Build and frustum query
  over a 10k-object scene, against the item type the builder is generic over,
  which its consumers rather than this package supply.

  ```
  cargo test -p concinnity-core --release -- --ignored --nocapture --test-threads=1 bench_bvh
  ```

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

Whole-frame, whole-world measurements (system schedules, streaming, physics
under load) are a different instrument: they run a real client against a
generated world and are out of scope here.
