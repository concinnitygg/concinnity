<!-- Auto-generated - do not edit. -->

# AppLimits

Optional per-application overrides for the runtime's thread and memory
budgets. Each field of `0` means "auto" (the engine picks a value from the
host machine); a non-zero value overrides that choice, clamped to what the
machine can safely give.

## Parameters

- `max_memory_mb`: An integer. Soft ceiling on host memory the runtime aims to stay under, in mebibytes. `0` = auto (a fraction of total RAM, capped by a built-in ceiling). A non-zero value is clamped so it never exceeds what the machine can safely give.
- `job_threads`: An integer. Worker threads for the shared job pool. `0` = auto (one per core, less one for the main thread). A non-zero value never exceeds the core count.
