<!-- Auto-generated - do not edit. -->

# FramePriority

Which of the threads every frame waits on run above normal priority, ahead
of the other applications on the machine.

A frame waits on three kinds of thread: the render thread, the simulation
thread, and the job workers both of them fan work out to. Raising a thread
keeps a busy machine (a browser, a build, a stream) from holding it off a
core, which is what keeps frame times steady under load. The cost lands on
those other applications: while the raised threads have work, an
application at normal priority waits for a core.

`all_threads` (the default) raises all three. A frame waits on its job
workers as much as on its render and simulation threads, so on a busy
machine raising only the main two keeps frames no steadier than raising
nothing. `main_threads` does exactly that, leaving the workers to share
the machine. `normal` raises nothing, for an application that should yield
to whatever else is running.

Every frame thread is kept off power throttling whichever is chosen. The
setting applies where the platform schedules by priority (Windows); on
macOS and iOS frame threads always run at the user-interactive
quality-of-service class, and elsewhere they keep the system's defaults.

## Values

- `normal`: Every frame thread at normal priority.
- `main_threads`: The render and simulation threads above normal priority, the job workers at normal priority.
- `all_threads`: The render and simulation threads and every job worker above normal priority.
