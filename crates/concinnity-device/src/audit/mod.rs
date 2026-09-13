// Ownership guard for the explicit backends' resource barriers. Backend-agnostic
// because the call sites are counted as text, so one build audits both explicit
// backends.
mod barrier;

// The companion guard: `barrier` proves every barrier is classified, this one
// proves a classified barrier is not redundant with one the graph executor
// already emits. Same text-scanning rationale, so it also covers DirectX from a
// macOS build.
mod double_drive;
