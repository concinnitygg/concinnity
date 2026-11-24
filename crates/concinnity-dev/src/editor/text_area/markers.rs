//! Per-line gutter markers: a diagnostic's severity and message pinned to the
//! line it names. A plain input to the layout; the text area never makes one.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Severity {
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GutterMarker {
    // Zero-based line index, and the zero-based column a jump to it lands on.
    pub(crate) line: usize,
    pub(crate) column: usize,
    pub(crate) severity: Severity,
    pub(crate) message: String,
}
