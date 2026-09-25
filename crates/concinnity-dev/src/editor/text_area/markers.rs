//! Per-line gutter markers: a diagnostic's severity and message pinned to the
//! line it names. A plain input to the layout; the text area never makes one.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

// The marker a line shows: its most severe, the first of equals.
pub(crate) fn marker_on(markers: &[GutterMarker], line: usize) -> Option<&GutterMarker> {
    markers.iter().filter(|m| m.line == line).fold(
        None,
        |best: Option<&GutterMarker>, m| match best {
            Some(b) if b.severity >= m.severity => Some(b),
            _ => Some(m),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker(line: usize, severity: Severity, message: &str) -> GutterMarker {
        GutterMarker {
            line,
            column: 0,
            severity,
            message: message.to_string(),
        }
    }

    #[test]
    fn a_line_shows_its_most_severe_marker() {
        let markers = [
            marker(2, Severity::Warning, "unused"),
            marker(2, Severity::Error, "undeclared"),
            marker(2, Severity::Error, "second error"),
            marker(5, Severity::Warning, "shadowed"),
        ];
        assert_eq!(marker_on(&markers, 2).unwrap().message, "undeclared");
        assert_eq!(marker_on(&markers, 5).unwrap().severity, Severity::Warning);
        assert!(marker_on(&markers, 3).is_none());
    }
}
