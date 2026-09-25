//! dxc's diagnostics, read back out of what it prints.
//!
//! dxc reports in clang's format: a `path:line:column: severity: message`
//! header, then the offending source line, a caret under the column, and any
//! notes. The path is whatever the source named through `#line`, or the file
//! name dxc was given where no directive applies.

use std::fmt;

/// How serious a diagnostic is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    /// The compile failed.
    Error,
    /// The compile went on.
    Warning,
}

impl Severity {
    /// The word dxc prints for it.
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One diagnostic dxc reported at a location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Diagnostic {
    /// The file, spelled as the `#line` directive in force named it.
    pub path: String,
    /// 1-based line in `path`.
    pub line: u32,
    /// 1-based column.
    pub column: u32,
    /// Whether it failed the compile.
    pub severity: Severity,
    /// The message, without the location or severity.
    pub message: String,
    /// The lines dxc printed under the header, verbatim: the source line, the
    /// caret, and any notes. Empty when it printed none.
    pub context: String,
}

impl Diagnostic {
    /// Whether two diagnostics report the same thing at the same place, which
    /// is what one error compiled into several programs looks like.
    pub fn same_as(&self, other: &Diagnostic) -> bool {
        self.key() == other.key()
    }

    fn key(&self) -> (&str, u32, u32, &str) {
        (&self.path, self.line, self.column, &self.message)
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}: {}",
            self.path, self.line, self.column, self.severity, self.message
        )?;
        if !self.context.is_empty() {
            write!(f, "\n{}", self.context)?;
        }
        Ok(())
    }
}

/// Every located error and warning in `output`, in the order dxc printed them.
///
/// A note is context of the diagnostic above it rather than one of its own.
/// Lines before the first header, and an unlocated diagnostic (the validator's
/// `error: validation errors`, say), are not diagnostics; a caller keeps the
/// raw text for those.
pub fn parse(output: &str) -> Vec<Diagnostic> {
    let mut out: Vec<Diagnostic> = Vec::new();
    let mut open = false;
    for line in output.lines() {
        match header(line) {
            Some(Header::Located(diagnostic)) => {
                finish(out.last_mut(), open);
                out.push(diagnostic);
                open = true;
            }
            Some(Header::Unlocated) => {
                finish(out.last_mut(), open);
                open = false;
            }
            None if open => {
                let context = &mut out.last_mut().expect("an open diagnostic").context;
                if !context.is_empty() {
                    context.push('\n');
                }
                context.push_str(line);
            }
            None => {}
        }
    }
    finish(out.last_mut(), open);
    out
}

/// `diagnostics` with every repeat of an earlier one dropped (see
/// [`Diagnostic::same_as`]), keeping the first of each in order.
pub fn dedup(diagnostics: impl IntoIterator<Item = Diagnostic>) -> Vec<Diagnostic> {
    let mut out: Vec<Diagnostic> = Vec::new();
    for diagnostic in diagnostics {
        if !out.iter().any(|d| d.same_as(&diagnostic)) {
            out.push(diagnostic);
        }
    }
    out
}

enum Header {
    // A new diagnostic, or a note (which continues the open one's context).
    Located(Diagnostic),
    // A severity with no location: it ends the open diagnostic.
    Unlocated,
}

// Trailing blank lines are dxc's separator between diagnostics, not context.
fn finish(last: Option<&mut Diagnostic>, open: bool) {
    if let (Some(d), true) = (last, open) {
        let trimmed = d.context.trim_end_matches(['\n', ' ']).len();
        d.context.truncate(trimmed);
    }
}

fn header(line: &str) -> Option<Header> {
    if severity_of(line).is_some() {
        return Some(Header::Unlocated);
    }
    // The path may itself hold `: ` or a drive colon, so try every split and
    // take the first that reads as `path:line:column`.
    for (at, _) in line.match_indices(": ") {
        let Some((path, line_no, column)) = location(&line[..at]) else {
            continue;
        };
        let rest = &line[at + 2..];
        if rest.starts_with("note: ") {
            return None;
        }
        let (severity, message) = severity_of(rest)?;
        return Some(Header::Located(Diagnostic {
            path: path.to_string(),
            line: line_no,
            column,
            severity,
            message: message.to_string(),
            context: String::new(),
        }));
    }
    None
}

// `path:line:column`, split from the right.
fn location(text: &str) -> Option<(&str, u32, u32)> {
    let (rest, column) = text.rsplit_once(':')?;
    let (path, line) = rest.rsplit_once(':')?;
    if path.is_empty() {
        return None;
    }
    let number = |s: &str| {
        s.bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| s.parse().ok())
            .flatten()
    };
    Some((path, number(line)?, number(column)?))
}

fn severity_of(text: &str) -> Option<(Severity, &str)> {
    [
        ("fatal error: ", Severity::Error),
        ("error: ", Severity::Error),
        ("warning: ", Severity::Warning),
    ]
    .into_iter()
    .find_map(|(prefix, severity)| text.strip_prefix(prefix).map(|m| (severity, m)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The shape dxc prints for a failed compile, as the toolchain reports it:
    // its own header line, then clang-style diagnostics.
    const FAILED: &str = "hlsl: dxc failed for main_bindless.hlsl:
shaders\\my dir/frag.hlsl:2:16: error: expected ';' at end of declaration
  float y = 1.0
               ^
               ;
shaders\\my dir/frag.hlsl:3:11: warning: implicit conversion from 'literal float' to 'int' changes value from 3.5 to 3 [-Wliteral-conversion]
  int z = 3.5;
      ~   ^~~
main_bindless.hlsl:1540:12: error: no matching function for call to 'shade'
    return shade(v, od);
           ^~~~~
shaders\\my dir/frag.hlsl:1:8: note: candidate function not viable: requires 1 argument, but 2 were provided
float4 shade(VertexOut v) { return 1.0; }
       ^

";

    #[test]
    fn errors_and_warnings_are_read_with_their_locations() {
        let got = parse(FAILED);
        let heads: Vec<(&str, u32, u32, Severity)> = got
            .iter()
            .map(|d| (d.path.as_str(), d.line, d.column, d.severity))
            .collect();
        assert_eq!(
            heads,
            [
                ("shaders\\my dir/frag.hlsl", 2, 16, Severity::Error),
                ("shaders\\my dir/frag.hlsl", 3, 11, Severity::Warning),
                ("main_bindless.hlsl", 1540, 12, Severity::Error),
            ]
        );
        assert_eq!(got[0].message, "expected ';' at end of declaration");
        assert_eq!(
            got[1].message,
            "implicit conversion from 'literal float' to 'int' changes value from 3.5 to 3 \
             [-Wliteral-conversion]"
        );
    }

    // The excerpt and caret stay with their diagnostic, and a note joins the
    // context of the one above it rather than standing alone.
    #[test]
    fn context_lines_and_notes_belong_to_the_diagnostic_above() {
        let got = parse(FAILED);
        assert_eq!(
            got[0].context,
            "  float y = 1.0\n               ^\n               ;"
        );
        assert!(got[2].context.starts_with("    return shade(v, od);\n"));
        assert!(
            got[2]
                .context
                .contains("frag.hlsl:1:8: note: candidate function not viable"),
            "{}",
            got[2].context
        );
        assert!(
            got[2].context.ends_with("       ^"),
            "trailing blank lines trimmed"
        );
    }

    // A Windows drive colon is part of the path, not the location.
    #[test]
    fn a_drive_letter_path_keeps_its_colon() {
        let got = parse("C:\\worlds\\lake\\water.hlsl:12:3: error: unknown type name 'flaot'\n");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].path, "C:\\worlds\\lake\\water.hlsl");
        assert_eq!((got[0].line, got[0].column), (12, 3));
    }

    // What does not read as a located diagnostic is not one: the toolchain's
    // own header, an unlocated validator error, and output with no diagnostic
    // at all. The unlocated error also closes the context above it.
    #[test]
    fn an_unparseable_line_is_not_a_diagnostic() {
        assert!(parse("hlsl: spirv-cross failed: unsupported capability\n").is_empty());
        assert!(parse("").is_empty());
        let got = parse(
            "a.hlsl:1:1: warning: w\n  x\n  ^\nerror: validation errors\nFunction: main: error\n",
        );
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].context, "  x\n  ^");
        assert!(parse("a.hlsl:x:1: error: not a line number\n").is_empty());
        assert!(parse(":1:1: error: no path\n").is_empty());
    }

    // A diagnostic prints back as dxc printed it.
    #[test]
    fn a_diagnostic_displays_as_dxc_printed_it() {
        let got = parse(FAILED);
        assert_eq!(
            got[0].to_string(),
            "shaders\\my dir/frag.hlsl:2:16: error: expected ';' at end of declaration\n  \
             float y = 1.0\n               ^\n               ;"
        );
    }

    // One error compiled into two programs is reported once; a different
    // message at the same place is not a repeat.
    #[test]
    fn dedup_drops_repeats_and_keeps_the_first_order() {
        let one = parse(FAILED);
        let mut other = parse(FAILED);
        other[0].context = String::from("printed differently");
        other[1].message = String::from("another warning at the same place");
        let merged = dedup(one.iter().cloned().chain(other.iter().cloned()));
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[..3], one[..]);
        assert_eq!(merged[3].message, "another warning at the same place");
        assert!(one[0].same_as(&other[0]));
        assert!(!one[1].same_as(&other[1]));
    }
}
