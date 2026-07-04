// GitHub-style anchor slug for a heading: lowercase, alphanumerics kept,
// runs of spaces/hyphens/underscores collapsed to one hyphen, everything
// else dropped. `[link](#the-wood)` targets the heading `# The Wood`.
pub(crate) fn slug(text: &str) -> String {
    let mut out = String::new();
    for c in text.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if (c == ' ' || c == '-' || c == '_') && !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_string()
}

// A variable name: lowercase alphanumerics, `_`, `-` (the same alphabet as
// heading slugs, so scripts read consistently with anchors).
pub(super) fn parse_flag(word: &str) -> Result<String, String> {
    if word.is_empty()
        || !word
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
    {
        return Err(format!(
            "'{}' is not a variable name (lowercase letters, digits, `_`, `-`)",
            word
        ));
    }
    Ok(word.to_string())
}

// An integer literal in a script line (negatives allowed).
pub(super) fn parse_int(word: &str) -> Result<i32, String> {
    word.parse::<i32>()
        .map_err(|_| format!("'{}' is not an integer", word))
}

// Split `a: 1, b: [2, 3]` on commas outside brackets.
pub(super) fn split_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

pub(super) fn unquote(s: &str) -> &str {
    let s = s.trim();
    s.strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(s)
}

// Wrap text to the dialog column on word boundaries, preserving authored
// hard breaks.
pub(crate) fn wrap_text(text: &str, columns: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    for source_line in text.split('\n') {
        let mut cur = String::new();
        for word in source_line.split_whitespace() {
            if cur.is_empty() {
                cur = word.to_string();
            } else if cur.chars().count() + 1 + word.chars().count() <= columns {
                cur.push(' ');
                cur.push_str(word);
            } else {
                lines.push(std::mem::take(&mut cur));
                cur = word.to_string();
            }
        }
        lines.push(cur);
    }
    lines.join("\n")
}

pub(super) const AUDIO_EXTENSIONS: [&str; 4] = ["ogg", "wav", "mp3", "flac"];
pub(super) const IMAGE_EXTENSIONS: [&str; 3] = ["png", "jpg", "jpeg"];

pub(super) fn file_extension(target: &str) -> String {
    std::path::Path::new(target)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default()
}
