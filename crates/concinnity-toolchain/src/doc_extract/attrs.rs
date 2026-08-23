// Attribute reading: rustdoc bodies, the serde keys that decide a field's
// serialized name, and the `rename_all` case rules applied to enum variants.

// Join a run of `#[doc = "..."]` attributes back into the source text, dropping
// the single leading space rustdoc inserts. Returned verbatim: what a reference
// strips out of it is the reader's decision, not this crate's.
pub(super) fn extract_doc(attrs: &[syn::Attribute]) -> String {
    let mut doc = String::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let syn::Meta::NameValue(nv) = &attr.meta else {
            continue;
        };
        let syn::Expr::Lit(lit) = &nv.value else {
            continue;
        };
        let syn::Lit::Str(s) = &lit.lit else {
            continue;
        };
        let line = s.value();
        doc.push_str(line.strip_prefix(' ').unwrap_or(&line));
        doc.push('\n');
    }
    while doc.ends_with('\n') {
        doc.pop();
    }
    doc
}

// Collapse a multi-line doc to a single line, for somewhere with room for one.
pub(super) fn collapse_doc(doc: &str) -> String {
    doc.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

// True for a field carrying `#[serde(skip)]` (the exact token, so
// `skip_serializing_if` does not match).
pub(super) fn has_serde_skip(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        if let syn::Meta::List(list) = &attr.meta
            && list
                .tokens
                .to_string()
                .split(',')
                .any(|p| p.trim() == "skip")
        {
            return true;
        }
    }
    false
}

// The string value of a `key = "..."` pair inside any `#[serde(...)]`
// attribute. The `=` boundary check keeps `rename` from matching `rename_all`.
pub(super) fn serde_kv(attrs: &[syn::Attribute], key: &str) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        let syn::Meta::List(list) = &attr.meta else {
            continue;
        };
        let tokens = list.tokens.to_string();
        for part in tokens.split(',') {
            let Some(rest) = part.trim().strip_prefix(key) else {
                continue;
            };
            let Some(rest) = rest.trim_start().strip_prefix('=') else {
                continue;
            };
            let rest = rest.trim();
            if let (Some(a), Some(b)) = (rest.find('"'), rest.rfind('"'))
                && a != b
            {
                return Some(rest[a + 1..b].to_string());
            }
        }
    }
    None
}

// Apply a serde `rename_all` rule to a PascalCase variant ident.
pub(super) fn apply_case(ident: &str, rule: Option<&str>) -> String {
    match rule {
        None | Some("PascalCase") => ident.to_string(),
        Some("lowercase") => ident.to_lowercase(),
        Some("UPPERCASE") => ident.to_uppercase(),
        Some("snake_case") => split_words(ident).join("_"),
        Some("SCREAMING_SNAKE_CASE") => split_words(ident).join("_").to_uppercase(),
        Some("kebab-case") => split_words(ident).join("-"),
        Some("camelCase") => {
            let mut s = String::new();
            for (i, word) in split_words(ident).iter().enumerate() {
                if i == 0 {
                    s.push_str(word);
                    continue;
                }
                let mut chars = word.chars();
                if let Some(first) = chars.next() {
                    s.push(first.to_ascii_uppercase());
                    s.push_str(chars.as_str());
                }
            }
            s
        }
        Some(_) => ident.to_string(),
    }
}

// Split a PascalCase ident into lowercase words: "VertexInstanced" -> [vertex,
// instanced]. Acronym runs are not special-cased (none occur in asset enums).
fn split_words(ident: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for (i, c) in ident.chars().enumerate() {
        if c.is_uppercase() && i != 0 && !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
        cur.push(c.to_ascii_lowercase());
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs_of(src: &str) -> Vec<syn::Attribute> {
        let file: syn::File = syn::parse_str(src).expect("parse");
        match &file.items[0] {
            syn::Item::Struct(s) => s.attrs.clone(),
            _ => panic!("expected a struct"),
        }
    }

    fn field_attrs(src: &str) -> Vec<syn::Attribute> {
        let file: syn::File = syn::parse_str(src).expect("parse");
        let syn::Item::Struct(s) = &file.items[0] else {
            panic!("expected a struct");
        };
        s.fields.iter().next().expect("one field").attrs.clone()
    }

    #[test]
    fn doc_lines_join_without_the_rustdoc_space() {
        let attrs = attrs_of("/// First line.\n/// Second line.\npub struct A;");
        assert_eq!(extract_doc(&attrs), "First line.\nSecond line.");
    }

    #[test]
    fn doc_is_empty_without_rustdoc() {
        assert_eq!(extract_doc(&attrs_of("pub struct A;")), "");
    }

    #[test]
    fn collapse_doc_folds_blank_lines_away() {
        assert_eq!(collapse_doc("One.\n\n  Two.  \n"), "One. Two.");
    }

    #[test]
    fn serde_skip_matches_only_the_bare_token() {
        assert!(has_serde_skip(&field_attrs(
            "struct A { #[serde(skip)] a: u32 }"
        )));
        assert!(has_serde_skip(&field_attrs(
            "struct A { #[serde(default, skip)] a: u32 }"
        )));
        assert!(!has_serde_skip(&field_attrs(
            "struct A { #[serde(skip_serializing_if = \"Option::is_none\")] a: u32 }"
        )));
        assert!(!has_serde_skip(&field_attrs("struct A { a: u32 }")));
    }

    #[test]
    fn serde_rename_does_not_match_rename_all() {
        let attrs = attrs_of("#[serde(rename_all = \"snake_case\")]\npub struct A;");
        assert_eq!(serde_kv(&attrs, "rename"), None);
        assert_eq!(
            serde_kv(&attrs, "rename_all"),
            Some("snake_case".to_string())
        );
    }

    #[test]
    fn serde_rename_reads_the_field_key() {
        let attrs = field_attrs("struct A { #[serde(rename = \"type\")] kind: u32 }");
        assert_eq!(serde_kv(&attrs, "rename"), Some("type".to_string()));
    }

    #[test]
    fn case_rules_cover_every_serde_spelling() {
        for (rule, want) in [
            (None, "VertexInstanced"),
            (Some("PascalCase"), "VertexInstanced"),
            (Some("lowercase"), "vertexinstanced"),
            (Some("UPPERCASE"), "VERTEXINSTANCED"),
            (Some("snake_case"), "vertex_instanced"),
            (Some("SCREAMING_SNAKE_CASE"), "VERTEX_INSTANCED"),
            (Some("kebab-case"), "vertex-instanced"),
            (Some("camelCase"), "vertexInstanced"),
            (Some("unrecognised"), "VertexInstanced"),
        ] {
            assert_eq!(apply_case("VertexInstanced", rule), want, "rule {rule:?}");
        }
    }
}
