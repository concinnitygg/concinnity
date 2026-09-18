//! Rustdoc as the derive sees it: one `#[doc = "..."]` attribute per `///`
//! line.

use syn::Attribute;

/// The item's rustdoc: its doc lines joined by newlines, each without the one
/// leading space `///` leaves, and no trailing blank lines. A doc attribute
/// that is not a string literal (`include_str!`) is left out.
pub(crate) fn doc_text(attrs: &[Attribute]) -> String {
    let mut doc = String::new();
    for attr in attrs.iter().filter(|a| a.path().is_ident("doc")) {
        let syn::Meta::NameValue(nv) = &attr.meta else {
            continue;
        };
        let syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(s),
            ..
        }) = &nv.value
        else {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn doc_of(input: syn::DeriveInput) -> String {
        doc_text(&input.attrs)
    }

    #[test]
    fn doc_lines_join_without_the_rustdoc_space() {
        let doc = doc_of(syn::parse_quote! {
            /// First line.
            ///
            ///     indented
            /// Last.
            ///
            struct A;
        });
        assert_eq!(doc, "First line.\n\n    indented\nLast.");
    }

    #[test]
    fn an_undocumented_item_has_an_empty_doc() {
        assert_eq!(doc_of(syn::parse_quote! { #[derive(Debug)] struct A; }), "");
    }

    #[test]
    fn a_non_literal_doc_is_left_out() {
        let doc = doc_of(syn::parse_quote! {
            /// Kept.
            #[doc = include_str!("x.md")]
            struct A;
        });
        assert_eq!(doc, "Kept.");
    }
}
