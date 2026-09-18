//! The slice of serde's attribute grammar that decides an authored key, a
//! variant's name, and what an omitted key deserializes to.

use syn::{Attribute, ExprPath, LitStr, Token};

/// A serde `default` attribute, on a field or a container.
#[derive(Clone)]
pub(crate) enum SerdeDefault {
    /// `default`: the type's own `Default`.
    Trait,
    /// `default = "path"`: the value the function returns.
    Path(ExprPath),
}

// syn prints and compares syntax trees only with its `extra-traits` feature;
// the path's tokens say the same.
impl std::fmt::Debug for SerdeDefault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trait => f.write_str("Trait"),
            Self::Path(path) => write!(f, "Path({})", quote::quote!(#path)),
        }
    }
}

impl PartialEq for SerdeDefault {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Trait, Self::Trait) => true,
            (Self::Path(a), Self::Path(b)) => {
                quote::quote!(#a).to_string() == quote::quote!(#b).to_string()
            }
            _ => false,
        }
    }
}

/// How serde reads one field from authored input.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct FieldAttrs {
    /// The authored key, when `rename` replaces the field name.
    pub(crate) rename: Option<String>,
    /// The field is never read from input.
    pub(crate) skip: bool,
    /// The field's own fields sit at the parent's level.
    pub(crate) flatten: bool,
    /// The field's `default`, when it declares one.
    pub(crate) default: Option<SerdeDefault>,
    /// The field reads through `deserialize_with` (or `with`), so serde does
    /// not treat a missing `Option` as `None`.
    pub(crate) custom_deserialize: bool,
}

/// How serde reads a struct or an enum as a whole.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ContainerAttrs {
    /// The container's `default`, when it declares one.
    pub(crate) default: Option<SerdeDefault>,
    /// The `rename_all` rule for the variants or fields.
    pub(crate) rename_all: Option<String>,
}

/// Parse every `#[serde(...)]` attribute on a field.
pub(crate) fn field_attrs(attrs: &[Attribute]) -> syn::Result<FieldAttrs> {
    let mut out = FieldAttrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") || meta.path.is_ident("skip_deserializing") {
                out.skip = true;
            } else if meta.path.is_ident("flatten") {
                out.flatten = true;
            } else if meta.path.is_ident("rename") {
                out.rename = rename_value(&meta)?.or(out.rename.take());
            } else if meta.path.is_ident("default") {
                out.default = Some(default_value(&meta)?);
            } else if meta.path.is_ident("deserialize_with") || meta.path.is_ident("with") {
                out.custom_deserialize = true;
                skip_meta_value(&meta)?;
            } else {
                skip_meta_value(&meta)?;
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// Parse every `#[serde(...)]` attribute on a struct or an enum.
pub(crate) fn container_attrs(attrs: &[Attribute]) -> syn::Result<ContainerAttrs> {
    let mut out = ContainerAttrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("default") {
                out.default = Some(default_value(&meta)?);
            } else if meta.path.is_ident("rename_all") {
                out.rename_all = rename_value(&meta)?.or(out.rename_all.take());
            } else {
                skip_meta_value(&meta)?;
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// The `rename` on an enum variant, if any.
pub(crate) fn variant_rename(attrs: &[Attribute]) -> syn::Result<Option<String>> {
    let mut out = None;
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("rename") {
                out = rename_value(&meta)?.or(out.take());
            } else {
                skip_meta_value(&meta)?;
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// Apply a serde `rename_all` rule to a PascalCase variant ident, the way
/// serde does.
pub(crate) fn apply_case(ident: &str, rule: Option<&str>) -> syn::Result<String> {
    Ok(match rule {
        None | Some("PascalCase") => ident.to_string(),
        Some("lowercase") => ident.to_lowercase(),
        Some("UPPERCASE") => ident.to_uppercase(),
        Some("snake_case") => snake(ident),
        Some("SCREAMING_SNAKE_CASE") => snake(ident).to_uppercase(),
        Some("kebab-case") => snake(ident).replace('_', "-"),
        Some("SCREAMING-KEBAB-CASE") => snake(ident).replace('_', "-").to_uppercase(),
        Some("camelCase") => {
            let mut chars = ident.chars();
            chars
                .next()
                .map(|c| c.to_ascii_lowercase().to_string() + chars.as_str())
                .unwrap_or_default()
        }
        Some(other) => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("unknown serde rename_all rule `{other}`"),
            ));
        }
    })
}

// serde's snake_case for a variant: an underscore before every uppercase
// letter but the first, then lowercase.
fn snake(ident: &str) -> String {
    let mut out = String::new();
    for (i, c) in ident.char_indices() {
        if i > 0 && c.is_uppercase() {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

// `rename = "x"` or `rename(deserialize = "x", ...)`: the authored (read) name.
fn rename_value(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<Option<String>> {
    if meta.input.peek(Token![=]) {
        return Ok(Some(meta.value()?.parse::<LitStr>()?.value()));
    }
    let mut out = None;
    meta.parse_nested_meta(|inner| {
        let name = inner.value()?.parse::<LitStr>()?.value();
        if inner.path.is_ident("deserialize") {
            out = Some(name);
        }
        Ok(())
    })?;
    Ok(out)
}

// `default` or `default = "path"`.
fn default_value(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<SerdeDefault> {
    if meta.input.peek(Token![=]) {
        let path = meta.value()?.parse::<LitStr>()?.parse::<ExprPath>()?;
        Ok(SerdeDefault::Path(path))
    } else {
        Ok(SerdeDefault::Trait)
    }
}

// Consume an attribute this derive does not read: `key`, `key = value`, or
// `key(...)`.
fn skip_meta_value(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<()> {
    if meta.input.peek(Token![=]) {
        meta.value()?.parse::<syn::Expr>()?;
    } else if meta.input.peek(syn::token::Paren) {
        let content;
        syn::parenthesized!(content in meta.input);
        content.parse::<proc_macro2::TokenStream>()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(tokens: proc_macro2::TokenStream) -> syn::Result<FieldAttrs> {
        let field: syn::Field = syn::parse::Parser::parse2(syn::Field::parse_named, tokens)
            .expect("the test field parses");
        field_attrs(&field.attrs)
    }

    #[test]
    fn a_plain_field_keeps_its_name() {
        let attrs = field(quote::quote! { #[serde(skip_serializing_if = "x")] x: f32 }).unwrap();
        assert_eq!(attrs, FieldAttrs::default());
    }

    #[test]
    fn rename_replaces_the_key_in_either_spelling() {
        let attrs = field(quote::quote! { #[serde(default, rename = "do")] body: u8 }).unwrap();
        assert_eq!(attrs.rename.as_deref(), Some("do"));
        let attrs = field(quote::quote! {
            #[serde(rename(serialize = "out", deserialize = "in"))] x: u8
        })
        .unwrap();
        assert_eq!(attrs.rename.as_deref(), Some("in"));
    }

    #[test]
    fn skip_and_flatten_are_recognized_among_other_attributes() {
        let attrs = field(quote::quote! { #[serde(skip)] id: u32 }).unwrap();
        assert!(attrs.skip);
        let attrs = field(quote::quote! {
            #[serde(default = "seven", skip_deserializing)] id: u32
        })
        .unwrap();
        assert!(attrs.skip);
        let attrs = field(quote::quote! {
            #[serde(flatten, deserialize_with = "de", bound(deserialize = "T: X"))] pose: P
        })
        .unwrap();
        assert!(attrs.flatten && !attrs.skip && attrs.custom_deserialize);
    }

    #[test]
    fn a_field_default_is_read_in_either_form() {
        let attrs = field(quote::quote! { #[serde(default)] x: u8 }).unwrap();
        assert_eq!(attrs.default, Some(SerdeDefault::Trait));
        let attrs = field(quote::quote! { #[serde(default = "self::seven")] x: u8 }).unwrap();
        let Some(SerdeDefault::Path(path)) = attrs.default else {
            panic!("a path default");
        };
        assert_eq!(quote::quote!(#path).to_string(), "self :: seven");
    }

    #[test]
    fn a_non_serde_attribute_is_ignored() {
        let attrs = field(quote::quote! { #[doc = "x"] #[expect(dead_code)] x: u8 }).unwrap();
        assert_eq!(attrs, FieldAttrs::default());
    }

    #[test]
    fn container_default_and_rename_all_are_read() {
        let input: syn::DeriveInput =
            syn::parse_quote! { #[serde(default, rename_all = "snake_case")] struct S {} };
        let attrs = container_attrs(&input.attrs).unwrap();
        assert_eq!(attrs.default, Some(SerdeDefault::Trait));
        assert_eq!(attrs.rename_all.as_deref(), Some("snake_case"));
        let input: syn::DeriveInput =
            syn::parse_quote! { #[serde(deny_unknown_fields)] struct S {} };
        assert_eq!(
            container_attrs(&input.attrs).unwrap(),
            ContainerAttrs::default()
        );
    }

    #[test]
    fn case_rules_match_serde() {
        for (rule, want) in [
            (None, "UltraPerformance"),
            (Some("PascalCase"), "UltraPerformance"),
            (Some("lowercase"), "ultraperformance"),
            (Some("UPPERCASE"), "ULTRAPERFORMANCE"),
            (Some("snake_case"), "ultra_performance"),
            (Some("SCREAMING_SNAKE_CASE"), "ULTRA_PERFORMANCE"),
            (Some("kebab-case"), "ultra-performance"),
            (Some("SCREAMING-KEBAB-CASE"), "ULTRA-PERFORMANCE"),
            (Some("camelCase"), "ultraPerformance"),
        ] {
            assert_eq!(
                apply_case("UltraPerformance", rule).unwrap(),
                want,
                "{rule:?}"
            );
        }
        assert_eq!(apply_case("Fsr3", Some("snake_case")).unwrap(), "fsr3");
        assert!(apply_case("A", Some("Title Case")).is_err());
    }

    #[test]
    fn a_variant_rename_is_read() {
        let variant: syn::Variant = syn::parse_quote! { #[serde(rename = "frag")] Fragment };
        assert_eq!(
            variant_rename(&variant.attrs).unwrap().as_deref(),
            Some("frag")
        );
    }
}
